"""FPGA-accelerated NeuralILT using tiled MiniUNet inference.

Follows the ModelILT interface from lithobench/model.py so it can be used
with the existing training and evaluation infrastructure.

Standalone:
    from lithobench.ilt.fpga_neuralilt import FPGANeuralILT
    model = FPGANeuralILT()
    model.load("saved/fpga/mini_unet_best.pth")
    mask = model.run(target_512x512)
"""

import glob
import os
import sys
sys.path.append(".")
import time
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader

from pycommon.settings import REALTYPE, DEVICE
import pylitho.exact as litho
from lithobench.model import ModelILT


class TiledInference:
    """Manages tiled inference of PenumbraUNet over a full-size image.

    The 512x512 input is reflection-padded, split into overlapping 64x64 tiles
    (stride 32), and reassembled after inference by keeping only the central
    32x32 crop of each tile output.

    Args:
        tile_size: Size of each square tile (default 64).
        overlap: Overlap in pixels on each side (default 16).
        image_size: Full input image size (default 512).
    """

    def __init__(self, tile_size=64, overlap=16, image_size=512):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - 2 * overlap  # 32
        self.image_size = image_size

    @property
    def num_tiles_per_side(self):
        return self.image_size // self.stride

    def extract_tiles(self, image):
        padded = F.pad(image, [self.overlap] * 4, mode="reflect")
        tiles, positions = [], []
        n = self.num_tiles_per_side
        for i in range(n):
            for j in range(n):
                r, c = i * self.stride, j * self.stride
                tiles.append(padded[:, :, r:r + self.tile_size, c:c + self.tile_size])
                positions.append((i, j))
        return torch.cat(tiles, dim=0), positions

    def reassemble(self, tile_outputs, positions):
        n = self.num_tiles_per_side
        crops = tile_outputs[
            :, :,
            self.overlap:self.tile_size - self.overlap,
            self.overlap:self.tile_size - self.overlap,
        ]
        crops = crops.reshape(n, n, 1, self.stride, self.stride)
        crops = crops.permute(2, 0, 3, 1, 4).contiguous()
        return crops.reshape(1, 1, n * self.stride, n * self.stride)

    def forward(self, image, model, batch_size=64):
        """Differentiable tiled forward pass (keeps gradients for training)."""
        tiles, positions = self.extract_tiles(image)
        outputs = [model(tiles[s:s + batch_size]) for s in range(0, tiles.shape[0], batch_size)]
        return self.reassemble(torch.cat(outputs, dim=0), positions)

    def run(self, image, model, batch_size=64, device=None):
        """Non-differentiable tiled inference (no gradients, for deployment)."""
        if device is None:
            device = image.device
        tiles, positions = self.extract_tiles(image)
        outputs = []
        for s in range(0, tiles.shape[0], batch_size):
            with torch.no_grad():
                outputs.append(model(tiles[s:s + batch_size].to(device)).cpu())
        return self.reassemble(torch.cat(outputs, dim=0), positions)


class PenumbraUNet(nn.Module):
    """
    Scaled-down UNet for FPGA deployment via hls4ml.

    Architecture mirrors the original NeuralILT UNet but with reduced channel
    widths for on-chip FPGA resource constraints.

    Original: 1 -> 64 -> 128 -> 256 -> 512  (~7.8M params)
    This:     1 ->  8 ->  16 ->  32 ->  64  (~122K params)

    Uses nn.Upsample(mode='nearest') for upsampling — exports to ONNX Resize
    which hls4ml supports — and explicit nn.Sequential blocks for torch.fx tracing.

    Input:  (B, 1, 64, 64)  — single-channel tile
    Output: (B, 1, 64, 64)  — predicted mask tile
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = nn.Sequential(
            nn.Conv2d(16 + 8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        c3 = self.conv3(x)
        x = self.pool3(c3)
        x = self.conv4(x)
        x = self.deconv3(torch.cat([self.up3(x), c3], dim=1))
        x = self.deconv2(torch.cat([self.up2(x), c2], dim=1))
        x = self.deconv1(torch.cat([self.up1(x), c1], dim=1))
        return self.sigmoid(self.final_conv(x))


def fuse_batchnorm(model):
    """Fuse BatchNorm into preceding Conv2d for hls4ml export.

    Returns a new model with BN folded into conv weights/biases.
    """
    import copy
    fused = copy.deepcopy(model)
    fused.eval()
    for attr_name, module in list(fused.named_children()):
        if not isinstance(module, nn.Sequential):
            continue
        layers = list(module.children())
        new_layers = []
        i = 0
        while i < len(layers):
            if (i + 1 < len(layers)
                    and isinstance(layers[i], nn.Conv2d)
                    and isinstance(layers[i + 1], nn.BatchNorm2d)):
                new_layers.append(_fuse_conv_bn(layers[i], layers[i + 1]))
                i += 2
            else:
                new_layers.append(layers[i])
                i += 1
        setattr(fused, attr_name, nn.Sequential(*new_layers))
    return fused


def _fuse_conv_bn(conv, bn):
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        fused_bias = (conv.bias - bn.running_mean if conv.bias is not None
                      else -bn.running_mean) * scale + bn.bias
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding,
                          dilation=conv.dilation, groups=conv.groups, bias=True)
        fused.weight.copy_(fused_weight)
        fused.bias.copy_(fused_bias)
    return fused


class FPGANeuralILT(ModelILT):
    """FPGA-deployable NeuralILT using tiled MiniUNet.

    In CPU/GPU mode (use_fpga=False), this runs the MiniUNet with tiled
    inference entirely in PyTorch. With use_fpga=True, each tile is sent
    to the F2 FPGA via XDMA DMA — see hls/host/driver.py.

    Args:
        size: Full image size (default 512x512, matching NeuralILT).
        tile_size: Tile size for inference (default 64).
        overlap: Overlap pixels per side (default 16).
        use_fpga: If True, route run() through the FPGA XDMA driver.
            Falls back to PyTorch if the FPGA device cannot be opened.
    """

    def __init__(self, size=(512, 512), tile_size=64, overlap=16, use_fpga=False):
        super().__init__(size=size, name="FPGANeuralILT")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.net = PenumbraUNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.tiler = TiledInference(
            tile_size=tile_size, overlap=overlap, image_size=size[0]
        )
        self.use_fpga = use_fpga
        self._fpga_driver = None
        if use_fpga:
            self._init_fpga()

    def _init_fpga(self):
        """Open the FPGA XDMA driver; fall back to PyTorch on failure."""
        try:
            from hls.host.driver import FPGADriver
            self._fpga_driver = FPGADriver()
            self._fpga_driver.open()
            print("[FPGANeuralILT] FPGA hardware mode enabled.")
        except Exception as e:
            print(f"[FPGANeuralILT] WARNING: could not open FPGA driver: {e}")
            print("[FPGANeuralILT] Falling back to PyTorch CPU/GPU mode.")
            self._fpga_driver = None
            self.use_fpga = False

    def __del__(self):
        if self._fpga_driver is not None:
            try:
                self._fpga_driver.close()
            except Exception:
                pass

    def _load_teacher(self):
        """Load a NeuralILT UNet teacher for knowledge distillation.

        Reads the teacher weights path from the PENUMBRA_TEACHER environment
        variable. Returns None if the variable is unset or the file is missing,
        in which case pretrain() falls back to plain supervised training.

        Usage::

            PENUMBRA_TEACHER=saved/MetalSet_NeuralILT/net.pth \\
                python3 lithobench/train.py -m lithobench/ilt/fpga_neuralilt.py ...
        """
        teacher_path = os.environ.get("PENUMBRA_TEACHER", "")
        if not teacher_path or not os.path.exists(teacher_path):
            return None
        from lithobench.ilt.neuralilt import UNet
        teacher = UNet()
        state = torch.load(teacher_path, map_location="cpu")
        teacher.load_state_dict(state)
        device = next(self.net.parameters()).device
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[FPGANeuralILT] Loaded distillation teacher from {teacher_path}")
        return teacher

    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=4,
                 checkpoint_dir=None, resume_checkpoint=None):
        """Pretraining on full-size images using tiled crops.

        If PENUMBRA_TEACHER is set to a NeuralILT weights path, uses knowledge
        distillation (alpha-weighted teacher + GT loss, alpha decays over epochs).
        Otherwise falls back to plain supervised MSE on tile crops.

        Args:
            checkpoint_dir: If set, save checkpoint_latest.pth here after each epoch.
            resume_checkpoint: Path to a checkpoint to resume from.
        """
        teacher = self._load_teacher()
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        tile_size = self.tiler.tile_size
        start_epoch = 0

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location="cpu")
            self.net.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None:
                sched.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[FPGANeuralILT] Resumed pretrain from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            print(f"[Pre-Epoch {epoch}] Training")
            self.net.train()
            progress = tqdm(train_loader)
            for target, label in progress:
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()

                tiles_t, tiles_l = self._random_crops(target, label, tile_size, n=8)
                mask = self.net(tiles_t)

                if teacher is not None:
                    alpha = 0.7 * (1.0 - epoch / epochs)
                    with torch.no_grad():
                        teacher_out = teacher(tiles_t)
                    loss = alpha * F.mse_loss(mask, teacher_out) + (1 - alpha) * F.mse_loss(mask, tiles_l)
                else:
                    loss = F.mse_loss(mask, tiles_l)

                opt.zero_grad()
                loss.backward()
                opt.step()
                progress.set_postfix(loss=loss.item())

            print(f"[Pre-Epoch {epoch}] Testing")
            self.net.eval()
            losses = []
            progress = tqdm(val_loader)
            for target, label in progress:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                        label = label.cuda()
                    tiles_t, tiles_l = self._random_crops(target, label, tile_size, n=4)
                    mask = self.net(tiles_t)
                    loss = F.mse_loss(mask, tiles_l)
                    losses.append(loss.item())
                    progress.set_postfix(loss=loss.item())
            print(f"[Pre-Epoch {epoch}] loss = {np.mean(losses)}")

            if epoch == epochs // 2:
                sched.step()

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
                torch.save({
                    "epoch": epoch,
                    "phase": "pretrain",
                    "model_state": self.net.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "best_val": float("inf"),
                }, ckpt_path)

    def train(self, train_loader, val_loader, epochs=1, batch_size=4,
              checkpoint_dir=None, resume_checkpoint=None):
        """Physics-informed training using the lithography simulator.

        For each full-size target image:
          1. Run MiniUNet on all tiles (differentiably)
          2. Reassemble into a full 512x512 mask
          3. Run litho sim on the full mask
          4. Backpropagate physics loss through tiling back to MiniUNet

        This matches the original NeuralILT training loop
        (neuralilt.py:171-221) where the loss is:
          l2loss  = MSE(printedNom, target)
          cpxloss = MSE(printedMax, printedMin)

        Args:
            checkpoint_dir: If set, save checkpoint_latest.pth here after each epoch.
            resume_checkpoint: Path to a checkpoint to resume from.
        """
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        device = next(self.net.parameters()).device
        start_epoch = 0

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location="cpu")
            self.net.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None:
                sched.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[FPGANeuralILT] Resumed train from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            print(f"[Epoch {epoch}] Training (physics-informed)")
            self.net.train()
            progress = tqdm(train_loader)
            for target, label in progress:
                if torch.cuda.is_available():
                    target = target.cuda()

                # Process one image at a time through full tiled pipeline
                for b in range(target.shape[0]):
                    img = target[b : b + 1]  # (1, 1, H, W)

                    # Differentiable tiled forward: tiles -> MiniUNet -> reassemble
                    mask_full = self.tiler.forward(img, self.net, batch_size=64)
                    # mask_full: (1, 1, 512, 512) with gradients

                    # Run lithography simulator on the full assembled mask
                    mask_sq = mask_full.squeeze(1)  # (1, 512, 512)
                    printedNom, printedMax, printedMin = self.simLitho(mask_sq)

                    l2loss = F.mse_loss(printedNom.unsqueeze(1), img)
                    cpxloss = F.mse_loss(printedMax, printedMin)
                    loss = l2loss + cpxloss

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                progress.set_postfix(l2=l2loss.item(), cpx=cpxloss.item())

            print(f"[Epoch {epoch}] Validation")
            self.net.eval()
            l2losses = []
            cpxlosses = []
            progress = tqdm(val_loader)
            for target, label in progress:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        target = target.cuda()
                    for b in range(target.shape[0]):
                        img = target[b : b + 1]
                        mask_full = self.tiler.forward(img, self.net, batch_size=64)
                        mask_sq = mask_full.squeeze(1)
                        printedNom, printedMax, printedMin = self.simLitho(mask_sq)
                        l2loss = F.mse_loss(printedNom.unsqueeze(1), img)
                        cpxloss = F.mse_loss(printedMax, printedMin)
                        l2losses.append(l2loss.item())
                        cpxlosses.append(cpxloss.item())
                    progress.set_postfix(l2=l2loss.item(), cpx=cpxloss.item())
            print(f"[Epoch {epoch}] L2={np.mean(l2losses):.6f} cpx={np.mean(cpxlosses):.6f}")

            if epoch == epochs // 2:
                sched.step()

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
                torch.save({
                    "epoch": epoch,
                    "phase": "physics",
                    "model_state": self.net.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "best_val": float("inf"),
                }, ckpt_path)

    def run(self, target):
        """Process a full-size target image via tiled MiniUNet inference.

        Args:
            target: Tensor of shape (1, 1, H, W) or (B, 1, H, W).

        Returns:
            Tensor of shape (H, W) — predicted mask for the first image in batch.
        """
        self.net.eval()
        if self.use_fpga and self._fpga_driver is not None:
            return self._run_fpga(target)
        device = next(self.net.parameters()).device
        output = self.tiler.run(target, self.net, batch_size=64, device=device)
        return output[0, 0].detach()

    def _run_fpga(self, target):
        """FPGA tiled inference: extract tiles → FPGA DMA → reassemble.

        Processes all tiles sequentially (one FPGA invocation per tile).
        Uses the same TiledInference.extract_tiles / reassemble as the
        PyTorch path so outputs are numerically comparable.

        Args:
            target: Tensor of shape (1, 1, H, W).

        Returns:
            Tensor of shape (H, W) on CPU.
        """
        tiles, positions = self.tiler.extract_tiles(target.cpu())
        results = []
        for i in range(tiles.shape[0]):
            tile_np = tiles[i, 0].numpy()                        # (64, 64) float32
            out_np = self._fpga_driver.process_tile(tile_np)     # (64, 64) float32
            results.append(torch.from_numpy(out_np).unsqueeze(0))  # (1, 64, 64)
        tile_outputs = torch.stack(results, dim=0)               # (N, 1, 64, 64)
        full = self.tiler.reassemble(tile_outputs, positions)    # (1, 1, H, W)
        return full[0, 0].detach()

    def save(self, filenames):
        filename = filenames[0] if isinstance(filenames, list) else filenames
        torch.save(self.net.state_dict(), filename)

    def load(self, filenames):
        filename = filenames[0] if isinstance(filenames, list) else filenames
        state = torch.load(filename, map_location="cpu")
        self.net.load_state_dict(state)
        if torch.cuda.is_available():
            self.net = self.net.cuda()

    @staticmethod
    def _random_crops(target, label, tile_size, n=8):
        """Extract n random tile crops from a batch of images.

        Args:
            target: (B, 1, H, W)
            label: (B, 1, H, W)
            tile_size: side length of square crop
            n: crops per image

        Returns:
            (tiles_target, tiles_label) each of shape (B*n, 1, tile_size, tile_size)
        """
        B, C, H, W = target.shape
        tiles_t = []
        tiles_l = []
        for _ in range(n):
            top = random.randint(0, H - tile_size)
            left = random.randint(0, W - tile_size)
            tiles_t.append(target[:, :, top : top + tile_size, left : left + tile_size])
            tiles_l.append(label[:, :, top : top + tile_size, left : left + tile_size])
        return torch.cat(tiles_t, dim=0), torch.cat(tiles_l, dim=0)
