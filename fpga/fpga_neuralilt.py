"""FPGA-accelerated NeuralILT using tiled MiniUNet inference.

Follows the ModelILT interface from lithobench/model.py so it can be used
with the existing training and evaluation infrastructure.

Usage with lithobench/train.py:
    python3 lithobench/train.py -m fpga/fpga_neuralilt.py -a FPGANeuralILT \
        -i 512 -t ILT -s MetalSet -n 16 -b 32

Standalone:
    from fpga.fpga_neuralilt import FPGANeuralILT
    model = FPGANeuralILT()
    model.load("saved/fpga/mini_unet_best.pth")
    mask = model.run(target_512x512)
"""

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
from lithobench.model import ModelILT
from fpga.mini_unet import MiniUNet
from fpga.tiled_inference import TiledInference


class FPGANeuralILT(ModelILT):
    """FPGA-deployable NeuralILT using tiled MiniUNet.

    In CPU/GPU mode, this runs the MiniUNet with tiled inference entirely in
    PyTorch. For actual FPGA deployment, the model forward pass would be
    replaced with DMA transfers to/from the F2 FPGA — see fpga/host/ for
    the driver interface.

    Args:
        size: Full image size (default 512x512, matching NeuralILT).
        tile_size: Tile size for inference (default 64).
        overlap: Overlap pixels per side (default 16).
    """

    def __init__(self, size=(512, 512), tile_size=64, overlap=16):
        super().__init__(size=size, name="FPGANeuralILT")
        self.net = MiniUNet()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.tiler = TiledInference(
            tile_size=tile_size, overlap=overlap, image_size=size[0]
        )

    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=4):
        """Supervised pretraining on full-size images using tiled crops.

        For each full-size (target, mask) pair, random tile crops are extracted
        and the MiniUNet is trained with MSE loss.
        """
        opt = optim.Adam(self.net.parameters(), lr=1e-3)
        sched = lr_sched.StepLR(opt, 1, gamma=0.1)
        tile_size = self.tiler.tile_size

        for epoch in range(epochs):
            print(f"[Pre-Epoch {epoch}] Training")
            self.net.train()
            progress = tqdm(train_loader)
            for target, label in progress:
                if torch.cuda.is_available():
                    target = target.cuda()
                    label = label.cuda()

                # Extract random tile crops from the batch
                tiles_t, tiles_l = self._random_crops(target, label, tile_size, n=8)
                mask = self.net(tiles_t)
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

    def train(self, train_loader, val_loader, epochs=1, batch_size=4):
        """Training phase — same as pretrain for MiniUNet (no litho sim).

        The full NeuralILT uses a lithography simulator in training, but
        running litho sim on individual tiles is not meaningful due to
        boundary effects. Instead, we continue supervised training.
        """
        self.pretrain(train_loader, val_loader, epochs=epochs, batch_size=batch_size)

    def run(self, target):
        """Process a full-size target image via tiled MiniUNet inference.

        Args:
            target: Tensor of shape (1, 1, H, W) or (B, 1, H, W).

        Returns:
            Tensor of shape (H, W) — predicted mask for the first image in batch.
        """
        self.net.eval()
        device = next(self.net.parameters()).device
        output = self.tiler.run(target, self.net, batch_size=64, device=device)
        return output[0, 0].detach()

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
