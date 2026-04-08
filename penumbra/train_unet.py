"""Knowledge distillation training for MiniUNet.

Phase 1: Distill from pretrained full-size NeuralILT UNet on random 64x64 crops.
Phase 2: Physics-informed training with lithography simulator.

Usage:
    python3 fpga/train_mini.py --benchmark MetalSet --epochs 16 --batch_size 32
    python3 fpga/train_mini.py --benchmark MetalSet --teacher saved/MetalSet_NeuralILT/net.pth

Resuming after interruption (use inside tmux for long runs):
    python3 fpga/train_mini.py --benchmark MetalSet --epochs 16 \\
        --resume saved/fpga/checkpoint_latest.pth
"""

import glob
import os
import sys
sys.path.append(".")
import argparse
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, Dataset

from pycommon.settings import REALTYPE, DEVICE
from lithobench.dataset import loadersILT
from lithobench.ilt.neuralilt import UNet
import pylitho.exact as litho
from fpga.penumbra_unet import PenumbraUNet
from fpga.tiled_inference import TiledInference


class TileCropDataset(Dataset):
    """Wraps an ILT dataset to yield random 64x64 tile crops.

    Each __getitem__ call returns a random crop from the underlying full-size
    image pair (target, mask).
    """

    def __init__(self, base_dataset, tile_size=64, crops_per_image=16):
        self.base = base_dataset
        self.tile_size = tile_size
        self.crops_per_image = crops_per_image

    def __len__(self):
        return len(self.base) * self.crops_per_image

    def __getitem__(self, index):
        img_idx = index // self.crops_per_image
        target, mask = self.base[img_idx]  # (1, H, W) each

        _, h, w = target.shape
        ts = self.tile_size
        top = random.randint(0, h - ts)
        left = random.randint(0, w - ts)

        target_tile = target[:, top : top + ts, left : left + ts]
        mask_tile = mask[:, top : top + ts, left : left + ts]

        # Random flips for augmentation
        if random.random() > 0.5:
            target_tile = target_tile.flip(1)
            mask_tile = mask_tile.flip(1)
        if random.random() > 0.5:
            target_tile = target_tile.flip(2)
            mask_tile = mask_tile.flip(2)

        return target_tile, mask_tile


def load_teacher(weights_path, device):
    """Load the pretrained full-size NeuralILT UNet as teacher."""
    teacher = UNet()
    state = torch.load(weights_path, map_location="cpu")
    teacher.load_state_dict(state)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def distill_epoch(student, teacher, loader, optimizer, device, alpha=0.7):
    """Run one epoch of knowledge distillation training.

    Loss = alpha * MSE(student, teacher_tile) + (1 - alpha) * MSE(student, gt_tile)
    """
    student.train()
    total_loss = 0.0
    count = 0
    progress = tqdm(loader, desc="Distill")

    for target_tile, mask_tile in progress:
        target_tile = target_tile.to(device)
        mask_tile = mask_tile.to(device)

        student_out = student(target_tile)

        with torch.no_grad():
            teacher_out = teacher(target_tile)

        loss_distill = F.mse_loss(student_out, teacher_out)
        loss_gt = F.mse_loss(student_out, mask_tile)
        loss = alpha * loss_distill + (1.0 - alpha) * loss_gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        progress.set_postfix(
            loss=loss.item(),
            distill=loss_distill.item(),
            gt=loss_gt.item(),
        )

    return total_loss / max(count, 1)


def pretrain_epoch(student, loader, optimizer, device):
    """Run one epoch of direct supervised pretraining (no teacher)."""
    student.train()
    total_loss = 0.0
    count = 0
    progress = tqdm(loader, desc="Pretrain")

    for target_tile, mask_tile in progress:
        target_tile = target_tile.to(device)
        mask_tile = mask_tile.to(device)

        student_out = student(target_tile)
        loss = F.mse_loss(student_out, mask_tile)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        progress.set_postfix(loss=loss.item())

    return total_loss / max(count, 1)


def validate(student, loader, device):
    """Compute validation MSE loss."""
    student.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for target_tile, mask_tile in loader:
            target_tile = target_tile.to(device)
            mask_tile = mask_tile.to(device)
            out = student(target_tile)
            loss = F.mse_loss(out, mask_tile)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def physics_epoch(student, tiler, sim, loader, optimizer, device):
    """Run one epoch of physics-informed training using the litho simulator.

    For each full-size target image:
      1. Run MiniUNet on all tiles (differentiably via tiler.forward)
      2. Reassemble into a full mask
      3. Run litho sim on the assembled mask
      4. Backprop: litho loss -> reassembly -> tiles -> MiniUNet weights

    This is the same physics loss used in the original NeuralILT
    (neuralilt.py:185-188):
      l2loss  = MSE(printedNom, target)   — print fidelity
      cpxloss = MSE(printedMax, printedMin) — process variation robustness
    """
    student.train()
    total_l2 = 0.0
    total_cpx = 0.0
    count = 0
    progress = tqdm(loader, desc="Physics")

    for target, _label in progress:
        target = target.to(device)

        # Process one image at a time (full tiled pipeline is memory-intensive)
        for b in range(target.shape[0]):
            img = target[b : b + 1]  # (1, 1, H, W)

            # Differentiable tiled forward pass
            mask_full = tiler.forward(img, student, batch_size=64)

            # Litho simulation on the assembled full-size mask
            mask_sq = mask_full.squeeze(1)  # (1, H, W)
            printedNom, printedMax, printedMin = sim(mask_sq)

            l2loss = F.mse_loss(printedNom.unsqueeze(1), img)
            cpxloss = F.mse_loss(printedMax, printedMin)
            loss = l2loss + cpxloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_l2 += l2loss.item()
            total_cpx += cpxloss.item()
            count += 1

        progress.set_postfix(l2=l2loss.item(), cpx=cpxloss.item())

    return total_l2 / max(count, 1), total_cpx / max(count, 1)


def save_checkpoint(path, epoch, phase, model, optimizer, best_val, scheduler=None):
    """Save full training state so training can be resumed."""
    torch.save({
        "epoch": epoch,
        "phase": phase,  # "distill" | "physics"
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
    }, path)


def _cleanup_old_checkpoints(output_dir, prefix, keep=3):
    """Delete old per-epoch checkpoints, keeping only the `keep` most recent."""
    pattern = os.path.join(output_dir, f"{prefix}_ep*.pth")
    files = sorted(glob.glob(pattern))
    for f in files[:-keep]:
        os.remove(f)


def main():
    parser = argparse.ArgumentParser(description="Train MiniUNet via knowledge distillation")
    parser.add_argument("--benchmark", "-s", default="MetalSet", type=str)
    parser.add_argument("--teacher", "-t", default="", type=str,
                        help="Path to pretrained teacher UNet weights. If empty, train without distillation.")
    parser.add_argument("--epochs", "-n", default=16, type=int)
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    parser.add_argument("--tile_size", default=64, type=int)
    parser.add_argument("--crops_per_image", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--alpha", default=0.7, type=float,
                        help="Distillation weight (0=GT only, 1=teacher only)")
    parser.add_argument("--physics_epochs", "-p", default=4, type=int,
                        help="Epochs of physics-informed training after distillation (0 to skip)")
    parser.add_argument("--output", "-o", default="saved/fpga", type=str)
    parser.add_argument("--njobs", "-j", default=8, type=int)
    parser.add_argument("--resume", default="", type=str,
                        help="Path to checkpoint_latest.pth to resume from")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = DEVICE

    # Load base dataset at 512x512 (matching NeuralILT)
    train_loader_full, val_loader_full = loadersILT(
        args.benchmark, (512, 512), batch_size=1, njobs=args.njobs
    )

    # Wrap into tile-crop datasets
    train_tile = TileCropDataset(
        train_loader_full.dataset,
        tile_size=args.tile_size,
        crops_per_image=args.crops_per_image,
    )
    val_tile = TileCropDataset(
        val_loader_full.dataset,
        tile_size=args.tile_size,
        crops_per_image=args.crops_per_image // 4,
    )
    train_loader = DataLoader(
        train_tile, batch_size=args.batch_size, shuffle=True,
        num_workers=args.njobs, drop_last=True,
    )
    val_loader = DataLoader(
        val_tile, batch_size=args.batch_size, shuffle=False,
        num_workers=args.njobs, drop_last=False,
    )

    # Create student
    student = PenumbraUNet().to(device)
    num_params = sum(p.numel() for p in student.parameters())
    print(f"MiniUNet parameters: {num_params:,}")

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load teacher if available
    teacher = None
    if args.teacher and os.path.exists(args.teacher):
        print(f"Loading teacher from {args.teacher}")
        teacher = load_teacher(args.teacher, device)

    # Resume from checkpoint if requested
    start_phase = "distill"
    distill_start_epoch = 0
    best_val = float("inf")
    resume_ckpt = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(resume_ckpt["model_state"])
        start_phase = resume_ckpt["phase"]
        best_val = resume_ckpt["best_val"]
        if start_phase == "distill":
            distill_start_epoch = resume_ckpt["epoch"] + 1
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if resume_ckpt["scheduler_state"] is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        print(f"  Resumed at phase={start_phase}, next epoch={distill_start_epoch}")

    # ------------------------------------------------------------------ #
    # Phase 1: Distillation / supervised pretraining
    # ------------------------------------------------------------------ #
    if start_phase == "distill":
        for epoch in range(distill_start_epoch, args.epochs):
            if teacher is not None:
                # Decay alpha: start high (rely on teacher), decrease over training
                alpha = args.alpha * (1.0 - epoch / args.epochs)
                train_loss = distill_epoch(
                    student, teacher, train_loader, optimizer, device, alpha=alpha
                )
                print(f"[Epoch {epoch}] Distill loss={train_loss:.6f} alpha={alpha:.3f}")
            else:
                train_loss = pretrain_epoch(student, train_loader, optimizer, device)
                print(f"[Epoch {epoch}] Pretrain loss={train_loss:.6f}")

            val_loss = validate(student, val_loader, device)
            print(f"[Epoch {epoch}] Val loss={val_loss:.6f}")

            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                save_path = os.path.join(args.output, "mini_unet_best.pth")
                torch.save(student.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")

            # Per-epoch checkpoint + rolling latest
            ckpt_path = os.path.join(args.output, f"ckpt_distill_ep{epoch}.pth")
            save_checkpoint(ckpt_path, epoch, "distill", student, optimizer, best_val, scheduler)
            save_checkpoint(os.path.join(args.output, "checkpoint_latest.pth"),
                            epoch, "distill", student, optimizer, best_val, scheduler)
            _cleanup_old_checkpoints(args.output, "ckpt_distill", keep=3)

    # ------------------------------------------------------------------ #
    # Phase 2: Physics-informed training with lithography simulator
    # ------------------------------------------------------------------ #
    if args.physics_epochs > 0:
        print(f"\n=== Phase 2: Physics-informed training ({args.physics_epochs} epochs) ===")
        print("Loading lithography simulator...")
        sim = litho.LithoSim("./config/lithosimple.txt")
        tiler = TiledInference(tile_size=args.tile_size, overlap=16, image_size=512)

        physics_optimizer = optim.Adam(student.parameters(), lr=args.lr * 0.1)
        physics_scheduler = lr_sched.StepLR(physics_optimizer, 1, gamma=0.1)

        physics_start_epoch = 0
        if start_phase == "physics" and resume_ckpt is not None:
            physics_start_epoch = resume_ckpt["epoch"] + 1
            physics_optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if resume_ckpt["scheduler_state"] is not None:
                physics_scheduler.load_state_dict(resume_ckpt["scheduler_state"])
            print(f"  Resuming physics at epoch {physics_start_epoch}")

        for epoch in range(physics_start_epoch, args.physics_epochs):
            l2_avg, cpx_avg = physics_epoch(
                student, tiler, sim, train_loader_full,
                physics_optimizer, device,
            )
            print(f"[Physics Epoch {epoch}] L2={l2_avg:.6f} cpx={cpx_avg:.6f}")

            # Weights-only best checkpoint (for deployment)
            save_path = os.path.join(args.output, "mini_unet_best.pth")
            torch.save(student.state_dict(), save_path)
            print(f"  Saved to {save_path}")

            # Per-epoch checkpoint + rolling latest
            ckpt_path = os.path.join(args.output, f"ckpt_physics_ep{epoch}.pth")
            save_checkpoint(ckpt_path, epoch, "physics",
                            student, physics_optimizer, best_val, physics_scheduler)
            save_checkpoint(os.path.join(args.output, "checkpoint_latest.pth"),
                            epoch, "physics",
                            student, physics_optimizer, best_val, physics_scheduler)
            _cleanup_old_checkpoints(args.output, "ckpt_physics", keep=3)

            if epoch == args.physics_epochs // 2:
                physics_scheduler.step()

    # Save final weights-only checkpoint
    final_path = os.path.join(args.output, "mini_unet_final.pth")
    torch.save(student.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    print(f"Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
