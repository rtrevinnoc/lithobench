import torch
import torch.nn.functional as F


class TiledInference:
    """Manages tiled inference of MiniUNet over a full-size image.

    The 512x512 input is reflection-padded to 544x544, split into overlapping
    64x64 tiles (stride 32), and reassembled after inference by keeping only
    the central 32x32 crop of each tile output.

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

    @property
    def total_tiles(self):
        return self.num_tiles_per_side ** 2

    def extract_tiles(self, image):
        """Extract overlapping tiles from the input image.

        Args:
            image: Tensor of shape (1, 1, H, W) where H = W = image_size.

        Returns:
            tiles: Tensor of shape (N, 1, tile_size, tile_size).
            positions: List of (row_idx, col_idx) grid positions.
        """
        padded = F.pad(
            image,
            [self.overlap, self.overlap, self.overlap, self.overlap],
            mode="reflect",
        )

        tiles = []
        positions = []
        n = self.num_tiles_per_side
        for i in range(n):
            for j in range(n):
                r = i * self.stride
                c = j * self.stride
                tile = padded[:, :, r : r + self.tile_size, c : c + self.tile_size]
                tiles.append(tile)
                positions.append((i, j))

        return torch.cat(tiles, dim=0), positions

    def reassemble(self, tile_outputs, positions):
        """Reassemble processed tiles into a full-size output image.

        Only the central (stride x stride) region of each tile is kept,
        discarding the overlap border.  This implementation uses only
        slicing, permute and reshape, so it is fully differentiable and
        supports autograd back-propagation through the reassembly.

        Args:
            tile_outputs: Tensor of shape (N, 1, tile_size, tile_size).
            positions: List of (row_idx, col_idx) grid positions (unused
                       when tiles are in row-major order, kept for API
                       compatibility).

        Returns:
            output: Tensor of shape (1, 1, image_size, image_size).
        """
        n = self.num_tiles_per_side  # 16

        # Centre-crop each tile: (N, 1, stride, stride)
        crops = tile_outputs[
            :, :,
            self.overlap : self.tile_size - self.overlap,
            self.overlap : self.tile_size - self.overlap,
        ]

        # Reshape to grid layout: (n_row, n_col, 1, stride, stride)
        crops = crops.reshape(n, n, 1, self.stride, self.stride)

        # Interleave rows and columns:
        #   (n_row, n_col, C, sh, sw) -> (C, n_row, sh, n_col, sw)
        crops = crops.permute(2, 0, 3, 1, 4).contiguous()

        # Merge spatial dims: (1, 1, image_size, image_size)
        output = crops.reshape(1, 1, n * self.stride, n * self.stride)

        return output

    def forward(self, image, model, batch_size=64):
        """Differentiable tiled forward pass — keeps gradients for training.

        Tiles are extracted, processed through the model, and reassembled.
        All operations support autograd so gradients flow from the
        reassembled output back through every tile and the model weights.

        Args:
            image: Tensor of shape (1, 1, H, W) on the model's device.
            model: nn.Module mapping (B, 1, tile, tile) -> (B, 1, tile, tile).
            batch_size: Tiles per forward pass (controls peak memory).

        Returns:
            output: Tensor of shape (1, 1, image_size, image_size), with grad.
        """
        tiles, positions = self.extract_tiles(image)
        all_outputs = []

        for start in range(0, tiles.shape[0], batch_size):
            batch = tiles[start : start + batch_size]
            out = model(batch)
            all_outputs.append(out)

        tile_outputs = torch.cat(all_outputs, dim=0)
        return self.reassemble(tile_outputs, positions)

    def run(self, image, model, batch_size=64, device=None):
        """Non-differentiable tiled inference (no gradients, for deployment).

        Args:
            image: Tensor of shape (1, 1, H, W).
            model: A callable mapping (B, 1, tile, tile) -> (B, 1, tile, tile).
            batch_size: Number of tiles to process per forward pass.
            device: Device to run inference on. If None, uses image's device.

        Returns:
            output: Tensor of shape (1, 1, image_size, image_size).
        """
        if device is None:
            device = image.device

        tiles, positions = self.extract_tiles(image)
        all_outputs = []

        for start in range(0, tiles.shape[0], batch_size):
            batch = tiles[start : start + batch_size].to(device)
            with torch.no_grad():
                out = model(batch)
            all_outputs.append(out.cpu())

        tile_outputs = torch.cat(all_outputs, dim=0)
        return self.reassemble(tile_outputs, positions)
