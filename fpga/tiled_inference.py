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
        discarding the overlap border.

        Args:
            tile_outputs: Tensor of shape (N, 1, tile_size, tile_size).
            positions: List of (row_idx, col_idx) grid positions.

        Returns:
            output: Tensor of shape (1, 1, image_size, image_size).
        """
        output = torch.zeros(
            1, 1, self.image_size, self.image_size,
            dtype=tile_outputs.dtype,
            device=tile_outputs.device,
        )

        for idx, (i, j) in enumerate(positions):
            crop = tile_outputs[
                idx : idx + 1,
                :,
                self.overlap : self.tile_size - self.overlap,
                self.overlap : self.tile_size - self.overlap,
            ]
            r = i * self.stride
            c = j * self.stride
            output[:, :, r : r + self.stride, c : c + self.stride] = crop

        return output

    def run(self, image, model, batch_size=64, device=None):
        """Run tiled inference end-to-end.

        Args:
            image: Tensor of shape (1, 1, H, W).
            model: A callable (nn.Module) that maps (B, 1, tile, tile) -> (B, 1, tile, tile).
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
