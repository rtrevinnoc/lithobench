"""FPGA driver for MiniUNet HLS IP on AWS F2 (VU47P).

Communicates with the CL via:
  - /dev/xdma0_h2c_0  : host-to-card DMA (writing input tiles)
  - /dev/xdma0_c2h_0  : card-to-host DMA (reading output tiles)
  - /dev/xdma0_user   : mmapped OCL BAR0 for register access

Register map (BAR0 offsets, 32-bit registers):
  0x00  CTRL        W   bit[0]=ap_start (pulse), bit[1]=soft_reset
  0x04  STATUS      R   bit[0]=ap_done (sticky), bit[1]=ap_idle
  0x08  TILE_COUNT  R   tiles processed since reset (debug)
  0x0C  ERROR_FLAGS R   sticky error bits

DMA address layout (PCIS BAR4, CL-internal):
  0x0000_0000  Input BRAM  (8 KB = 4096 x 16-bit samples)
  0x0000_2000  Output BRAM (8 KB)
"""

import mmap
import os
import struct
import time

import numpy as np

# ---- Tile geometry ----
TILE_H = 64
TILE_W = 64
TILE_SAMPLES = TILE_H * TILE_W           # 4096
BYTES_PER_SAMPLE = 2                      # ap_fixed<16,6> = 16 bits
TILE_BYTES = TILE_SAMPLES * BYTES_PER_SAMPLE  # 8192

# ap_fixed<16,6>: 6 integer bits, 10 fractional bits → scale = 2^10
FIXED_SCALE = 1024.0

# ---- XDMA device paths (AWS F2 defaults) ----
H2C_DEV = "/dev/xdma0_h2c_0"
C2H_DEV = "/dev/xdma0_c2h_0"
USER_DEV = "/dev/xdma0_user"

# ---- DMA addresses within CL address space ----
INPUT_BRAM_OFFSET = 0x00000000
OUTPUT_BRAM_OFFSET = 0x00002000

# ---- OCL BAR0 register offsets ----
REG_CTRL = 0x00
REG_STATUS = 0x04
REG_TILE_COUNT = 0x08
REG_ERROR_FLAGS = 0x0C

# ---- Register bit masks ----
CTRL_AP_START = 0x00000001
CTRL_SOFT_RST = 0x00000002
STATUS_AP_DONE = 0x00000001
STATUS_AP_IDLE = 0x00000002

# ---- Polling ----
POLL_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.0001  # 100 µs

BAR0_MMAP_SIZE = 4096  # one page


class FPGADriver:
    """Low-level FPGA driver for MiniUNet tile inference on AWS F2.

    Usage::

        driver = FPGADriver()
        driver.open()
        result = driver.process_tile(input_np_float32)
        driver.close()

        # Or as a context manager:
        with FPGADriver() as drv:
            result = drv.process_tile(tile)
    """

    def __init__(self, h2c_dev=H2C_DEV, c2h_dev=C2H_DEV, user_dev=USER_DEV):
        self.h2c_dev = h2c_dev
        self.c2h_dev = c2h_dev
        self.user_dev = user_dev
        self._h2c_fd = None
        self._c2h_fd = None
        self._user_fd = None
        self._bar0 = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Open XDMA device files and mmap the OCL BAR."""
        self._h2c_fd = open(self.h2c_dev, "wb", buffering=0)
        self._c2h_fd = open(self.c2h_dev, "rb", buffering=0)
        self._user_fd = open(self.user_dev, "r+b", buffering=0)
        self._bar0 = mmap.mmap(
            self._user_fd.fileno(),
            BAR0_MMAP_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )

    def close(self):
        """Close all device handles."""
        if self._bar0 is not None:
            self._bar0.close()
            self._bar0 = None
        for attr in ("_h2c_fd", "_c2h_fd", "_user_fd"):
            fd = getattr(self, attr)
            if fd is not None:
                fd.close()
                setattr(self, attr, None)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Register access
    # ------------------------------------------------------------------

    def _reg_read(self, offset):
        """Read a 32-bit little-endian register from OCL BAR0."""
        self._bar0.seek(offset)
        return struct.unpack("<I", self._bar0.read(4))[0]

    def _reg_write(self, offset, value):
        """Write a 32-bit little-endian value to OCL BAR0."""
        self._bar0.seek(offset)
        self._bar0.write(struct.pack("<I", value))
        self._bar0.flush()

    # ------------------------------------------------------------------
    # DMA helpers
    # ------------------------------------------------------------------

    def _dma_write(self, offset, data_bytes):
        """Write bytes to FPGA PCIS address space via H2C channel."""
        os.pwrite(self._h2c_fd.fileno(), data_bytes, offset)

    def _dma_read(self, offset, length):
        """Read bytes from FPGA PCIS address space via C2H channel."""
        return os.pread(self._c2h_fd.fileno(), length, offset)

    # ------------------------------------------------------------------
    # Fixed-point conversion
    # ------------------------------------------------------------------

    @staticmethod
    def float_to_fixed(arr_f32):
        """Convert a float32 array to ap_fixed<16,6> int16 bytes.

        Args:
            arr_f32: numpy float32 array, any shape.

        Returns:
            bytes, little-endian int16-packed, length = arr_f32.size * 2.
        """
        flat = arr_f32.flatten().astype(np.float32)
        scaled = np.round(flat * FIXED_SCALE)
        scaled = np.clip(scaled, -32768, 32767)
        return scaled.astype(np.int16).astype("<i2").tobytes()

    @staticmethod
    def fixed_to_float(raw_bytes, shape=(TILE_H, TILE_W)):
        """Convert ap_fixed<16,6> int16 bytes back to float32.

        Args:
            raw_bytes: bytes, little-endian int16-packed.
            shape: output numpy array shape.

        Returns:
            numpy float32 array reshaped to `shape`.
        """
        int16 = np.frombuffer(raw_bytes, dtype="<i2")
        return int16.astype(np.float32).reshape(shape) / FIXED_SCALE

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self):
        """Assert then deassert soft reset on the CL."""
        self._reg_write(REG_CTRL, CTRL_SOFT_RST)
        time.sleep(0.001)
        self._reg_write(REG_CTRL, 0x00000000)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def process_tile(self, tile_f32):
        """Process one 64x64 tile through the FPGA MiniUNet.

        Args:
            tile_f32: numpy float32 array, shape broadcastable to (64, 64).

        Returns:
            numpy float32 array of shape (64, 64).

        Raises:
            RuntimeError: on FPGA timeout or error flags.
        """
        arr = np.asarray(tile_f32, dtype=np.float32).reshape(TILE_H, TILE_W)

        # 1. Write input tile to FPGA input BRAM
        self._dma_write(INPUT_BRAM_OFFSET, self.float_to_fixed(arr))

        # 2. Pulse ap_start (STATUS ap_done sticky bit cleared by hardware on start)
        self._reg_write(REG_CTRL, CTRL_AP_START)

        # 3. Poll STATUS until ap_done
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while True:
            status = self._reg_read(REG_STATUS)
            if status & STATUS_AP_DONE:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"FPGA tile inference timed out after {POLL_TIMEOUT_S}s. "
                    f"STATUS=0x{status:08x}"
                )
            time.sleep(POLL_INTERVAL_S)

        # 4. Check error flags
        errors = self._reg_read(REG_ERROR_FLAGS)
        if errors:
            raise RuntimeError(f"FPGA error flags: 0x{errors:08x}")

        # 5. Read result from output BRAM
        raw = self._dma_read(OUTPUT_BRAM_OFFSET, TILE_BYTES)
        if len(raw) != TILE_BYTES:
            raise RuntimeError(
                f"C2H DMA returned {len(raw)} bytes, expected {TILE_BYTES}"
            )

        return self.fixed_to_float(raw)

    def get_tile_count(self):
        """Return the cumulative tile count from the hardware debug register."""
        return self._reg_read(REG_TILE_COUNT)
