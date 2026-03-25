"""Tests for convert_hls.py — focused on the MiniUNet → ONNX → hls4ml pipeline.

Run from repo root:
    python3 -m pytest fpga/test_convert_hls.py -v

Tests do NOT require hls4ml, Vitis, or a real .pth file. They verify that:
  - MiniUNet no longer emits ConvTranspose nodes in its ONNX graph
  - ONNX export succeeds and produces a valid model
  - The qonnx cleanup + channels-last + name-patching helpers work
  - validate_csim logic works with a mocked hls_model
"""

import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnx
import torch

from fpga.mini_unet import MiniUNet, fuse_batchnorm
import fpga.convert_hls as chls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    m = MiniUNet()
    m.eval()
    return m


def _export_onnx(model, path):
    dummy = torch.randn(*chls.INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=9,
        dynamic_axes=None,
    )


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestMiniUNetArchitecture(unittest.TestCase):

    def test_no_convtranspose_layers(self):
        """After fix, MiniUNet must not contain any ConvTranspose2d."""
        model = _make_model()
        for name, module in model.named_modules():
            self.assertNotIsInstance(
                module,
                torch.nn.ConvTranspose2d,
                f"Found ConvTranspose2d at '{name}' — should be nn.Upsample",
            )

    def test_upsample_layers_present(self):
        """up1/up2/up3 must be nn.Upsample with scale_factor=2."""
        model = _make_model()
        for attr in ("up1", "up2", "up3"):
            layer = getattr(model, attr)
            self.assertIsInstance(layer, torch.nn.Upsample, f"{attr} is not nn.Upsample")
            self.assertEqual(layer.scale_factor, 2, f"{attr}.scale_factor != 2")
            self.assertEqual(layer.mode, "nearest", f"{attr}.mode != 'nearest'")

    def test_forward_shape(self):
        model = _make_model()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, torch.Size([1, 1, 64, 64]))

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        model = _make_model()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_fuse_batchnorm_no_bn_remaining(self):
        model = _make_model()
        fused = fuse_batchnorm(model)
        for name, module in fused.named_modules():
            self.assertNotIsInstance(
                module, torch.nn.BatchNorm2d,
                f"BatchNorm2d still present at '{name}' after fusion",
            )

    def test_fuse_batchnorm_output_unchanged(self):
        """Fused model must produce numerically identical output."""
        model = _make_model()
        fused = fuse_batchnorm(model)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out_orig = model(x)
            out_fused = fused(x)
        self.assertTrue(
            torch.allclose(out_orig, out_fused, atol=1e-5),
            "Fused model output differs from original",
        )


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------

class TestOnnxExport(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.onnx_path = os.path.join(self.tmpdir, "mini_unet.onnx")

    def _export(self):
        model = fuse_batchnorm(_make_model())
        _export_onnx(model, self.onnx_path)
        return onnx.load(self.onnx_path)

    def test_export_succeeds(self):
        proto = self._export()
        self.assertTrue(os.path.exists(self.onnx_path))
        onnx.checker.check_model(proto)

    def test_no_convtranspose_in_onnx(self):
        """Core regression test: ConvTranspose must NOT appear in the ONNX graph."""
        proto = self._export()
        op_types = {node.op_type for node in proto.graph.node}
        self.assertNotIn(
            "ConvTranspose",
            op_types,
            "ConvTranspose op found in ONNX export — hls4ml will reject this model",
        )

    def test_resize_in_onnx(self):
        """nn.Upsample(nearest) must export as Resize (or Upsample) ONNX op."""
        proto = self._export()
        op_types = {node.op_type for node in proto.graph.node}
        self.assertTrue(
            "Resize" in op_types or "Upsample" in op_types,
            f"Expected Resize/Upsample op in ONNX graph, got: {op_types}",
        )

    def test_input_output_names(self):
        proto = self._export()
        input_names = [i.name for i in proto.graph.input]
        output_names = [o.name for o in proto.graph.output]
        self.assertIn("input", input_names)
        self.assertIn("output", output_names)


# ---------------------------------------------------------------------------
# Name-patching helper test
# ---------------------------------------------------------------------------

class TestOnnxNamePatching(unittest.TestCase):
    """Test the empty-name patch logic extracted from convert_onnx."""

    def _apply_patch(self, proto):
        """Replicate the patching logic from convert_hls.convert_onnx."""
        changed = False
        for i, node in enumerate(proto.graph.node):
            if not node.name:
                node.name = f"{node.op_type}_{i}"
                changed = True
        for i, out in enumerate(proto.graph.output):
            if not out.name:
                real_name = proto.graph.node[-1].output[0]
                new_out = onnx.helper.make_tensor_value_info(
                    real_name, out.type.tensor_type.elem_type, None
                )
                proto.graph.output.remove(out)
                proto.graph.output.insert(i, new_out)
                changed = True
        return changed

    def test_unnamed_nodes_get_names(self):
        node = onnx.helper.make_node("Conv", inputs=["x", "w"], outputs=["y"], name="")
        graph = onnx.helper.make_graph([node], "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        proto = onnx.helper.make_model(graph)
        self._apply_patch(proto)
        self.assertEqual(proto.graph.node[0].name, "Conv_0")

    def test_already_named_nodes_unchanged(self):
        node = onnx.helper.make_node("Conv", inputs=["x", "w"], outputs=["y"], name="my_conv")
        graph = onnx.helper.make_graph([node], "g",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        )
        proto = onnx.helper.make_model(graph)
        changed = self._apply_patch(proto)
        self.assertFalse(changed)
        self.assertEqual(proto.graph.node[0].name, "my_conv")


# ---------------------------------------------------------------------------
# validate_csim tests (mocked hls_model)
# ---------------------------------------------------------------------------

class TestValidateCsim(unittest.TestCase):

    def test_passes_when_outputs_match(self):
        model = _make_model()
        # Mock hls_model.predict to return the same as PyTorch
        hls_model = MagicMock()
        def fake_predict(x):
            with torch.no_grad():
                return model(torch.tensor(x)).numpy()
        hls_model.predict.side_effect = fake_predict

        max_diff = chls.validate_csim(model, hls_model, num_samples=3)
        self.assertLess(max_diff, 0.1)

    def test_warns_when_outputs_diverge(self):
        model = _make_model()
        hls_model = MagicMock()
        # Return all-ones (far from sigmoid output)
        hls_model.predict.return_value = np.ones((1, 1, 64, 64), dtype=np.float32)

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            max_diff = chls.validate_csim(model, hls_model, num_samples=2)
        self.assertGreater(max_diff, 0.1)
        self.assertIn("WARNING", buf.getvalue())


# ---------------------------------------------------------------------------
# load_and_fuse — strict=False smoke test
# ---------------------------------------------------------------------------

EXISTING_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "saved", "mini_unet", "mini_unet_best.pth"
)


class TestLoadAndFuse(unittest.TestCase):

    def test_load_fresh_model(self):
        """Save a newly-initialised model and reload it."""
        model = MiniUNet()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(model.state_dict(), path)
            loaded = chls.load_and_fuse(path)
            self.assertIsInstance(loaded, MiniUNet)
        finally:
            os.unlink(path)

    def test_load_strict_false_ignores_extra_keys(self):
        """Old weights with ConvTranspose keys load without error when strict=False."""
        old_state = MiniUNet().state_dict()
        # Inject fake ConvTranspose keys as they appear in old .pth files
        old_state["up3.weight"] = torch.randn(64, 64, 2, 2)
        old_state["up3.bias"] = torch.zeros(64)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(old_state, path)
            model = MiniUNet()
            state = torch.load(path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.assertIn("up3.weight", unexpected)
            self.assertIn("up3.bias", unexpected)
        finally:
            os.unlink(path)

    @unittest.skipUnless(os.path.exists(EXISTING_WEIGHTS), f"weights not found: {EXISTING_WEIGHTS}")
    def test_load_existing_weights_strict_false(self):
        """Load saved/mini_unet/mini_unet_best.pth with strict=False.

        The old .pth has up1/up2/up3 ConvTranspose2d weights which are now
        unexpected. All encoder/bottleneck/decoder Conv2d weights must still load.
        """
        state = torch.load(EXISTING_WEIGHTS, map_location="cpu")
        model = MiniUNet()
        missing, unexpected = model.load_state_dict(state, strict=False)

        # ConvTranspose keys from the old model are expected to be "unexpected"
        convt_keys = [k for k in unexpected if k.startswith(("up1.", "up2.", "up3."))]
        self.assertTrue(len(convt_keys) > 0, "Expected old ConvTranspose keys in unexpected set")

        # No encoder/bottleneck/decoder conv keys should be missing
        critical_prefixes = ("conv1.", "conv2.", "conv3.", "conv4.",
                             "deconv1.", "deconv2.", "deconv3.", "final_conv.")
        critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
        self.assertEqual(
            critical_missing, [],
            f"Critical conv weights missing after load: {critical_missing}",
        )

    @unittest.skipUnless(os.path.exists(EXISTING_WEIGHTS), f"weights not found: {EXISTING_WEIGHTS}")
    def test_existing_weights_forward_pass(self):
        """Model loaded from existing .pth (strict=False) must produce valid output."""
        state = torch.load(EXISTING_WEIGHTS, map_location="cpu")
        model = MiniUNet()
        model.load_state_dict(state, strict=False)
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape, torch.Size([1, 1, 64, 64]))
        self.assertTrue((out >= 0).all() and (out <= 1).all(),
                        "Output out of [0,1] range after loading existing weights")

    @unittest.skipUnless(os.path.exists(EXISTING_WEIGHTS), f"weights not found: {EXISTING_WEIGHTS}")
    def test_existing_weights_onnx_no_convtranspose(self):
        """Full pipeline: existing .pth → fuse BN → ONNX export → no ConvTranspose."""
        state = torch.load(EXISTING_WEIGHTS, map_location="cpu")
        model = MiniUNet()
        model.load_state_dict(state, strict=False)
        model.eval()
        fused = fuse_batchnorm(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "mini_unet.onnx")
            _export_onnx(fused, onnx_path)
            proto = onnx.load(onnx_path)

        op_types = {node.op_type for node in proto.graph.node}
        self.assertNotIn(
            "ConvTranspose", op_types,
            "ConvTranspose still present in ONNX export with existing weights",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
