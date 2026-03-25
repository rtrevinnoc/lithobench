"""Convert trained MiniUNet to HLS firmware via hls4ml.

Supports two conversion paths:
  1. PyTorch frontend (default) — uses torch.fx tracing
  2. ONNX fallback — exports to ONNX first, then converts via hls4ml ONNX frontend

Usage:
    python3 fpga/convert_hls.py --weights saved/fpga/mini_unet_best.pth
    python3 fpga/convert_hls.py --weights saved/fpga/mini_unet_best.pth --backend onnx
    python3 fpga/convert_hls.py --weights saved/fpga/mini_unet_best.pth --synth
"""

import os
import sys
sys.path.append(".")
import argparse

import numpy as np
import torch

from fpga.mini_unet import MiniUNet, fuse_batchnorm


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FPGA_PART = "xcvu47p-fsvh2892-2L-e"  # AWS F2 VU47P
CLOCK_PERIOD = 4  # ns → 250 MHz
DEFAULT_PRECISION = "ap_fixed<8,4>"
DEFAULT_REUSE_FACTOR = 16
BOTTLENECK_REUSE_FACTOR = 32
IO_TYPE = "io_stream"
STRATEGY = "Resource"
INPUT_SHAPE = (1, 1, 64, 64)  # (batch, channels, height, width)


def load_and_fuse(weights_path):
    """Load MiniUNet weights and fuse BatchNorm into Conv2d."""
    model = MiniUNet()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    fused = fuse_batchnorm(model)
    fused.eval()
    return fused


def convert_pytorch(model, output_dir):
    """Convert MiniUNet via hls4ml's PyTorch frontend."""
    import hls4ml

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=INPUT_SHAPE,
        granularity="name",
        backend="Vitis",
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=DEFAULT_REUSE_FACTOR,
    )

    # Global settings
    config["Model"]["IOType"] = IO_TYPE
    config["Model"]["Strategy"] = STRATEGY

    # Increase reuse factor for bottleneck layers to fit DSP budget
    for layer_name in config.get("LayerName", {}):
        if "conv4" in layer_name:
            config["LayerName"][layer_name]["ReuseFactor"] = BOTTLENECK_REUSE_FACTOR

    # Sigmoid lookup table configuration
    for layer_name in config.get("LayerName", {}):
        if "sigmoid" in layer_name.lower():
            config["LayerName"][layer_name]["table_size"] = 512
            config["LayerName"][layer_name]["table_t"] = "ap_fixed<10,6>"

    print("hls4ml configuration:")
    print(hls4ml.utils.config.print_config(config))

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=INPUT_SHAPE,
        hls_config=config,
        output_dir=output_dir,
        backend="Vitis",
        part=FPGA_PART,
        clock_period=CLOCK_PERIOD,
    )

    return hls_model


def convert_onnx(model, output_dir):
    """Fallback: export to ONNX, then convert via hls4ml ONNX frontend."""
    import hls4ml

    onnx_path = os.path.join(output_dir, "mini_unet.onnx")
    os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    print(f"ONNX model exported to {onnx_path}")

    config = {
        "Model": {
            "Precision": DEFAULT_PRECISION,
            "ReuseFactor": DEFAULT_REUSE_FACTOR,
            "IOType": IO_TYPE,
            "Strategy": STRATEGY,
        }
    }

    # hls4ml's ONNX parser requires channels-last layout
    cl_onnx_path = onnx_path.replace(".onnx", "_cl.onnx")
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.to_channels_last import to_channels_last
    cl_model = to_channels_last(ModelWrapper(onnx_path))
    cl_model.save(cl_onnx_path)
    print(f"Channels-last ONNX written to {cl_onnx_path}")

    hls_model = hls4ml.converters.convert_from_onnx_model(
        cl_onnx_path,
        hls_config=config,
        output_dir=output_dir,
        backend="Vitis",
        part=FPGA_PART,
        clock_period=CLOCK_PERIOD,
    )

    return hls_model


def validate_csim(model, hls_model, num_samples=10):
    """Compare PyTorch and HLS C-simulation outputs."""
    print(f"\nValidating C-simulation with {num_samples} random inputs...")
    max_diffs = []

    for i in range(num_samples):
        test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        # Clip to valid mask range
        test_input = np.clip(test_input, 0.0, 1.0)

        # PyTorch reference
        with torch.no_grad():
            pt_out = model(torch.tensor(test_input)).numpy()

        # HLS C-simulation
        hls_out = hls_model.predict(test_input)

        diff = np.max(np.abs(pt_out - hls_out))
        max_diffs.append(diff)
        print(f"  Sample {i + 1}: max|diff| = {diff:.6f}")

    mean_diff = np.mean(max_diffs)
    max_diff = np.max(max_diffs)
    print(f"\nC-sim validation: mean max|diff| = {mean_diff:.6f}, worst = {max_diff:.6f}")

    if max_diff > 0.1:
        print("WARNING: Large difference between PyTorch and HLS outputs.")
        print("  Consider adjusting quantization precision.")
    else:
        print("C-simulation validation PASSED.")

    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Convert MiniUNet to HLS via hls4ml")
    parser.add_argument("--weights", "-w", required=True, type=str,
                        help="Path to trained MiniUNet weights (.pth)")
    parser.add_argument("--output_dir", "-o", default="hls4ml_mini_unet", type=str,
                        help="Output directory for generated HLS project")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"],
                        help="Conversion backend: pytorch (default) or onnx (fallback)")
    parser.add_argument("--synth", action="store_true",
                        help="Run Vitis HLS synthesis after conversion")
    parser.add_argument("--cosim", action="store_true",
                        help="Run C/RTL co-simulation")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip C-simulation validation")
    args = parser.parse_args()

    print(f"Loading and fusing model from {args.weights}")
    model = load_and_fuse(args.weights)

    print(f"\nConverting via {args.backend} backend...")
    try:
        if args.backend == "pytorch":
            hls_model = convert_pytorch(model, args.output_dir)
        else:
            hls_model = convert_onnx(model, args.output_dir)
    except Exception as e:
        if args.backend == "pytorch":
            print(f"\nPyTorch frontend failed: {e}")
            print("Falling back to ONNX frontend...")
            hls_model = convert_onnx(model, args.output_dir)
        else:
            raise

    print("\nCompiling HLS model...")
    hls_model.compile()

    if not args.no_validate:
        validate_csim(model, hls_model)

    if args.synth:
        print("\nRunning Vitis HLS synthesis (this may take a while)...")
        report = hls_model.build(csim=False, synth=True, cosim=args.cosim)
        print("\nSynthesis complete. Reading report...")
        try:
            import hls4ml
            hls4ml.report.read_vivado_report(args.output_dir)
        except Exception as e:
            print(f"Could not read report: {e}")
            print(f"Check {args.output_dir} for synthesis results.")

    print(f"\nHLS project generated at: {args.output_dir}")
    print("Next steps:")
    print(f"  1. Review the generated code in {args.output_dir}/")
    print(f"  2. Run synthesis: python3 fpga/convert_hls.py --weights {args.weights} --synth")
    print(f"  3. Integrate IP into F2 shell: bash fpga/f2_deploy.sh {args.output_dir}")


if __name__ == "__main__":
    main()
