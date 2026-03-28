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
import subprocess
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
    import torch

    model.eval()
    model.cpu()

    # 1. Use 16-bit as the base for the entire model
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=INPUT_SHAPE, 
        granularity="name",
        backend="Vitis",
        default_precision="ap_fixed<16,6>", # Standard for HLS DL
        channels_last_conversion='full', 
        transpose_outputs=True,
        default_reuse_factor=128
    )

    # 2. Global settings
    config["Model"]["IOType"] = "io_stream" 
    config['Model']['Strategy'] = 'Resource' # Necessary for large U-Nets

    # 3. Patch the Layer Config
    if "LayerName" not in config:
        config["LayerName"] = {}

    for layer_name in list(config.get("LayerName", {}).keys()):
        if 'conv' in layer_name or 'up' in layer_name:
                # If the layer is deep, make it share even more
                config['LayerName'][layer_name]['ReuseFactor'] = 256 
                # Use "Resource" strategy specifically for these
                config['LayerName'][layer_name]['Strategy'] = 'Resource'
        
        # Sigmoid: Keep the "Hardened" LUT to prevent C-Sim crashes/NaNs
        if 'sigmoid' in layer_name.lower():
            config['LayerName'][layer_name]['Precision'] = 'ap_fixed<16,6>'
            config['LayerName'][layer_name]['table_size'] = 2048
            # Wider table_t allows the LUT to handle larger activation swings
            config['LayerName'][layer_name]['table_t'] = 'ap_fixed<18,8>'

    # 4. Critical: Downsize the LayerType defaults
    # This ensures skip-connections and resizes don't waste 32-bit registers
    config['LayerType'] = {
        'Conv2D': {'Precision': 'ap_fixed<16,6>'},
        'Activation': {'Precision': 'ap_fixed<16,6>'},
        'Concatenate': {'Precision': 'ap_fixed<16,6>'},
        'Resize': {'Precision': 'ap_fixed<16,6>'},
        'Merge': {'Precision': 'ap_fixed<16,6>'}
    }

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=INPUT_SHAPE,
        hls_config=config,
        output_dir=output_dir,
        backend="Vitis",
        part=FPGA_PART,
        io_type='io_stream',
    )

    return hls_model


def convert_onnx(model, output_dir):
    import hls4ml
    import onnx as _onnx
    import torch.onnx

    onnx_path = os.path.join(output_dir, "mini_unet.onnx")
    cl_onnx_path = os.path.join(output_dir, "mini_unet_cl.onnx")
    os.makedirs(output_dir, exist_ok=True)

    # 1. FORCE LEGACY EXPORT (Fixes the Opset 18 / kernel_shape error)
    dummy_input = torch.randn(1, 1, 64, 64) # Ensure NCHW for export
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=["input"], 
        output_names=["output"],
        opset_version=11, # Crucial for qonnx
        do_constant_folding=True,
        # Use the legacy exporter type to avoid "Dynamo" overhead
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )

    # 2. CHANNELS LAST CONVERSION
    # We must run this, or hls4ml will refuse the ONNX model
    try:
        subprocess.run(["qonnx-cleanup", onnx_path, "--out-file", onnx_path], check=True)
        subprocess.run(["qonnx-to-channels-last", "--make-input-channels-last", 
                        f"--out-file={cl_onnx_path}", onnx_path], check=True)
    except subprocess.CalledProcessError:
        print("CRITICAL: qonnx failed. Your model likely has incompatible reshapes/squeezes.")
        raise

    # 3. CONFIGURE HLS4ML
    model_proto = _onnx.load(cl_onnx_path)
    config = hls4ml.utils.config_from_onnx_model(
        model_proto,
        default_precision='ap_fixed<16,6>',
        backend="Vitis"
    )

    # BYPASS THE CRASHING OPTIMIZER
    config['SkipOptimizers'] = ['transpose_optimizer']
    config["Model"]["IOType"] = "io_stream"
    
    # Catch-all for precision errors
    if "LayerType" not in config: config["LayerType"] = {}
    config["LayerType"]["Precision"] = 'ap_fixed<16,6>'

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model_proto,
        hls_config=config,
        output_dir=output_dir,
        backend="Vitis",
        part=FPGA_PART,
    )
    return hls_model


def validate_csim(model, hls_model, num_samples=10):
    print(f"\nValidating C-simulation with {num_samples} random inputs...")
    max_diffs = []

    for i in range(num_samples):
        # 1. Create the input (Batch, Channel, Height, Width)
        test_input_pt = np.random.rand(1, 1, 64, 64).astype(np.float32)

        # 2. Prepare for HLS
        # Instead of np.squeeze(test_input_pt), remove ONLY the first dimension:
        test_input_hls = test_input_pt[0]  # Result: (1, 64, 64)

        # 3. Now transpose to (64, 64, 1)
        test_input_hls = np.transpose(test_input_hls, (1, 2, 0)) 

        # 4. Final safety check (optional but helpful for debugging)
        if test_input_hls.shape != (64, 64, 1):
            print(f"DEBUG: Shape is {test_input_hls.shape}, expected (64, 64, 1)")

        test_input_hls = np.ascontiguousarray(test_input_hls)
        
        # Ensure it is contiguous in memory (C++ hates non-contiguous arrays)
        test_input_hls = np.ascontiguousarray(test_input_hls)

        # 4. HLS C-simulation
        try:
            hls_out = hls_model.predict(test_input_hls)
        except Exception as e:
            print(f"HLS Predict failed: {e}")
            continue

        # 5. Post-process HLS output to match PyTorch (if hls_out is H,W,C)
        # hls4ml output is often flattened or (H, W, C)
        hls_out_reshaped = hls_out.reshape(pt_out.shape)

        diff = np.max(np.abs(pt_out - hls_out_reshaped))
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
