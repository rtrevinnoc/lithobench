#!/usr/bin/env bash
# ============================================================================
# F2 FPGA Deployment Script for MiniUNet HLS IP
#
# Integrates the hls4ml-generated Vitis HLS IP into the AWS F2 HDK shell
# and builds a Design Checkpoint (DCP) for AFI creation.
#
# Prerequisites:
#   - AWS F2 FPGA Developer AMI (with Vivado 2025.x)
#   - aws-fpga repo cloned and HDK set up:
#       git clone https://github.com/aws/aws-fpga.git -b f2
#       source aws-fpga/hdk_setup.sh
#   - hls4ml project already synthesized:
#       python3 fpga/convert_hls.py --weights saved/fpga/mini_unet_best.pth --synth
#
# Usage:
#   bash fpga/f2_deploy.sh <hls4ml_output_dir>
#
# Example:
#   bash fpga/f2_deploy.sh hls4ml_mini_unet
# ============================================================================

set -euo pipefail

HLS_DIR="${1:?Usage: $0 <hls4ml_output_dir>}"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------

if [ -z "${HDK_DIR:-}" ]; then
    echo "ERROR: HDK_DIR not set. Run 'source aws-fpga/hdk_setup.sh' first."
    exit 1
fi

if [ -z "${CL_DIR:-}" ]; then
    # Default CL project location
    export CL_DIR="${HDK_DIR}/cl/developer_designs/cl_mini_unet"
fi

echo "=== F2 MiniUNet Deployment ==="
echo "HLS source:  ${HLS_DIR}"
echo "CL project:  ${CL_DIR}"
echo "HDK:         ${HDK_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Find the synthesized HLS IP
# ---------------------------------------------------------------------------

# hls4ml generates a Vitis HLS project; the RTL export is under:
#   <output_dir>/myproject_prj/solution1/syn/verilog/
# or for the Vitis backend:
#   <output_dir>/myproject_prj/solution1/impl/export.zip

HLS_RTL_DIR="${HLS_DIR}/myproject_prj/solution1/syn/verilog"
HLS_EXPORT="${HLS_DIR}/myproject_prj/solution1/impl/export.zip"

if [ ! -d "${HLS_RTL_DIR}" ] && [ ! -f "${HLS_EXPORT}" ]; then
    echo "ERROR: Cannot find synthesized RTL at:"
    echo "  ${HLS_RTL_DIR}"
    echo "  ${HLS_EXPORT}"
    echo ""
    echo "Run synthesis first:"
    echo "  python3 fpga/convert_hls.py --weights saved/fpga/mini_unet_best.pth --synth"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Create CL project from template
# ---------------------------------------------------------------------------

TEMPLATE_DIR="${HDK_DIR}/cl/examples/CL_TEMPLATE"

if [ ! -d "${TEMPLATE_DIR}" ]; then
    echo "ERROR: CL_TEMPLATE not found at ${TEMPLATE_DIR}"
    echo "Ensure aws-fpga f2 branch is checked out and HDK is set up."
    exit 1
fi

if [ -d "${CL_DIR}" ]; then
    echo "CL project already exists at ${CL_DIR}. Skipping template copy."
else
    echo "Creating CL project from template..."
    cp -r "${TEMPLATE_DIR}" "${CL_DIR}"
    echo "  Created ${CL_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 3: Copy HLS RTL into CL project
# ---------------------------------------------------------------------------

IP_DIR="${CL_DIR}/design/ip/mini_unet_hls"
mkdir -p "${IP_DIR}"

if [ -d "${HLS_RTL_DIR}" ]; then
    echo "Copying HLS RTL to CL project..."
    cp -r "${HLS_RTL_DIR}"/* "${IP_DIR}/"
elif [ -f "${HLS_EXPORT}" ]; then
    echo "Extracting HLS export to CL project..."
    unzip -o "${HLS_EXPORT}" -d "${IP_DIR}/"
fi

echo "  HLS IP installed at ${IP_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Generate wrapper (instructions)
# ---------------------------------------------------------------------------

cat <<'WRAPPER_INFO'

=== Next Steps (Manual Integration) ===

The HLS IP has been copied into the CL project. You now need to:

1. Edit the CL top-level to instantiate the HLS IP:
   File: ${CL_DIR}/design/cl_top.sv

   Connect the HLS module's AXI interfaces:
   - AXI-Lite slave (ap_ctrl) -> OCL BAR0 (register control)
   - AXI-Stream / AXI-MM input -> PCIS (tile data from host)
   - AXI-Stream / AXI-MM output -> PCIS (tile results to host)

2. Add the HLS IP files to the build file list:
   File: ${CL_DIR}/build/scripts/synth_cl.tcl

   Add:
     read_verilog [glob ${CL_DIR}/design/ip/mini_unet_hls/*.v]
     read_verilog [glob ${CL_DIR}/design/ip/mini_unet_hls/*.sv]

3. Implement the DMA tile transfer logic:
   - Host writes 64x64x1 tile (4096 bytes at ap_fixed<8,4>) to FPGA via PCIS
   - FPGA processes tile through HLS IP
   - Host reads 64x64x1 result from FPGA via PCIS

4. Build the DCP:
   cd ${CL_DIR}/build/scripts
   python3 aws_build_dcp_from_cl.py

5. Create AFI from DCP:
   aws ec2 create-fpga-image \
     --name "mini-unet-ilt" \
     --description "MiniUNet ILT mask optimizer" \
     --input-storage-location Bucket=<your-bucket>,Key=<dcp-tar-path>

6. Load AFI on F2 instance:
   sudo fpga-load-local-image -S 0 -I <agfi-id>
   sudo fpga-describe-local-image -S 0 -H

WRAPPER_INFO

echo ""
echo "=== CL project ready at ${CL_DIR} ==="
echo "See the instructions above for manual integration steps."
echo "For Vivado IPI (GUI) flow, see:"
echo "  ${HDK_DIR}/hdk/docs/IPI-GUI-Vivado-Setup.md"
