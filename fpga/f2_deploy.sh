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

IP_DIR="${CL_DIR}/design"

if [ -d "${HLS_RTL_DIR}" ]; then
    echo "Copying HLS RTL to CL design directory..."
    cp "${HLS_RTL_DIR}"/*.v  "${IP_DIR}/" 2>/dev/null || true
    cp "${HLS_RTL_DIR}"/*.sv "${IP_DIR}/" 2>/dev/null || true
    cp "${HLS_RTL_DIR}"/*.dat "${IP_DIR}/" 2>/dev/null || true
elif [ -f "${HLS_EXPORT}" ]; then
    echo "Extracting HLS export to CL design directory..."
    unzip -o "${HLS_EXPORT}" -d "${IP_DIR}/"
fi

echo "  HLS IP installed at ${IP_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Copy cl_top.sv (AXI bridge + BRAM + FSM) into CL design dir
# ---------------------------------------------------------------------------

# cl_top.sv lives alongside this script in fpga/host/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CL_TOP_SRC="${SCRIPT_DIR}/host/cl_top.sv"
CL_DESIGN_DIR="${CL_DIR}/design"

if [ -f "${CL_TOP_SRC}" ]; then
    echo "Copying cl_top.sv to CL design directory..."
    cp "${CL_TOP_SRC}" "${CL_DESIGN_DIR}/cl_top.sv"
    echo "  Installed ${CL_DESIGN_DIR}/cl_top.sv"
else
    echo "WARNING: ${CL_TOP_SRC} not found. cl_top.sv must be installed manually."
fi

# ---------------------------------------------------------------------------
# Step 5: Create synth_cl_mini_unet.tcl from template and add HLS IP files
#
# The AWS build system expects synth_${CL}.tcl (i.e. synth_cl_mini_unet.tcl).
# The CL_TEMPLATE provides synth_CL_TEMPLATE.tcl as the starting point.
# ---------------------------------------------------------------------------

SCRIPTS_DIR="${CL_DIR}/build/scripts"
SYNTH_TCL="${SCRIPTS_DIR}/synth_cl_mini_unet.tcl"
TEMPLATE_TCL="${SCRIPTS_DIR}/synth_CL_TEMPLATE.tcl"

if [ ! -f "${TEMPLATE_TCL}" ]; then
    echo "ERROR: Template TCL not found at ${TEMPLATE_TCL}"
    exit 1
fi

echo "Creating synth_cl_mini_unet.tcl from template..."
cp "${TEMPLATE_TCL}" "${SYNTH_TCL}"

# The HDK template reads .v/.sv from ${src_post_enc_dir} (the encryption
# staging dir), which only contains files listed in encrypt.tcl — NOT
# the 290 HLS Verilog files we placed in design/.  We must add
# read_verilog commands for those files BEFORE synth_design runs.
#
# Insert right before the first "#---- End of section replaced by User"
# marker (i.e. inside the user-editable read_verilog block).

awk '
/^#---- End of section replaced by User ----/ && !done {
    print ""
    print "# ---- hls4ml MiniUNet HLS IP (added by f2_deploy.sh) ----"
    print "set HLS_DESIGN_DIR [file join $env(CL_DIR) design]"
    print "foreach f [glob -nocomplain [file join $HLS_DESIGN_DIR *.v]] {"
    print "    read_verilog $f"
    print "}"
    print "set_property -name {xpm_libraries} -value {XPM_MEMORY XPM_CDC XPM_FIFO} \\"
    print "    -objects [current_project]"
    print "# ---- end hls4ml patch ----"
    print ""
    done=1
}
{ print }
' "${SYNTH_TCL}" > "${SYNTH_TCL}.tmp" && mv "${SYNTH_TCL}.tmp" "${SYNTH_TCL}"

echo "  Created ${SYNTH_TCL}"

# ---------------------------------------------------------------------------
# Step 6: Verify HLS port names (reminder)
# ---------------------------------------------------------------------------

HLS_TOP_V="${IP_DIR}/myproject.v"
if [ -f "${HLS_TOP_V}" ]; then
    echo ""
    echo "=== HLS IP port list (verify against cl_top.sv instantiation) ==="
    grep -n "input\|output\|inout" "${HLS_TOP_V}" | head -20
    echo ""
fi

echo ""
echo "=== CL project ready at ${CL_DIR} ==="
echo ""
echo "Next steps:"
echo "  1. Compare HLS port names above against the myproject instantiation"
echo "     in ${CL_DESIGN_DIR}/cl_top.sv and update if they differ."
echo "  2. Build the DCP:"
echo "       export CL_DIR=${CL_DIR}"
echo "       cd ${CL_DIR}/build/scripts"
echo "       python3 aws_build_dcp_from_cl.py --cl cl_mini_unet"
echo "  3. Create AFI:"
echo "       aws ec2 create-fpga-image \\"
echo "         --name mini-unet-ilt \\"
echo "         --input-storage-location Bucket=<bucket>,Key=<dcp-tar>"
echo "  4. Load AFI on F2:"
echo "       sudo fpga-load-local-image -S 0 -I <agfi-id>"
echo ""
echo "For Vivado IPI (GUI) flow, see:"
echo "  ${HDK_DIR}/hdk/docs/IPI-GUI-Vivado-Setup.md"
