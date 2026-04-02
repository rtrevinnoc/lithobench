#!/usr/bin/env bash
# ============================================================================
# F2 FPGA Deployment Script for MiniUNet HLS IP
#
# Integrates the hls4ml-generated Vitis HLS IP into the existing "penumbra"
# CL project (created via create_new_cl.py) and builds a Design Checkpoint
# (DCP) for AFI creation.
#
# Prerequisites:
#   - AWS F2 FPGA Developer AMI (with Vivado 2025.x)
#   - aws-fpga repo cloned and HDK set up:
#       git clone https://github.com/aws/aws-fpga.git -b f2
#       source aws-fpga/hdk_setup.sh
#   - CL project already created:
#       cd $AWS_FPGA_REPO_DIR/hdk/cl/examples
#       ./create_new_cl.py --new_cl_name penumbra
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
    export CL_DIR="${HDK_DIR}/cl/examples/penumbra"
fi

echo "=== F2 Penumbra (MiniUNet) Deployment ==="
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
# Step 2: Verify CL project exists
# ---------------------------------------------------------------------------

if [ ! -d "${CL_DIR}" ]; then
    echo "ERROR: CL project not found at ${CL_DIR}"
    echo ""
    echo "Create it first:"
    echo "  cd \${AWS_FPGA_REPO_DIR}/hdk/cl/examples"
    echo "  ./create_new_cl.py --new_cl_name penumbra"
    exit 1
fi

CL_DESIGN_DIR="${CL_DIR}/design"

if [ ! -d "${CL_DESIGN_DIR}" ]; then
    echo "ERROR: design/ directory not found at ${CL_DESIGN_DIR}"
    exit 1
fi

echo "CL project verified at ${CL_DIR}"

# ---------------------------------------------------------------------------
# Step 3: Copy HLS RTL into CL design directory
# ---------------------------------------------------------------------------

if [ -d "${HLS_RTL_DIR}" ]; then
    echo "Copying HLS RTL to CL design directory..."
    cp "${HLS_RTL_DIR}"/*.v  "${CL_DESIGN_DIR}/" 2>/dev/null || true
    cp "${HLS_RTL_DIR}"/*.sv "${CL_DESIGN_DIR}/" 2>/dev/null || true
    cp "${HLS_RTL_DIR}"/*.dat "${CL_DESIGN_DIR}/" 2>/dev/null || true
elif [ -f "${HLS_EXPORT}" ]; then
    echo "Extracting HLS export to CL design directory..."
    unzip -o "${HLS_EXPORT}" -d "${CL_DESIGN_DIR}/"
fi

echo "  HLS IP installed at ${CL_DESIGN_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Install penumbra.sv (AXI bridge + BRAM + FSM) as the CL top level
# ---------------------------------------------------------------------------

# cl_top.sv lives alongside this script in fpga/host/ — it defines the
# "penumbra" module that wraps the HLS IP with PCIS/OCL interfaces.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CL_TOP_SRC="${SCRIPT_DIR}/host/cl_top.sv"

if [ -f "${CL_TOP_SRC}" ]; then
    echo "Installing cl_top.sv as penumbra.sv (CL top level)..."
    cp "${CL_TOP_SRC}" "${CL_DESIGN_DIR}/penumbra.sv"
    echo "  Installed ${CL_DESIGN_DIR}/penumbra.sv"
else
    echo "WARNING: ${CL_TOP_SRC} not found. penumbra.sv must be installed manually."
fi

# ---------------------------------------------------------------------------
# Step 5: Patch synth_penumbra.tcl to include HLS IP verilog files
#
# The AWS build system expects synth_${CL}.tcl (i.e. synth_penumbra.tcl).
# The file-list auto-generation only applies to simulation (verif/), not to
# the Vivado synthesis build.  We must add read_verilog commands for the HLS
# IP files before synth_design runs.
# ---------------------------------------------------------------------------

SCRIPTS_DIR="${CL_DIR}/build/scripts"
SYNTH_TCL="${SCRIPTS_DIR}/synth_penumbra.tcl"

if [ ! -f "${SYNTH_TCL}" ]; then
    echo "ERROR: synth_penumbra.tcl not found at ${SYNTH_TCL}"
    exit 1
fi

# Only patch if not already patched
if grep -q "hls4ml MiniUNet HLS IP" "${SYNTH_TCL}"; then
    echo "synth_penumbra.tcl already patched — skipping."
else
    echo "Patching synth_penumbra.tcl with HLS IP file reads..."

    # Insert HLS read_verilog block right before the first
    # "#---- End of section replaced by User ----" marker.
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

    echo "  Patched ${SYNTH_TCL}"
fi

# ---------------------------------------------------------------------------
# Step 6: Verify HLS port names (reminder)
# ---------------------------------------------------------------------------

HLS_TOP_V="${CL_DESIGN_DIR}/myproject.v"
if [ -f "${HLS_TOP_V}" ]; then
    echo ""
    echo "=== HLS IP port list (verify against penumbra.sv instantiation) ==="
    grep -n "input\|output\|inout" "${HLS_TOP_V}" | head -20
    echo ""
fi

echo ""
echo "=== CL project ready at ${CL_DIR} ==="
echo ""
echo "Next steps:"
echo "  1. Compare HLS port names above against the myproject instantiation"
echo "     in ${CL_DESIGN_DIR}/penumbra.sv and update if they differ."
echo "  2. Build the DCP:"
echo "       export CL_DIR=${CL_DIR}"
echo "       cd ${CL_DIR}/build/scripts"
echo "       ./aws_build_dcp_from_cl.py -c penumbra"
echo "  3. Create AFI:"
echo "       aws ec2 create-fpga-image \\"
echo "         --name penumbra-mini-unet-ilt \\"
echo "         --input-storage-location Bucket=<bucket>,Key=<dcp-tar>"
echo "  4. Load AFI on F2:"
echo "       sudo fpga-load-local-image -S 0 -I <agfi-id>"
echo ""
