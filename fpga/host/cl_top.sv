// ============================================================================
// cl_top.sv — AWS F2 CL for MiniUNet HLS IP
//
// Based on CL_TEMPLATE.sv + verified against cl_ports.vh.
// Module name must match --cl argument: penumbra.
// All port names taken directly from cl_ports.vh.
//
// HLS IP port list (from myproject.v):
//   input  [1023:0] x_TDATA       — 64 × 16-bit samples per beat
//   output [15:0]   layer42_out_TDATA — 1 sample per beat
//   input           ap_clk, ap_rst_n (active-low)
//   input/output    x_TVALID, x_TREADY
//   input           ap_start; output ap_done, ap_ready, ap_idle
//   input/output    layer42_out_TREADY, layer42_out_TVALID
//
// PCIS address map:
//   0x0000_0000 – 0x0000_1FFF : Input  BRAM (8 KB = 64 × 1024-bit words)
//   0x0000_2000 – 0x0000_3FFF : Output BRAM (8 KB = 4096 × 16-bit words)
//
// OCL register map (32-bit, byte offsets):
//   0x00  CTRL        W  bit[0]=ap_start pulse, bit[1]=soft_reset
//   0x04  STATUS      R  bit[0]=ap_done (sticky), bit[1]=ap_idle
//   0x08  TILE_COUNT  R  tiles processed since reset
//   0x0C  ERROR_FLAGS R  reserved
// ============================================================================

module penumbra
    #(
      parameter EN_DDR = 0,
      parameter EN_HBM = 0
    )
    (
      `include "cl_ports.vh"
    );

`include "cl_id_defines.vh"
`include "penumbra_defines.vh"

// ============================================================================
// Required shell outputs
// ============================================================================

always_comb begin
    cl_sh_flr_done    = 'b1;
    cl_sh_status0     = 'b0;
    cl_sh_status1     = 'b0;
    cl_sh_status2     = 'b0;
    cl_sh_id0         = `CL_SH_ID0;
    cl_sh_id1         = `CL_SH_ID1;
    cl_sh_status_vled = 'b0;
    cl_sh_dma_wr_full = 'b0;
    cl_sh_dma_rd_full = 'b0;
end

// ============================================================================
// PCIM — unused, tie off
// ============================================================================

always_comb begin
    cl_sh_pcim_awid    = 'b0;
    cl_sh_pcim_awaddr  = 'b0;
    cl_sh_pcim_awlen   = 'b0;
    cl_sh_pcim_awsize  = 'b0;
    cl_sh_pcim_awburst = 'b0;
    cl_sh_pcim_awcache = 'b0;
    cl_sh_pcim_awlock  = 'b0;
    cl_sh_pcim_awprot  = 'b0;
    cl_sh_pcim_awqos   = 'b0;
    cl_sh_pcim_awuser  = 'b0;
    cl_sh_pcim_awvalid = 'b0;

    cl_sh_pcim_wid     = 'b0;
    cl_sh_pcim_wdata   = 'b0;
    cl_sh_pcim_wstrb   = 'b0;
    cl_sh_pcim_wlast   = 'b0;
    cl_sh_pcim_wuser   = 'b0;
    cl_sh_pcim_wvalid  = 'b0;

    cl_sh_pcim_bready  = 'b0;

    cl_sh_pcim_arid    = 'b0;
    cl_sh_pcim_araddr  = 'b0;
    cl_sh_pcim_arlen   = 'b0;
    cl_sh_pcim_arsize  = 'b0;
    cl_sh_pcim_arburst = 'b0;
    cl_sh_pcim_arcache = 'b0;
    cl_sh_pcim_arlock  = 'b0;
    cl_sh_pcim_arprot  = 'b0;
    cl_sh_pcim_arqos   = 'b0;
    cl_sh_pcim_aruser  = 'b0;
    cl_sh_pcim_arvalid = 'b0;

    cl_sh_pcim_rready  = 'b0;
end

// ============================================================================
// SDA — unused, tie off (signal prefix: sda_cl_* / cl_sda_*)
// ============================================================================

always_comb begin
    cl_sda_awready = 'b0;
    cl_sda_wready  = 'b0;
    cl_sda_bresp   = 'b0;
    cl_sda_bvalid  = 'b0;
    cl_sda_arready = 'b0;
    cl_sda_rdata   = 'b0;
    cl_sda_rresp   = 'b0;
    cl_sda_rvalid  = 'b0;
end

// ============================================================================
// DDR — required instantiation even when unused
// ============================================================================

sh_ddr #(.DDR_PRESENT(EN_DDR)) SH_DDR (
    .clk                       (clk_main_a0),
    .rst_n                     (),
    .stat_clk                  (clk_main_a0),
    .stat_rst_n                (),
    .CLK_DIMM_DP               (CLK_DIMM_DP),
    .CLK_DIMM_DN               (CLK_DIMM_DN),
    .M_ACT_N                   (M_ACT_N),
    .M_MA                      (M_MA),
    .M_BA                      (M_BA),
    .M_BG                      (M_BG),
    .M_CKE                     (M_CKE),
    .M_ODT                     (M_ODT),
    .M_CS_N                    (M_CS_N),
    .M_CLK_DN                  (M_CLK_DN),
    .M_CLK_DP                  (M_CLK_DP),
    .M_PAR                     (M_PAR),
    .M_DQ                      (M_DQ),
    .M_ECC                     (M_ECC),
    .M_DQS_DP                  (M_DQS_DP),
    .M_DQS_DN                  (M_DQS_DN),
    .cl_RST_DIMM_N             (RST_DIMM_N),
    .cl_sh_ddr_axi_awid        (),
    .cl_sh_ddr_axi_awaddr      (),
    .cl_sh_ddr_axi_awlen       (),
    .cl_sh_ddr_axi_awsize      (),
    .cl_sh_ddr_axi_awvalid     (),
    .cl_sh_ddr_axi_awburst     (),
    .cl_sh_ddr_axi_awuser      (),
    .cl_sh_ddr_axi_awready     (),
    .cl_sh_ddr_axi_wdata       (),
    .cl_sh_ddr_axi_wstrb       (),
    .cl_sh_ddr_axi_wlast       (),
    .cl_sh_ddr_axi_wvalid      (),
    .cl_sh_ddr_axi_wready      (),
    .cl_sh_ddr_axi_bid         (),
    .cl_sh_ddr_axi_bresp       (),
    .cl_sh_ddr_axi_bvalid      (),
    .cl_sh_ddr_axi_bready      (),
    .cl_sh_ddr_axi_arid        (),
    .cl_sh_ddr_axi_araddr      (),
    .cl_sh_ddr_axi_arlen       (),
    .cl_sh_ddr_axi_arsize      (),
    .cl_sh_ddr_axi_arvalid     (),
    .cl_sh_ddr_axi_arburst     (),
    .cl_sh_ddr_axi_aruser      (),
    .cl_sh_ddr_axi_arready     (),
    .cl_sh_ddr_axi_rid         (),
    .cl_sh_ddr_axi_rdata       (),
    .cl_sh_ddr_axi_rresp       (),
    .cl_sh_ddr_axi_rlast       (),
    .cl_sh_ddr_axi_rvalid      (),
    .cl_sh_ddr_axi_rready      (),
    .sh_ddr_stat_bus_addr      (),
    .sh_ddr_stat_bus_wdata     (),
    .sh_ddr_stat_bus_wr        (),
    .sh_ddr_stat_bus_rd        (),
    .sh_ddr_stat_bus_ack       (),
    .sh_ddr_stat_bus_rdata     (),
    .ddr_sh_stat_int           (),
    .sh_cl_ddr_is_ready        ()
);

// ============================================================================
// DDR stat outputs — not used, tie to zero (matches CL_TEMPLATE)
// ============================================================================

always_comb begin
    cl_sh_ddr_stat_ack   = 'b0;
    cl_sh_ddr_stat_rdata = 'b0;
    cl_sh_ddr_stat_int   = 'b0;
end

// ============================================================================
// Interrupts, JTAG, HBM monitor, PCIE — tie off
// ============================================================================

always_comb cl_sh_apppf_irq_req = 'b0;
always_comb tdo = 'b0;

always_comb begin
    hbm_apb_paddr_1   = 'b0; hbm_apb_pprot_1   = 'b0;
    hbm_apb_psel_1    = 'b0; hbm_apb_penable_1 = 'b0;
    hbm_apb_pwrite_1  = 'b0; hbm_apb_pwdata_1  = 'b0;
    hbm_apb_pstrb_1   = 'b0; hbm_apb_pready_1  = 'b0;
    hbm_apb_prdata_1  = 'b0; hbm_apb_pslverr_1 = 'b0;
    hbm_apb_paddr_0   = 'b0; hbm_apb_pprot_0   = 'b0;
    hbm_apb_psel_0    = 'b0; hbm_apb_penable_0 = 'b0;
    hbm_apb_pwrite_0  = 'b0; hbm_apb_pwdata_0  = 'b0;
    hbm_apb_pstrb_0   = 'b0; hbm_apb_pready_0  = 'b0;
    hbm_apb_prdata_0  = 'b0; hbm_apb_pslverr_0 = 'b0;
end

always_comb begin
    PCIE_EP_TXP    = 'b0; PCIE_EP_TXN    = 'b0;
    PCIE_RP_PERSTN = 'b0; PCIE_RP_TXP    = 'b0; PCIE_RP_TXN = 'b0;
end

// ============================================================================
// Input BRAM — 1024-bit wide, 64 words deep
//   Port A: PCIS write path  (two 512-bit beats → one 1024-bit word)
//   Port B: FSM read path    (1024-bit → x_TDATA)
// ============================================================================

logic [5:0]    ibram_addra, ibram_addrb;
logic [1023:0] ibram_dina,  ibram_doutb;
logic          ibram_ena,   ibram_enb,   ibram_wea;

xpm_memory_sdpram #(
    .ADDR_WIDTH_A       (6),
    .ADDR_WIDTH_B       (6),
    .BYTE_WRITE_WIDTH_A (1024),
    .WRITE_DATA_WIDTH_A (1024),
    .READ_DATA_WIDTH_B  (1024),
    .MEMORY_SIZE        (65536),
    .READ_LATENCY_B     (2),
    .MEMORY_PRIMITIVE   ("auto"),
    .CLOCKING_MODE      ("common_clock"),
    .MEMORY_INIT_FILE   ("none"),
    .MEMORY_INIT_PARAM  ("0")
) ibram (
    .clka (clk_main_a0), .clkb (clk_main_a0),
    .ena  (ibram_ena),   .enb  (ibram_enb),
    .wea  (ibram_wea),
    .addra(ibram_addra), .addrb(ibram_addrb),
    .dina (ibram_dina),  .doutb(ibram_doutb),
    .injectdbiterra(1'b0), .injectsbiterra(1'b0),
    .regceb(1'b1), .rstb(~rst_main_n), .sleep(1'b0),
    .dbiterrb(), .sbiterrb()
);

// ============================================================================
// Output BRAM — 16-bit wide, 4096 words deep
//   Port A: FSM write path   (layer42_out_TDATA, 1 sample/cycle)
//   Port B: PCIS read path   (pack 32 samples → 512-bit beat)
// ============================================================================

logic [11:0] obram_addra, obram_addrb;
logic [15:0] obram_dina,  obram_doutb;
logic        obram_ena,   obram_enb,   obram_wea;

xpm_memory_sdpram #(
    .ADDR_WIDTH_A       (12),
    .ADDR_WIDTH_B       (12),
    .BYTE_WRITE_WIDTH_A (16),
    .WRITE_DATA_WIDTH_A (16),
    .READ_DATA_WIDTH_B  (16),
    .MEMORY_SIZE        (65536),
    .READ_LATENCY_B     (2),
    .MEMORY_PRIMITIVE   ("auto"),
    .CLOCKING_MODE      ("common_clock"),
    .MEMORY_INIT_FILE   ("none"),
    .MEMORY_INIT_PARAM  ("0")
) obram (
    .clka (clk_main_a0), .clkb (clk_main_a0),
    .ena  (obram_ena),   .enb  (obram_enb),
    .wea  (obram_wea),
    .addra(obram_addra), .addrb(obram_addrb),
    .dina (obram_dina),  .doutb(obram_doutb),
    .injectdbiterra(1'b0), .injectsbiterra(1'b0),
    .regceb(1'b1), .rstb(~rst_main_n), .sleep(1'b0),
    .dbiterrb(), .sbiterrb()
);

// ============================================================================
// HLS IP signals
// ============================================================================

logic          hls_ap_start, hls_ap_done, hls_ap_idle, hls_ap_ready;
logic [1023:0] hls_in_tdata;
logic          hls_in_tvalid, hls_in_tready;
logic [15:0]   hls_out_tdata;
logic          hls_out_tvalid, hls_out_tready;

// ============================================================================
// OCL register file
// Signals from cl_ports.vh: ocl_cl_awaddr, ocl_cl_awvalid, ocl_cl_wdata,
// ocl_cl_wstrb, ocl_cl_wvalid, ocl_cl_bready, ocl_cl_araddr, ocl_cl_arvalid,
// ocl_cl_rready  (prefix ocl_cl_*, not sh_ocl_*)
// ============================================================================

logic [31:0] reg_status;
logic [31:0] reg_tile_count;
logic [31:0] reg_error_flags;
logic        ocl_ap_start_pulse;
logic        ocl_soft_reset;
logic        ocl_aw_done, ocl_w_done;
logic [31:0] ocl_awaddr_r;

always_ff @(posedge clk_main_a0) begin
    if (!rst_main_n) begin
        ocl_aw_done        <= 1'b0; ocl_w_done    <= 1'b0;
        ocl_awaddr_r       <= '0;
        cl_ocl_awready     <= 1'b0; cl_ocl_wready <= 1'b0;
        cl_ocl_bvalid      <= 1'b0; cl_ocl_bresp  <= 2'b00;
        cl_ocl_arready     <= 1'b0; cl_ocl_rvalid <= 1'b0;
        cl_ocl_rdata       <= '0;   cl_ocl_rresp  <= 2'b00;
        ocl_ap_start_pulse <= 1'b0; ocl_soft_reset<= 1'b0;
    end else begin
        ocl_ap_start_pulse <= 1'b0;
        ocl_soft_reset     <= 1'b0;
        cl_ocl_awready     <= 1'b0;
        cl_ocl_wready      <= 1'b0;

        // Write address (ocl_cl_awaddr / ocl_cl_awvalid)
        if (ocl_cl_awvalid && !ocl_aw_done) begin
            cl_ocl_awready <= 1'b1;
            ocl_awaddr_r   <= ocl_cl_awaddr;
            ocl_aw_done    <= 1'b1;
        end

        // Write data (ocl_cl_wdata / ocl_cl_wvalid)
        if (ocl_cl_wvalid && !ocl_w_done) begin
            cl_ocl_wready <= 1'b1;
            ocl_w_done    <= 1'b1;
            case (ocl_awaddr_r[3:0])
                4'h0: begin
                    if (ocl_cl_wdata[0]) ocl_ap_start_pulse <= 1'b1;
                    if (ocl_cl_wdata[1]) ocl_soft_reset     <= 1'b1;
                end
                default: ;
            endcase
        end

        // Write response (ocl_cl_bready)
        if (ocl_aw_done && ocl_w_done && !cl_ocl_bvalid) begin
            cl_ocl_bvalid <= 1'b1; cl_ocl_bresp <= 2'b00;
            ocl_aw_done   <= 1'b0; ocl_w_done   <= 1'b0;
        end else if (cl_ocl_bvalid && ocl_cl_bready)
            cl_ocl_bvalid <= 1'b0;

        // Read (ocl_cl_araddr / ocl_cl_arvalid / ocl_cl_rready)
        cl_ocl_arready <= 1'b0;
        if (ocl_cl_arvalid && !cl_ocl_rvalid) begin
            cl_ocl_arready <= 1'b1;
            cl_ocl_rvalid  <= 1'b1;
            cl_ocl_rresp   <= 2'b00;
            case (ocl_cl_araddr[3:0])
                4'h0: cl_ocl_rdata <= 32'h0;
                4'h4: cl_ocl_rdata <= reg_status;
                4'h8: cl_ocl_rdata <= reg_tile_count;
                4'hC: cl_ocl_rdata <= reg_error_flags;
                default: cl_ocl_rdata <= 32'hDEADBEEF;
            endcase
        end
        if (cl_ocl_rvalid && ocl_cl_rready)
            cl_ocl_rvalid <= 1'b0;
    end
end

// ============================================================================
// PCIS write path → Input BRAM
// Two 512-bit beats fill one 1024-bit BRAM word.
// ============================================================================

logic         pcis_aw_valid_r;
logic [511:0] pcis_beat_acc;
logic         pcis_beat_half;
logic [5:0]   pcis_wr_word;

always_ff @(posedge clk_main_a0) begin
    if (!rst_main_n) begin
        pcis_aw_valid_r        <= 1'b0;
        pcis_beat_acc          <= '0;
        pcis_beat_half         <= 1'b0;
        pcis_wr_word           <= '0;
        ibram_wea              <= 1'b0; ibram_ena <= 1'b0;
        cl_sh_dma_pcis_awready <= 1'b0;
        cl_sh_dma_pcis_wready  <= 1'b0;
        cl_sh_dma_pcis_bvalid  <= 1'b0;
        cl_sh_dma_pcis_bresp   <= 2'b00;
        cl_sh_dma_pcis_bid     <= '0;
    end else begin
        ibram_wea              <= 1'b0; ibram_ena <= 1'b0;
        cl_sh_dma_pcis_awready <= 1'b0;

        if (sh_cl_dma_pcis_awvalid && !pcis_aw_valid_r) begin
            cl_sh_dma_pcis_awready <= 1'b1;
            pcis_aw_valid_r        <= 1'b1;
            cl_sh_dma_pcis_bid     <= sh_cl_dma_pcis_awid;
            pcis_wr_word           <= sh_cl_dma_pcis_awaddr[12:7];
        end

        cl_sh_dma_pcis_wready <= pcis_aw_valid_r && !ibram_wea;
        if (sh_cl_dma_pcis_wvalid && pcis_aw_valid_r) begin
            if (!pcis_beat_half) begin
                pcis_beat_acc  <= sh_cl_dma_pcis_wdata;
                pcis_beat_half <= 1'b1;
            end else begin
                ibram_ena      <= 1'b1; ibram_wea <= 1'b1;
                ibram_addra    <= pcis_wr_word;
                ibram_dina     <= {sh_cl_dma_pcis_wdata, pcis_beat_acc};
                pcis_beat_half         <= 1'b0;
                pcis_aw_valid_r        <= 1'b0;
                cl_sh_dma_pcis_bvalid  <= 1'b1;
                cl_sh_dma_pcis_bresp   <= 2'b00;
            end
        end

        if (cl_sh_dma_pcis_bvalid && sh_cl_dma_pcis_bready)
            cl_sh_dma_pcis_bvalid <= 1'b0;
    end
end

// ============================================================================
// PCIS read path ← Output BRAM
// 32 × 16-bit words packed into one 512-bit beat.
// ============================================================================

logic [11:0] pcis_rd_addr;
logic [4:0]  pack_word_idx;
logic [511:0] rdata_accum;
logic        packing;
logic [1:0]  bram_rd_lat;

always_ff @(posedge clk_main_a0) begin
    if (!rst_main_n) begin
        pcis_rd_addr           <= '0;
        pack_word_idx          <= '0;
        rdata_accum            <= '0;
        packing                <= 1'b0;
        bram_rd_lat            <= '0;
        obram_enb              <= 1'b0;
        cl_sh_dma_pcis_arready <= 1'b0;
        cl_sh_dma_pcis_rvalid  <= 1'b0;
        cl_sh_dma_pcis_rdata   <= '0;
        cl_sh_dma_pcis_rresp   <= 2'b00;
        cl_sh_dma_pcis_rlast   <= 1'b0;
        cl_sh_dma_pcis_rid     <= '0;
        cl_sh_dma_pcis_ruser   <= '0;
    end else begin
        obram_enb              <= 1'b0;
        cl_sh_dma_pcis_arready <= 1'b0;

        if (!packing && !cl_sh_dma_pcis_rvalid) begin
            cl_sh_dma_pcis_arready <= 1'b1;
            if (sh_cl_dma_pcis_arvalid) begin
                cl_sh_dma_pcis_arready <= 1'b0;
                cl_sh_dma_pcis_rid     <= sh_cl_dma_pcis_arid;
                pcis_rd_addr  <= sh_cl_dma_pcis_araddr[13:1];
                pack_word_idx <= '0;
                bram_rd_lat   <= '0;
                rdata_accum   <= '0;
                packing       <= 1'b1;
            end
        end

        if (packing) begin
            obram_enb    <= 1'b1;
            obram_addrb  <= pcis_rd_addr;
            pcis_rd_addr <= pcis_rd_addr + 1;

            if (bram_rd_lat < 2) begin
                bram_rd_lat <= bram_rd_lat + 1;
            end else begin
                rdata_accum[pack_word_idx * 16 +: 16] <= obram_doutb;
                pack_word_idx <= pack_word_idx + 1;
                if (pack_word_idx == 5'd31) begin
                    packing                <= 1'b0;
                    cl_sh_dma_pcis_rvalid  <= 1'b1;
                    cl_sh_dma_pcis_rdata   <= rdata_accum;
                    cl_sh_dma_pcis_rresp   <= 2'b00;
                    cl_sh_dma_pcis_rlast   <= 1'b1;
                    cl_sh_dma_pcis_ruser   <= '0;
                end
            end
        end

        if (cl_sh_dma_pcis_rvalid && sh_cl_dma_pcis_rready) begin
            cl_sh_dma_pcis_rvalid <= 1'b0;
            cl_sh_dma_pcis_rlast  <= 1'b0;
        end
    end
end

// ============================================================================
// Bridge FSM
//   IDLE → BRAM_LATCH → STREAM_IN → WAIT_DONE → DONE → IDLE
// ============================================================================

typedef enum logic [2:0] {
    S_IDLE       = 3'd0,
    S_BRAM_LATCH = 3'd1,
    S_STREAM_IN  = 3'd2,
    S_WAIT_DONE  = 3'd3,
    S_DONE       = 3'd4
} fsm_state_t;

localparam HLS_IN_BEATS = 64;

fsm_state_t  fsm_state;
logic [5:0]  fsm_rd_ptr;
logic [11:0] fsm_wr_ptr;
logic [1:0]  fsm_lat_cnt;

always_ff @(posedge clk_main_a0) begin
    if (!rst_main_n || ocl_soft_reset) begin
        fsm_state       <= S_IDLE;
        fsm_rd_ptr      <= '0; fsm_wr_ptr  <= '0; fsm_lat_cnt <= '0;
        hls_ap_start    <= 1'b0;
        ibram_enb       <= 1'b0;
        obram_ena       <= 1'b0; obram_wea  <= 1'b0;
        reg_status      <= 32'h2;
        reg_tile_count  <= '0;
        reg_error_flags <= '0;
    end else begin
        ibram_enb <= 1'b0;
        obram_ena <= 1'b0; obram_wea <= 1'b0;

        if (hls_out_tvalid) begin
            obram_ena   <= 1'b1; obram_wea <= 1'b1;
            obram_addra <= fsm_wr_ptr;
            obram_dina  <= hls_out_tdata;
            fsm_wr_ptr  <= fsm_wr_ptr + 1;
        end

        case (fsm_state)
            S_IDLE: begin
                hls_ap_start  <= 1'b0;
                reg_status[1] <= 1'b1;
                if (ocl_ap_start_pulse) begin
                    fsm_rd_ptr    <= '0; fsm_wr_ptr <= '0; fsm_lat_cnt <= '0;
                    reg_status[0] <= 1'b0; reg_status[1] <= 1'b0;
                    hls_ap_start  <= 1'b1;
                    fsm_state     <= S_BRAM_LATCH;
                end
            end
            S_BRAM_LATCH: begin
                ibram_enb   <= 1'b1;
                ibram_addrb <= fsm_rd_ptr;
                fsm_rd_ptr  <= fsm_rd_ptr + 1;
                fsm_lat_cnt <= fsm_lat_cnt + 1;
                if (fsm_lat_cnt == 2'd1) fsm_state <= S_STREAM_IN;
            end
            S_STREAM_IN: begin
                if (fsm_rd_ptr < HLS_IN_BEATS) begin
                    ibram_enb   <= 1'b1;
                    ibram_addrb <= fsm_rd_ptr;
                end
                if (hls_in_tvalid && hls_in_tready) begin
                    fsm_rd_ptr <= fsm_rd_ptr + 1;
                    if (fsm_rd_ptr == 6'(HLS_IN_BEATS - 1)) begin
                        hls_ap_start <= 1'b0;
                        fsm_state    <= S_WAIT_DONE;
                    end
                end
            end
            S_WAIT_DONE: begin
                if (hls_ap_done) begin
                    reg_tile_count <= reg_tile_count + 1;
                    fsm_state      <= S_DONE;
                end
            end
            S_DONE: begin
                reg_status[0] <= 1'b1;
                reg_status[1] <= 1'b1;
                fsm_state     <= S_IDLE;
            end
            default: fsm_state <= S_IDLE;
        endcase
    end
end

logic in_valid_d1, in_valid_d2;
always_ff @(posedge clk_main_a0) begin
    in_valid_d1 <= (fsm_state == S_STREAM_IN);
    in_valid_d2 <= in_valid_d1;
end

assign hls_in_tdata   = ibram_doutb;
assign hls_in_tvalid  = in_valid_d2;
assign hls_out_tready = 1'b1;

// ============================================================================
// HLS IP instantiation
// ============================================================================

myproject hls_ip (
    .ap_clk              (clk_main_a0),
    .ap_rst_n            (rst_main_n),
    .ap_start            (hls_ap_start),
    .ap_done             (hls_ap_done),
    .ap_idle             (hls_ap_idle),
    .ap_ready            (hls_ap_ready),
    .x_TDATA             (hls_in_tdata),
    .x_TVALID            (hls_in_tvalid),
    .x_TREADY            (hls_in_tready),
    .layer42_out_TDATA   (hls_out_tdata),
    .layer42_out_TVALID  (hls_out_tvalid),
    .layer42_out_TREADY  (hls_out_tready)
);

endmodule // penumbra
