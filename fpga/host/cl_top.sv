// ============================================================================
// cl_top.sv — AWS F2 Custom Logic Top-Level for MiniUNet HLS IP
//
// Verified HLS IP port list (from myproject.v):
//   input  [1023:0] x_TDATA;       — 64 samples × 16-bit per beat
//   output [15:0]   layer42_out_TDATA;  — 1 sample per beat
//   input           ap_clk;
//   input           ap_rst_n;      — active-LOW reset
//   input           x_TVALID;
//   output          x_TREADY;
//   input           ap_start;
//   output          layer42_out_TVALID;
//   input           layer42_out_TREADY;
//   output          ap_done;
//   output          ap_ready;
//   output          ap_idle;
//
// Data flow:
//
//   PCIS AXI4 write (512-bit) ─── 2-beat accumulator ─── Input BRAM (64×1024-bit)
//                                                               │
//                                                       FSM streams 64 beats
//                                                               │
//                                                   HLS IP x_TDATA (1024-bit in)
//                                                               │
//                                          HLS IP layer42_out_TDATA (16-bit out)
//                                                               │
//                                               Output BRAM (4096×16-bit)
//                                                               │
//   PCIS AXI4 read  (512-bit) ─── 32-word packer ─────────────┘
//
// PCIS address map (CL-internal):
//   0x0000_0000 – 0x0000_1FFF : Input BRAM  (8 KB = 64 × 1024-bit words)
//   0x0000_2000 – 0x0000_3FFF : Output BRAM (8 KB = 4096 × 16-bit words)
//
// OCL register map (32-bit, byte addresses):
//   0x00  CTRL        W  bit[0]=ap_start pulse, bit[1]=soft_reset
//   0x04  STATUS      R  bit[0]=ap_done (sticky), bit[1]=ap_idle
//   0x08  TILE_COUNT  R  tiles processed since reset
//   0x0C  ERROR_FLAGS R  unused, reserved
//
// ============================================================================

`default_nettype none
`timescale 1ns/1ps

module cl_top
(
    // -------------------------------------------------------------------
    // Global signals (from AWS F2 shell)
    // -------------------------------------------------------------------
    input  logic        clk_main_a0,       // 250 MHz main clock
    input  logic        rst_main_n_sync,   // Active-low synchronous reset

    // -------------------------------------------------------------------
    // PCIS AXI4 slave — 512-bit DMA path (host ↔ CL)
    // Adapt signal names to match the actual AWS F2 HDK template.
    // -------------------------------------------------------------------

    // Write address channel
    input  logic        sh_cl_dma_pcis_awvalid,
    output logic        cl_sh_dma_pcis_awready,
    input  logic [63:0] sh_cl_dma_pcis_awaddr,
    input  logic [7:0]  sh_cl_dma_pcis_awlen,
    input  logic [2:0]  sh_cl_dma_pcis_awsize,

    // Write data channel
    input  logic         sh_cl_dma_pcis_wvalid,
    output logic         cl_sh_dma_pcis_wready,
    input  logic [511:0] sh_cl_dma_pcis_wdata,
    input  logic [63:0]  sh_cl_dma_pcis_wstrb,
    input  logic         sh_cl_dma_pcis_wlast,

    // Write response channel
    output logic        cl_sh_dma_pcis_bvalid,
    input  logic        sh_cl_dma_pcis_bready,
    output logic [1:0]  cl_sh_dma_pcis_bresp,

    // Read address channel
    input  logic        sh_cl_dma_pcis_arvalid,
    output logic        cl_sh_dma_pcis_arready,
    input  logic [63:0] sh_cl_dma_pcis_araddr,
    input  logic [7:0]  sh_cl_dma_pcis_arlen,
    input  logic [2:0]  sh_cl_dma_pcis_arsize,

    // Read data channel
    output logic         cl_sh_dma_pcis_rvalid,
    input  logic         sh_cl_dma_pcis_rready,
    output logic [511:0] cl_sh_dma_pcis_rdata,
    output logic [1:0]   cl_sh_dma_pcis_rresp,
    output logic         cl_sh_dma_pcis_rlast,

    // -------------------------------------------------------------------
    // OCL AXI-Lite slave — 32-bit register access
    // -------------------------------------------------------------------
    input  logic        sh_ocl_awvalid,
    output logic        cl_ocl_awready,
    input  logic [31:0] sh_ocl_awaddr,

    input  logic        sh_ocl_wvalid,
    output logic        cl_ocl_wready,
    input  logic [31:0] sh_ocl_wdata,
    input  logic [3:0]  sh_ocl_wstrb,

    output logic        cl_ocl_bvalid,
    input  logic        sh_ocl_bready,
    output logic [1:0]  cl_ocl_bresp,

    input  logic        sh_ocl_arvalid,
    output logic        cl_ocl_arready,
    input  logic [31:0] sh_ocl_araddr,

    output logic        cl_ocl_rvalid,
    input  logic        sh_ocl_rready,
    output logic [31:0] cl_ocl_rdata,
    output logic [1:0]  cl_ocl_rresp
);

    // ===================================================================
    // Local aliases
    // ===================================================================
    logic clk  = clk_main_a0;
    logic rstn = rst_main_n_sync;  // active-low

    // ===================================================================
    // Parameters
    // ===================================================================
    // HLS input: x_TDATA = 1024 bits = 64 × 16-bit samples per beat
    // Tile:      64×64 = 4096 samples → 64 HLS input beats
    localparam HLS_IN_W      = 1024;
    localparam SAMPLES_PER_BEAT = HLS_IN_W / 16;   // 64
    localparam TILE_SAMPLES  = 4096;                // 64×64
    localparam HLS_IN_BEATS  = TILE_SAMPLES / SAMPLES_PER_BEAT;  // 64

    // PCIS write: 512-bit beat → 32 × 16-bit samples
    // Two PCIS beats fill one 1024-bit HLS input word
    localparam PCIS_W        = 512;
    localparam PCIS_SAMPS    = PCIS_W / 16;  // 32
    // Total PCIS write beats for full tile: 4096 / 32 = 128 (2 per BRAM word)

    // ===================================================================
    // Input BRAM — 1024-bit wide, 64 words deep
    //   Port A: PCIS write path (two 512-bit beats → one 1024-bit word)
    //   Port B: FSM read path   (1024-bit → x_TDATA)
    // ===================================================================
    logic [5:0]    ibram_addra, ibram_addrb;   // 6-bit for 64 words
    logic [1023:0] ibram_dina,  ibram_doutb;
    logic          ibram_ena,   ibram_enb;
    logic          ibram_wea;

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A       (6),
        .ADDR_WIDTH_B       (6),
        .BYTE_WRITE_WIDTH_A (1024),
        .WRITE_DATA_WIDTH_A (1024),
        .READ_DATA_WIDTH_B  (1024),
        .MEMORY_SIZE        (65536),    // 64 × 1024 bits
        .READ_LATENCY_B     (2),
        .MEMORY_PRIMITIVE   ("auto"),
        .CLOCKING_MODE      ("common_clock"),
        .MEMORY_INIT_FILE   ("none"),
        .MEMORY_INIT_PARAM  ("0")
    ) ibram (
        .clka   (clk),
        .clkb   (clk),
        .ena    (ibram_ena),
        .enb    (ibram_enb),
        .wea    (ibram_wea),
        .addra  (ibram_addra),
        .addrb  (ibram_addrb),
        .dina   (ibram_dina),
        .doutb  (ibram_doutb),
        .injectdbiterra (1'b0),
        .injectsbiterra (1'b0),
        .regceb (1'b1),
        .rstb   (~rstn),
        .sleep  (1'b0),
        .dbiterrb (),
        .sbiterrb ()
    );

    // ===================================================================
    // Output BRAM — 16-bit wide, 4096 words deep
    //   Port A: FSM write path  (layer42_out_TDATA, 1 sample/cycle)
    //   Port B: PCIS read path  (pack 32 samples → 512-bit beat)
    // ===================================================================
    logic [11:0] obram_addra, obram_addrb;
    logic [15:0] obram_dina,  obram_doutb;
    logic        obram_ena,   obram_enb;
    logic        obram_wea;

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A       (12),
        .ADDR_WIDTH_B       (12),
        .BYTE_WRITE_WIDTH_A (16),
        .WRITE_DATA_WIDTH_A (16),
        .READ_DATA_WIDTH_B  (16),
        .MEMORY_SIZE        (65536),    // 4096 × 16 bits
        .READ_LATENCY_B     (2),
        .MEMORY_PRIMITIVE   ("auto"),
        .CLOCKING_MODE      ("common_clock"),
        .MEMORY_INIT_FILE   ("none"),
        .MEMORY_INIT_PARAM  ("0")
    ) obram (
        .clka   (clk),
        .clkb   (clk),
        .ena    (obram_ena),
        .enb    (obram_enb),
        .wea    (obram_wea),
        .addra  (obram_addra),
        .addrb  (obram_addrb),
        .dina   (obram_dina),
        .doutb  (obram_doutb),
        .injectdbiterra (1'b0),
        .injectsbiterra (1'b0),
        .regceb (1'b1),
        .rstb   (~rstn),
        .sleep  (1'b0),
        .dbiterrb (),
        .sbiterrb ()
    );

    // ===================================================================
    // HLS IP signals
    // ===================================================================
    logic          hls_ap_start, hls_ap_done, hls_ap_idle, hls_ap_ready;
    logic [1023:0] hls_in_tdata;
    logic          hls_in_tvalid, hls_in_tready;
    logic [15:0]   hls_out_tdata;
    logic          hls_out_tvalid, hls_out_tready;

    // ===================================================================
    // OCL register file
    // ===================================================================
    logic [31:0] reg_status;      // STATUS — read by host
    logic [31:0] reg_tile_count;  // TILE_COUNT — debug
    logic [31:0] reg_error_flags; // ERROR_FLAGS — reserved

    logic ocl_ap_start_pulse;
    logic ocl_soft_reset;

    // OCL write path
    logic ocl_aw_done, ocl_w_done;
    logic [31:0] ocl_awaddr_r;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            ocl_aw_done        <= 1'b0;
            ocl_w_done         <= 1'b0;
            ocl_awaddr_r       <= '0;
            cl_ocl_awready     <= 1'b0;
            cl_ocl_wready      <= 1'b0;
            cl_ocl_bvalid      <= 1'b0;
            cl_ocl_bresp       <= 2'b00;
            ocl_ap_start_pulse <= 1'b0;
            ocl_soft_reset     <= 1'b0;
        end else begin
            ocl_ap_start_pulse <= 1'b0;
            ocl_soft_reset     <= 1'b0;
            cl_ocl_awready     <= 1'b0;
            cl_ocl_wready      <= 1'b0;

            if (sh_ocl_awvalid && !ocl_aw_done) begin
                cl_ocl_awready <= 1'b1;
                ocl_awaddr_r   <= sh_ocl_awaddr;
                ocl_aw_done    <= 1'b1;
            end

            if (sh_ocl_wvalid && !ocl_w_done) begin
                cl_ocl_wready <= 1'b1;
                ocl_w_done    <= 1'b1;
                case (ocl_awaddr_r[3:0])
                    4'h0: begin
                        if (sh_ocl_wdata[0]) ocl_ap_start_pulse <= 1'b1;
                        if (sh_ocl_wdata[1]) ocl_soft_reset     <= 1'b1;
                    end
                    default: ;
                endcase
            end

            if (ocl_aw_done && ocl_w_done && !cl_ocl_bvalid) begin
                cl_ocl_bvalid <= 1'b1;
                cl_ocl_bresp  <= 2'b00;
                ocl_aw_done   <= 1'b0;
                ocl_w_done    <= 1'b0;
            end else if (cl_ocl_bvalid && sh_ocl_bready) begin
                cl_ocl_bvalid <= 1'b0;
            end
        end
    end

    // OCL read path
    always_ff @(posedge clk) begin
        if (!rstn) begin
            cl_ocl_arready <= 1'b0;
            cl_ocl_rvalid  <= 1'b0;
            cl_ocl_rdata   <= '0;
            cl_ocl_rresp   <= 2'b00;
        end else begin
            cl_ocl_arready <= 1'b0;
            if (sh_ocl_arvalid && !cl_ocl_rvalid) begin
                cl_ocl_arready <= 1'b1;
                cl_ocl_rvalid  <= 1'b1;
                cl_ocl_rresp   <= 2'b00;
                case (sh_ocl_araddr[3:0])
                    4'h0: cl_ocl_rdata <= 32'h0;
                    4'h4: cl_ocl_rdata <= reg_status;
                    4'h8: cl_ocl_rdata <= reg_tile_count;
                    4'hC: cl_ocl_rdata <= reg_error_flags;
                    default: cl_ocl_rdata <= 32'hDEADBEEF;
                endcase
            end
            if (cl_ocl_rvalid && sh_ocl_rready)
                cl_ocl_rvalid <= 1'b0;
        end
    end

    // ===================================================================
    // PCIS write path → Input BRAM
    //
    // Two consecutive 512-bit PCIS beats fill one 1024-bit BRAM word.
    // A 512-bit accumulator latches the first beat; on the second beat
    // the pair is written to BRAM and the word address increments.
    //
    // Address decode: writes to 0x0000–0x1FFF go to input BRAM.
    // ===================================================================
    logic         pcis_aw_valid_r;
    logic [63:0]  pcis_awaddr_r;
    logic [511:0] pcis_beat_acc;     // first-beat accumulator
    logic         pcis_beat_half;    // 0 = waiting for first, 1 = waiting for second
    logic [5:0]   pcis_wr_word;      // 1024-bit BRAM word index (0..63)

    always_ff @(posedge clk) begin
        if (!rstn) begin
            pcis_aw_valid_r        <= 1'b0;
            pcis_awaddr_r          <= '0;
            pcis_beat_acc          <= '0;
            pcis_beat_half         <= 1'b0;
            pcis_wr_word           <= '0;
            ibram_wea              <= 1'b0;
            ibram_ena              <= 1'b0;
            cl_sh_dma_pcis_awready <= 1'b0;
            cl_sh_dma_pcis_wready  <= 1'b0;
            cl_sh_dma_pcis_bvalid  <= 1'b0;
            cl_sh_dma_pcis_bresp   <= 2'b00;
        end else begin
            ibram_wea              <= 1'b0;
            ibram_ena              <= 1'b0;
            cl_sh_dma_pcis_awready <= 1'b0;

            // Accept write address
            if (sh_cl_dma_pcis_awvalid && !pcis_aw_valid_r) begin
                cl_sh_dma_pcis_awready <= 1'b1;
                pcis_awaddr_r          <= sh_cl_dma_pcis_awaddr;
                pcis_aw_valid_r        <= 1'b1;
                // BRAM word address: each word = 128 bytes; addr[13:7] selects word
                // (1024 bits = 128 bytes; 64 words; byte addr[6:0] within word)
                pcis_wr_word <= sh_cl_dma_pcis_awaddr[12:7];
            end

            // Accept write data
            cl_sh_dma_pcis_wready <= pcis_aw_valid_r && !ibram_wea;
            if (sh_cl_dma_pcis_wvalid && pcis_aw_valid_r) begin
                if (!pcis_beat_half) begin
                    // First 512-bit half: latch and wait for second
                    pcis_beat_acc  <= sh_cl_dma_pcis_wdata;
                    pcis_beat_half <= 1'b1;
                end else begin
                    // Second 512-bit half: write full 1024-bit word to BRAM
                    ibram_ena    <= 1'b1;
                    ibram_wea    <= 1'b1;
                    ibram_addra  <= pcis_wr_word;
                    ibram_dina   <= {sh_cl_dma_pcis_wdata, pcis_beat_acc};
                    pcis_beat_half  <= 1'b0;
                    pcis_aw_valid_r <= 1'b0;
                    // Send write response after BRAM write
                    cl_sh_dma_pcis_bvalid <= 1'b1;
                    cl_sh_dma_pcis_bresp  <= 2'b00;
                end
            end

            if (cl_sh_dma_pcis_bvalid && sh_cl_dma_pcis_bready)
                cl_sh_dma_pcis_bvalid <= 1'b0;
        end
    end

    // ===================================================================
    // PCIS read path ← Output BRAM
    //
    // Reads pack 32 × 16-bit BRAM words into one 512-bit PCIS beat.
    // bram_rd_lat handles the 2-cycle BRAM read latency.
    // ===================================================================
    logic [11:0] pcis_rd_addr;
    logic [4:0]  pack_word_idx;
    logic [511:0] rdata_accum;
    logic        packing;
    logic [1:0]  bram_rd_lat;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            pcis_rd_addr            <= '0;
            pack_word_idx           <= '0;
            rdata_accum             <= '0;
            packing                 <= 1'b0;
            bram_rd_lat             <= '0;
            obram_enb               <= 1'b0;
            cl_sh_dma_pcis_arready  <= 1'b0;
            cl_sh_dma_pcis_rvalid   <= 1'b0;
            cl_sh_dma_pcis_rdata    <= '0;
            cl_sh_dma_pcis_rresp    <= 2'b00;
            cl_sh_dma_pcis_rlast    <= 1'b0;
        end else begin
            obram_enb              <= 1'b0;
            cl_sh_dma_pcis_arready <= 1'b0;

            if (!packing && !cl_sh_dma_pcis_rvalid) begin
                cl_sh_dma_pcis_arready <= 1'b1;
                if (sh_cl_dma_pcis_arvalid) begin
                    cl_sh_dma_pcis_arready <= 1'b0;
                    // Byte addr[13:1] → 16-bit word address (output BRAM base = 0x2000)
                    pcis_rd_addr  <= (sh_cl_dma_pcis_araddr[13:1]);
                    pack_word_idx <= '0;
                    bram_rd_lat   <= '0;
                    rdata_accum   <= '0;
                    packing       <= 1'b1;
                end
            end

            if (packing) begin
                obram_enb   <= 1'b1;
                obram_addrb <= pcis_rd_addr;
                pcis_rd_addr <= pcis_rd_addr + 1;

                if (bram_rd_lat < 2) begin
                    bram_rd_lat <= bram_rd_lat + 1;
                end else begin
                    rdata_accum[pack_word_idx * 16 +: 16] <= obram_doutb;
                    pack_word_idx <= pack_word_idx + 1;
                    if (pack_word_idx == 5'd31) begin
                        packing <= 1'b0;
                        cl_sh_dma_pcis_rvalid <= 1'b1;
                        cl_sh_dma_pcis_rdata  <= rdata_accum;
                        cl_sh_dma_pcis_rresp  <= 2'b00;
                        cl_sh_dma_pcis_rlast  <= 1'b1;
                    end
                end
            end

            if (cl_sh_dma_pcis_rvalid && sh_cl_dma_pcis_rready) begin
                cl_sh_dma_pcis_rvalid <= 1'b0;
                cl_sh_dma_pcis_rlast  <= 1'b0;
            end
        end
    end

    // ===================================================================
    // Bridge FSM
    //
    // States:
    //   IDLE       : waiting for ap_start pulse from OCL register
    //   BRAM_LATCH : 2-cycle BRAM read latency warm-up for input BRAM
    //   STREAM_IN  : send 64 × 1024-bit beats to x_TDATA
    //   WAIT_DONE  : all input sent; capture output stream; wait ap_done
    //   DONE       : latch ap_done into STATUS, return to IDLE
    // ===================================================================
    typedef enum logic [2:0] {
        S_IDLE       = 3'd0,
        S_BRAM_LATCH = 3'd1,
        S_STREAM_IN  = 3'd2,
        S_WAIT_DONE  = 3'd3,
        S_DONE       = 3'd4
    } fsm_state_t;

    fsm_state_t  fsm_state;
    logic [5:0]  fsm_rd_ptr;   // input BRAM word index (0..63)
    logic [11:0] fsm_wr_ptr;   // output BRAM sample index (0..4095)
    logic [1:0]  fsm_lat_cnt;  // BRAM latency counter

    always_ff @(posedge clk) begin
        if (!rstn || ocl_soft_reset) begin
            fsm_state      <= S_IDLE;
            fsm_rd_ptr     <= '0;
            fsm_wr_ptr     <= '0;
            fsm_lat_cnt    <= '0;
            hls_ap_start   <= 1'b0;
            ibram_enb      <= 1'b0;
            obram_ena      <= 1'b0;
            obram_wea      <= 1'b0;
            reg_status     <= 32'h2;  // ap_idle = 1
            reg_tile_count <= '0;
            reg_error_flags <= '0;
        end else begin
            ibram_enb <= 1'b0;
            obram_ena <= 1'b0;
            obram_wea <= 1'b0;

            // Always accept HLS output — write each 16-bit sample to output BRAM
            if (hls_out_tvalid) begin
                obram_ena   <= 1'b1;
                obram_wea   <= 1'b1;
                obram_addra <= fsm_wr_ptr;
                obram_dina  <= hls_out_tdata;
                fsm_wr_ptr  <= fsm_wr_ptr + 1;
            end

            case (fsm_state)
                S_IDLE: begin
                    hls_ap_start  <= 1'b0;
                    reg_status[1] <= 1'b1;  // ap_idle
                    if (ocl_ap_start_pulse) begin
                        fsm_rd_ptr    <= '0;
                        fsm_wr_ptr    <= '0;
                        fsm_lat_cnt   <= '0;
                        reg_status[0] <= 1'b0;  // clear ap_done
                        reg_status[1] <= 1'b0;  // clear ap_idle
                        hls_ap_start  <= 1'b1;
                        fsm_state     <= S_BRAM_LATCH;
                    end
                end

                S_BRAM_LATCH: begin
                    // Pre-fetch first BRAM word; absorb 2-cycle read latency
                    ibram_enb   <= 1'b1;
                    ibram_addrb <= fsm_rd_ptr;
                    fsm_rd_ptr  <= fsm_rd_ptr + 1;
                    fsm_lat_cnt <= fsm_lat_cnt + 1;
                    if (fsm_lat_cnt == 2'd1)
                        fsm_state <= S_STREAM_IN;
                end

                S_STREAM_IN: begin
                    // Pre-fetch next BRAM word while current is being consumed
                    if (fsm_rd_ptr < HLS_IN_BEATS) begin
                        ibram_enb   <= 1'b1;
                        ibram_addrb <= fsm_rd_ptr;
                    end
                    // Advance on handshake
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
                        fsm_state      <= S_DONE;
                        reg_tile_count <= reg_tile_count + 1;
                    end
                end

                S_DONE: begin
                    reg_status[0] <= 1'b1;  // ap_done sticky
                    reg_status[1] <= 1'b1;  // ap_idle
                    fsm_state     <= S_IDLE;
                end

                default: fsm_state <= S_IDLE;
            endcase
        end
    end

    // ===================================================================
    // HLS AXI-Stream input handshake
    //
    // ibram_doutb is valid 2 cycles after the address is presented.
    // TVALID follows the same 2-cycle registered delay.
    // ===================================================================
    logic in_valid_d1, in_valid_d2;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            in_valid_d1 <= 1'b0;
            in_valid_d2 <= 1'b0;
        end else begin
            in_valid_d1 <= (fsm_state == S_STREAM_IN);
            in_valid_d2 <= in_valid_d1;
        end
    end

    assign hls_in_tdata  = ibram_doutb;
    assign hls_in_tvalid = in_valid_d2;

    // HLS output: always ready (output BRAM can always accept)
    assign hls_out_tready = 1'b1;

    // ===================================================================
    // HLS IP instantiation
    //
    // ap_rst_n is active-LOW and connects directly to rst_main_n_sync.
    // ===================================================================
    myproject hls_ip (
        .ap_clk               (clk),
        .ap_rst_n             (rstn),           // both active-low
        .ap_start             (hls_ap_start),
        .ap_done              (hls_ap_done),
        .ap_idle              (hls_ap_idle),
        .ap_ready             (hls_ap_ready),

        // AXI-Stream input (1024-bit, 64 samples per beat)
        .x_TDATA              (hls_in_tdata),
        .x_TVALID             (hls_in_tvalid),
        .x_TREADY             (hls_in_tready),

        // AXI-Stream output (16-bit, 1 sample per beat)
        .layer42_out_TDATA    (hls_out_tdata),
        .layer42_out_TVALID   (hls_out_tvalid),
        .layer42_out_TREADY   (hls_out_tready)
    );

endmodule

`default_nettype wire
