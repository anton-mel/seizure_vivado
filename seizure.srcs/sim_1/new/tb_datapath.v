`timescale 1ns / 1ps
`default_nettype none

module tb_datapath;
    // Clock and reset
    reg clk = 1'b0;
    always #5 clk = ~clk; // 100 MHz

    reg rst = 1'b1;
    initial begin
        #50 rst = 1'b0;
    end

    // DUT interface
    reg         start = 1'b0;
    reg  [4:0]  win_size  = 5'd8;
    reg  [8:0]  step_size = 9'd3;
    reg  [4:0]  thr       = 5'd1;
    reg  [63:0] timestamp = 64'h0;
    reg         mem_we    = 1'b0;
    reg  [11:0] mem_waddr = 12'd0;
    reg  [7:0]  mem_wdata = 8'd0;
    wire [1279:0] detected_flat;
    wire          done;

    // Instantiate DUT
    datapath dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .win_size(win_size),
        .step_size(step_size),
        .thr(thr),
        .timestamp(timestamp),
        .mem_we(mem_we),
        .mem_waddr(mem_waddr),
        .mem_wdata(mem_wdata),
        .detected_flat(detected_flat)
    );

    integer i;

    initial begin
        // Wait for reset deassert
        @(negedge rst);

        // Load 4KB alternating 0x00 / 0xFF into DUT memory interface
        for (i = 0; i < 4096; i = i + 1) begin
            @(posedge clk);
            mem_we    <= 1'b1;
            mem_waddr <= i[11:0];
            mem_wdata <= (i[0] == 1'b0) ? 8'h00 : 8'hFF;
        end
        @(posedge clk);
        mem_we <= 1'b0;

        // Pulse start
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        // Wait for done and display results
        wait(done == 1'b1);
        $display("DONE asserted at time %0t", $time);
        $display("detected_flat = %h", detected_flat);

        // End simulation a few cycles later
        repeat (10) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire

