#!/usr/bin/env python3

""" Run seizure detection test on Opal Kelly board using synthetic neural data """

import os
import sys
import argparse
import random
import time
from pathlib import Path

# Add synthetic data generator to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic import generate_data_intan16, NUM_CHANNELS

# Add Opal Kelly FrontPanel Python API to path (default, change if updated)
# Detect if running in WSL and use appropriate path format
if os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower():
    # Running in WSL
    OK_PYTHON_API = "/mnt/c/Program Files/Opal Kelly/FrontPanelUSB/API/Python/x64"
    OK_DLL_DIR = "/mnt/c/Program Files/Opal Kelly/FrontPanelUSB/API/lib/x64"
else:
    # Running in Windows
    OK_PYTHON_API = r"C:\Program Files\Opal Kelly\FrontPanelUSB\API\Python\x64"
    OK_DLL_DIR = r"C:\Program Files\Opal Kelly\FrontPanelUSB\API\lib\x64"

if os.path.exists(OK_PYTHON_API) and OK_PYTHON_API not in sys.path:
    sys.path.insert(0, OK_PYTHON_API)

# Add DLL directory to Windows DLL search path (Python 3.8+)
if os.path.exists(OK_DLL_DIR):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(OK_DLL_DIR)
    else:
        # Fallback for older Python versions: add to PATH
        dll_path = os.environ.get('PATH', '')
        if OK_DLL_DIR not in dll_path:
            os.environ['PATH'] = OK_DLL_DIR + os.pathsep + dll_path

import ok

PIPE_IN_ADDR = 0x80   # matches okPipeIn in datapath endpoint address
PIPE_OUT_ADDR = 0xA0  # matches okPipeOut in datapath endpoint address
WIREIN_CTRL = 0x00    # reset(31)
WIREIN_TS_LO = 0x01   # timestamp lower 32 bits
WIREIN_TS_HI = 0x02   # timestamp upper 32 bits
WIREOUT_TS_LO = 0x20  # timestamp lower 32 bits (WireOut range 0x20-0x3F)
WIREOUT_TS_HI = 0x21  # timestamp upper 32 bits

# Data format: 32 channels × 128 samples per chunk = 4096 samples = 16384 bytes per chunk
# Each sample is packed into a 32-bit word: [21:16]=channel_id, [15:0]=sample
SAMPLES_PER_CHUNK = 128  # samples per channel per chunk
CHUNK_BYTES = NUM_CHANNELS * SAMPLES_PER_CHUNK * 4  # 32 * 128 * 4 = 16384 bytes

HERE = Path(__file__).resolve().parent

def ensure_multiple_of_16(data: bytes) -> bytes:
    """Ensure data length is multiple of 16 bytes (required for PipeIn)"""
    rem = len(data) % 16
    if rem == 0:
        return data
    pad = 16 - rem
    return data + bytes(pad)

def generate_synthetic_data_chunks(num_chunks=450, samples_per_channel=60000, seed=None):
    """
    Generate synthetic neural data and format into chunks for FPGA.
    
    Returns:
        List of byte arrays, each representing one chunk formatted for First.sv.
        Format: Each 32-bit word contains:
            [15:0]  : 16-bit sample (ADC code)
            [21:16] : 6-bit channel_id (0-31)
            [31:22] : unused (set to 0)
        Data is interleaved: [ch0_sample0, ch0_sample1, ..., ch0_sample127, ch1_sample0, ..., ch31_sample127]
    """
    # Generate data for all 32 channels
    # This creates a flat array: [ch0_0, ..., ch0_59999, ch1_0, ..., ch31_59999]
    all_data_16 = generate_data_intan16(
        NUM_CHANNELS,
        samples_per_channel,
        sample_rate=1000.0,
        enable_seizures=True,
    )
    
    chunks = []
    for chunk_id in range(num_chunks):
        chunk_bytes = bytearray()
        
        # For each channel, extract 128 samples starting at chunk_id * 128
        for ch in range(NUM_CHANNELS):
            for sample_in_chunk in range(SAMPLES_PER_CHUNK):
                sample_idx = chunk_id * SAMPLES_PER_CHUNK + sample_in_chunk
                idx = ch * samples_per_channel + sample_idx
                
                if idx < len(all_data_16):
                    code16 = int(all_data_16[idx])
                else:
                    # Pad with mid-point value if we run out of data
                    code16 = 32768
                
                # Format as 32-bit word: [31:22]=0, [21:16]=channel_id, [15:0]=sample
                word32 = (ch << 16) | code16
                
                # Convert to little-endian bytes (4 bytes per word)
                chunk_bytes.extend(word32.to_bytes(4, byteorder='little'))
        
        chunks.append(bytes(chunk_bytes))
    
    return chunks

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream synthetic neural data to Opal Kelly FPGA and read seizure detection events."
    )
    p.add_argument("--bitfile", "-f", type=str, default=None,
                   help="Path to .bit file (optional; if omitted, won't reprogram device)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible synthetic data")
    p.add_argument("--chunks", type=int, default=450,
                   help="Number of chunks to send (default: 450 = ~57.6 seconds at 1kHz)")
    p.add_argument("--samples", type=int, default=60000,
                   help="Samples per channel to generate (default: 60000 = 60 seconds at 1kHz)")
    p.add_argument("--log", type=str, default=str(HERE / "run_halo_log.txt"),
                   help="Log file path (overwritten each run)")
    return p.parse_args()

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        # Also seed numpy if available (for synthetic.py)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass
    
    with open(args.log, "w", encoding="utf-8") as lf:
        def log(msg: str):
            print(msg)
            lf.write(msg + "\n")
            lf.flush()
        
        log("=" * 70)
        log("Opal Kelly Seizure Detection Test with Synthetic Neural Data")
        log("=" * 70)
        
        # Generate synthetic data chunks
        log(f"\nGenerating {args.chunks} chunks of synthetic neural data...")
        log(f"  Samples per channel: {args.samples}")
        log(f"  Samples per chunk per channel: {SAMPLES_PER_CHUNK}")
        log(f"  Bytes per chunk: {CHUNK_BYTES} (32-bit words: {NUM_CHANNELS * SAMPLES_PER_CHUNK} words)")
        
        chunks = generate_synthetic_data_chunks(
            num_chunks=args.chunks,
            samples_per_channel=args.samples,
            seed=args.seed
        )
        log(f"Generated {len(chunks)} chunks ({len(chunks) * CHUNK_BYTES} total bytes)")
        
        # Initialize Opal Kelly device
        dev = ok.okCFrontPanel()
        
        # Open first available device
        count = dev.GetDeviceCount()
        log(f"\nDevice count: {count}")
        
        if count <= 0:
            log("ERROR: No Opal Kelly devices found.")
            print(f"FAILED - see log: {args.log}")
            sys.exit(1)
        
        serial = dev.GetDeviceListSerial(0)
        rc = dev.OpenBySerial(serial)
        log(f"OpenBySerial({serial}) rc={rc}")
        
        if rc != ok.okCFrontPanel.NoError:
            log(f"ERROR: OpenBySerial failed: {ok.okCFrontPanel.GetErrorString(rc)}")
            print(f"FAILED - see log: {args.log}")
            sys.exit(1)
        
        # Optionally program bitfile
        if args.bitfile:
            bit = os.path.abspath(args.bitfile)
            log(f"\nConfiguring FPGA with {bit} ...")
            rc = dev.ConfigureFPGA(bit)
            log(f"ConfigureFPGA rc={rc}")
            
            if rc != ok.okCFrontPanel.NoError:
                log(f"ERROR: ConfigureFPGA failed: {ok.okCFrontPanel.GetErrorString(rc)}")
                print(f"FAILED - see log: {args.log}")
                sys.exit(1)
            log("FPGA configured successfully")
        else:
            log("\nSkipping FPGA configuration (no --bitfile specified)")
        
        # Apply control wires: pulse reset and set timestamp
        ctrl_reset = 0x8000_0000
        ctrl_release = 0x0000_0000
        ts = random.getrandbits(64) if args.seed is None else args.seed
        
        dev.SetWireInValue(WIREIN_TS_LO, ts & 0xFFFFFFFF, 0xFFFFFFFF)
        dev.SetWireInValue(WIREIN_TS_HI, (ts >> 32) & 0xFFFFFFFF, 0xFFFFFFFF)
        
        # Pulse reset high
        dev.SetWireInValue(WIREIN_CTRL, ctrl_reset, 0xFFFFFFFF)
        rc = dev.UpdateWireIns()
        log(f"\nUpdateWireIns (reset assert, ts set) rc={rc}, ts=0x{ts:016X}")
        
        if rc != ok.okCFrontPanel.NoError:
            log(f"ERROR: UpdateWireIns failed: {ok.okCFrontPanel.GetErrorString(rc)}")
        
        # Release reset
        dev.SetWireInValue(WIREIN_CTRL, ctrl_release, 0xFFFFFFFF)
        rc = dev.UpdateWireIns()
        log(f"UpdateWireIns (reset deassert) rc={rc}")
        
        if rc != ok.okCFrontPanel.NoError:
            log(f"ERROR: UpdateWireIns failed: {ok.okCFrontPanel.GetErrorString(rc)}")
        
        # Send chunks to PipeIn
        log(f"\nSending {len(chunks)} chunks to PipeIn 0x{PIPE_IN_ADDR:02X}...")
        total_bytes_sent = 0
        
        for chunk_idx, chunk_data in enumerate(chunks):
            # Ensure chunk is multiple of 16 bytes
            padded_chunk = ensure_multiple_of_16(chunk_data)
            # Convert to bytearray (required by Opal Kelly API)
            padded_chunk = bytearray(padded_chunk)
            
            status = dev.WriteToPipeIn(PIPE_IN_ADDR, padded_chunk)
            
            if status < 0:
                log(f"ERROR: WriteToPipeIn failed for chunk {chunk_idx}: "
                    f"{ok.okCFrontPanel.GetErrorString(dev.GetLastError())}")
                print(f"FAILED - see log: {args.log}")
                sys.exit(1)
            
            total_bytes_sent += len(padded_chunk)
            
            if (chunk_idx + 1) % 50 == 0:
                log(f"  Sent {chunk_idx + 1}/{len(chunks)} chunks ({total_bytes_sent} bytes)")
        
        log(f"Successfully sent all {len(chunks)} chunks ({total_bytes_sent} total bytes)")
        
        # Wait a bit for processing
        time.sleep(0.1)
        
        # Read back detection events from PipeOut
        # Allocate buffer large enough for many events (each event is 4 bytes)
        max_events = 10000
        event_buffer_size = max_events * 4
        
        log(f"\nReading detection events from PipeOut 0x{PIPE_OUT_ADDR:02X}...")
        buf = bytearray(event_buffer_size)
        status = dev.ReadFromPipeOut(PIPE_OUT_ADDR, buf)
        
        if status < 0:
            log(f"ERROR: ReadFromPipeOut failed: {ok.okCFrontPanel.GetErrorString(dev.GetLastError())}")
            print(f"FAILED - see log: {args.log}")
            sys.exit(1)
        
        # status is the number of bytes actually read
        out = bytes(buf[:max(status, 0)])
        
        # Read timestamp out
        rc = dev.UpdateWireOuts()
        ts_out_lo = dev.GetWireOutValue(WIREOUT_TS_LO)
        ts_out_hi = dev.GetWireOutValue(WIREOUT_TS_HI)
        ts_out = (ts_out_hi << 32) | ts_out_lo
        
        log(f"Successfully read {len(out)} bytes from PipeOut")
        log(f"Config timestamp latched: 0x{ts_out:016X} "
            f"(low=0x{ts_out_lo:08X}, high=0x{ts_out_hi:08X})")
        
        # Parse events
        # Each 32-bit word encodes (from First.sv):
        #   [31]   : output_event (1 = seizure start, 0 = seizure end)
        #   [30:26]: channel_id (0-31, 5 bits)
        #   [25:0] : lower 26 bits of datapath output_timestamp
        
        words = [
            int.from_bytes(out[i:i+4], byteorder="little")
            for i in range(0, len(out), 4)
            if i + 4 <= len(out)
        ]
        
        log(f"\n" + "=" * 70)
        log(f"Index, EventBit, Channel, Timestamp26Hex, RawWordHex")
        log(f"(EventBit: 1 = seizure start, 0 = seizure end; Timestamp26Hex = lower 26 bits of datapath timestamp)")
        
        seizure_starts = 0
        seizure_ends = 0
        
        for idx, w in enumerate(words):
            # Extract fields according to First.sv format
            event_bit = (w >> 31) & 0x1
            channel_id = (w >> 26) & 0x1F  # bits [30:26]
            timestamp26 = w & 0x03FF_FFFF  # bits [25:0], 26 bits
            
            event_type = "START" if event_bit else "END"
            log(f"{idx:04d}, {event_bit}, {channel_id}, 0x{timestamp26:08X}, 0x{w:08X}")
            
            if event_bit:
                seizure_starts += 1
            else:
                seizure_ends += 1
        
        log("=" * 70)
        log(f"\nTotal events decoded: {len(words)}")
        log(f"Seizure starts: {seizure_starts}")
        log(f"Seizure ends:   {seizure_ends}")
        log("=" * 70)
        
        if len(words) == 0:
            log("\nWARNING: No detection events received. This could mean:")
            log("  1. No seizures were detected in the synthetic data")
            log("  2. Threshold is too high (check THRESHOLD_VALUE in datapath.sv)")
            log("  3. FPGA is not processing data correctly")
        else:
            log(f"\nSUCCESS: Received {len(words)} detection events")
    
    print(f"\nDONE - log written to {args.log}")

if __name__ == "__main__":
    main()
