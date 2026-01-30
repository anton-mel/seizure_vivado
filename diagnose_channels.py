#!/usr/bin/env python3
"""
Diagnostic script to check channel data ordering and detect potential issues.
"""

import numpy as np
from pathlib import Path
from synthetic import generate_data_intan16, NUM_CHANNELS

def check_channel_data_ordering():
    """Check if channel data is correctly ordered in chunks."""
    print("=" * 70)
    print("Channel Data Ordering Diagnostic")
    print("=" * 70)
    
    SAMPLES_PER_CHUNK = 128
    samples_per_channel = 1000
    num_chunks = 10
    
    # Generate test data
    print(f"\n[1] Generating test data...")
    all_data_16 = generate_data_intan16(NUM_CHANNELS, samples_per_channel, 
                                       sample_rate=1000.0, enable_seizures=False)
    print(f"    Generated {len(all_data_16)} samples")
    
    # Generate chunks
    print(f"\n[2] Generating chunks...")
    chunks = []
    for chunk_id in range(num_chunks):
        chunk_bytes = bytearray()
        for ch in range(NUM_CHANNELS):
            for sample_in_chunk in range(SAMPLES_PER_CHUNK):
                sample_idx = chunk_id * SAMPLES_PER_CHUNK + sample_in_chunk
                idx = ch * samples_per_channel + sample_idx
                code16 = int(all_data_16[idx]) if idx < len(all_data_16) else 32768
                word32 = (ch << 16) | code16
                chunk_bytes.extend(word32.to_bytes(4, byteorder='little'))
        chunks.append(bytes(chunk_bytes))
    
    print(f"    Generated {len(chunks)} chunks")
    
    # Verify chunk structure
    print(f"\n[3] Verifying chunk structure...")
    for chunk_id in [0, 1]:
        chunk = chunks[chunk_id]
        print(f"\n    Chunk {chunk_id}:")
        
        # Check first few words
        for word_idx in [0, 127, 128, 255, 256, 383, 384]:
            if word_idx * 4 < len(chunk):
                word_bytes = chunk[word_idx*4:(word_idx+1)*4]
                word32 = int.from_bytes(word_bytes, byteorder='little')
                extracted_ch = (word32 >> 16) & 0x3F
                extracted_code = word32 & 0xFFFF
                expected_ch = word_idx // SAMPLES_PER_CHUNK
                expected_sample = word_idx % SAMPLES_PER_CHUNK
                
                status = "✓" if extracted_ch == expected_ch else "✗"
                print(f"      Word {word_idx:4d}: {status} ch={extracted_ch:2d} (exp {expected_ch:2d}), "
                      f"sample={expected_sample:3d}, code=0x{extracted_code:04X}")
                
                if extracted_ch != expected_ch:
                    print(f"        *** MISMATCH! ***")
    
    # Check specific channels
    print(f"\n[4] Checking specific channels (0, 11, 12, 20, 31)...")
    for ch in [0, 11, 12, 20, 31]:
        chunk_id = 0
        sample_in_chunk = 0
        sample_idx = chunk_id * SAMPLES_PER_CHUNK + sample_in_chunk
        idx = ch * samples_per_channel + sample_idx
        
        # Get word position in chunk
        word_idx = ch * SAMPLES_PER_CHUNK + sample_in_chunk
        
        chunk = chunks[chunk_id]
        if word_idx * 4 < len(chunk):
            word_bytes = chunk[word_idx*4:(word_idx+1)*4]
            word32 = int.from_bytes(word_bytes, byteorder='little')
            extracted_ch = (word32 >> 16) & 0x3F
            extracted_code = word32 & 0xFFFF
            expected_code = all_data_16[idx]
            
            status = "✓" if (extracted_ch == ch and extracted_code == expected_code) else "✗"
            print(f"    Ch{ch:2d}: {status} word_idx={word_idx:4d}, extracted_ch={extracted_ch:2d}, "
                  f"code=0x{extracted_code:04X} (exp 0x{expected_code:04X})")
            
            if extracted_ch != ch or extracted_code != expected_code:
                print(f"      *** MISMATCH! ***")
    
    print(f"\n[5] Checking data statistics per channel...")
    for ch in [0, 11, 12, 20, 31]:
        start_idx = ch * samples_per_channel
        end_idx = start_idx + samples_per_channel
        ch_data = all_data_16[start_idx:end_idx]
        print(f"    Ch{ch:2d}: min={ch_data.min()}, max={ch_data.max()}, "
              f"mean={ch_data.mean():.1f}, std={ch_data.std():.1f}")
    
    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70)

if __name__ == "__main__":
    check_channel_data_ordering()
