#!/usr/bin/env python3

""" Generate synthetic data for Verilog datapath testing """

from synthetic import generate_data_intan16, NUM_CHANNELS
import numpy as np
import os

# helper function to save 1 4KB chunk to a SystemVerilog task file
def save_chunk_to_file(chunk_id, chunk_data, output_dir='fpga/fpga-tests'):
    """ Save a 4KB chunk to a SystemVerilog task file """
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = NUM_CHANNELS * 128
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size
    
    # save to file
    filepath = os.path.join(output_dir, f'test_data_{chunk_id:03d}.sv')
    with open(filepath, 'w') as f:
        f.write(f"// Auto-generated test data chunk {chunk_id} from synthetic.py (1 minute dataset)\n")
        f.write(f"// Covers samples {start_idx} to {end_idx} (4KB chunk)\n")
        f.write("task load_test_data;\n")
        f.write("begin\n")
        for i, byte_value in enumerate(chunk_data):
            f.write(f"    input_data[{i}] = 8'd{byte_value};\n")
        f.write("end\n")
        f.write("endtask\n")
    
    np.frombuffer(chunk_data, dtype=np.uint8)

# generate exactly 450 chunks of 4KB data (1,843,200 bytes)
def generate_full_dataset_for_verilog():
    SAMPLES_PER_CHANNEL_1MIN = 60000
    NUM_CHUNKS = 450
    SAMPLES_PER_CHUNK = 128  # per channel
    
    # Generate data for all 32 channels with 60k samples each
    # This creates a flat array of 16-bit Intan-style ADC codes:
    # [ch0_0, ch0_1, ..., ch0_59999, ch1_0, ch1_1, ..., ch1_59999, ...]
    all_data_16 = generate_data_intan16(
        NUM_CHANNELS,
        SAMPLES_PER_CHANNEL_1MIN,
        sample_rate=1000.0,
        enable_seizures=True,
    )
    
    # Now reorganize into chunks with interleaved format
    # Each chunk has: [ch0_128i+0, ..., ch0_128i+127, ch1_128i+0, ..., ch31_128i+127]
    output_file = 'fpga/build/full_dataset.hex'
    os.makedirs('fpga/build', exist_ok=True)
    
    with open(output_file, 'w') as f:
        for chunk_id in range(NUM_CHUNKS):
            # For each channel, extract 128 samples starting at chunk_id * 128
            for ch in range(NUM_CHANNELS):
                for sample_in_chunk in range(SAMPLES_PER_CHUNK):
                    sample_idx = chunk_id * SAMPLES_PER_CHUNK + sample_in_chunk
                    idx = ch * SAMPLES_PER_CHANNEL_1MIN + sample_idx
                    code16 = int(all_data_16[idx])
                    # One 16-bit word per line (hex, 4 digits)
                    f.write(f"{code16:04x}\n")
    
    # 32 channels × 128 samples = 4096 16-bit words per chunk (8192 bytes)
    BYTES_PER_CHUNK = 4096 * 2
    print(f"Generated {NUM_CHUNKS} chunks ({NUM_CHUNKS * BYTES_PER_CHUNK} bytes)")
    return all_data_16, output_file

def plot_generated_data(data, filename, title="Verilog Synthetic Data", chunk_count=450):
    import matplotlib.pyplot as plt
    
    # Only plot the data that matches what's in full_dataset.hex
    # 450 chunks × 4096 bytes = 1,843,200 bytes
    # Per channel: 450 chunks × 128 samples = 57,600 samples (not full 60k)
    max_samples = chunk_count * 128  # samples per channel in the saved hex file
    
    samples_per_channel = NUM_CHANNELS * max_samples
    
    _, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    for channel in range(4):
        start_idx = channel * max_samples  # This is how synthetic.py organizes it: channel * samples_per_channel
        end_idx = start_idx + max_samples
        channel_data = data[start_idx:end_idx][:max_samples]
        
        # Time in milliseconds (1 sample = 1ms at 1kHz)
        time_ms = [i for i in range(len(channel_data))]
        
        # Plot waveform
        axes[channel].plot(time_ms, channel_data, linewidth=0.5, alpha=0.7, color='blue')
        axes[channel].set_title(f'Channel {channel + 1} - {title}')
        axes[channel].set_xlabel('Time (ms)')
        axes[channel].set_ylabel('ADC Value (0-255)')
        axes[channel].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    os.makedirs('fpga', exist_ok=True)
    data, hex_file = generate_full_dataset_for_verilog()
    
    plot_generated_data(data, 'fpga/graphs/generated_verilog_data.png', 
                        title='Generated Synthetic Data for Verilog')
