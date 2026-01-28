#!/usr/bin/env python3
"""Diagnostic script to analyze why Channel 0 has false detections."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_neo(signal):
    """Compute NEO: psi[n] = x[n]^2 - x[n-1]*x[n+1]"""
    # Center around 0 (subtract 32768)
    x_centered = signal.astype(np.int32) - 32768
    
    # NEO computation
    neo = np.zeros(len(signal))
    for i in range(1, len(signal) - 1):
        curr_sq = x_centered[i] ** 2
        neigh_mul = x_centered[i-1] * x_centered[i+1]
        neo[i] = abs(curr_sq - neigh_mul)  # Absolute value
    
    return neo

def main():
    log_base = "test_output"
    inputs_dir = Path("inputs")
    raw_data_file = inputs_dir / f"{log_base}_raw_data.npy"
    
    if not raw_data_file.exists():
        print(f"File not found: {raw_data_file}")
        return
    
    raw_data = np.load(raw_data_file)
    samples_per_channel = len(raw_data) // 32
    
    # Extract Channel 0 and Channel 1 data
    ch0_start = 0 * samples_per_channel
    ch1_start = 1 * samples_per_channel
    
    ch0_data = raw_data[ch0_start:ch0_start + samples_per_channel]
    ch1_data = raw_data[ch1_start:ch1_start + samples_per_channel]
    
    # Compute NEO for both channels
    ch0_neo = compute_neo(ch0_data)
    ch1_neo = compute_neo(ch1_data)
    
    threshold = 120000
    
    # Find where NEO exceeds threshold
    ch0_detections = np.where(ch0_neo > threshold)[0]
    ch1_detections = np.where(ch1_neo > threshold)[0]
    
    print(f"Channel 0: {len(ch0_detections)} samples with NEO > {threshold}")
    print(f"Channel 1: {len(ch1_detections)} samples with NEO > {threshold}")
    
    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Channel 0 data
    time_ms = np.arange(len(ch0_data))
    axes[0].plot(time_ms[:5000], ch0_data[:5000], 'b-', linewidth=0.5, alpha=0.7, label='Channel 0 ADC')
    axes[0].axhline(y=32768, color='g', linestyle='--', alpha=0.5, label='Midpoint (32768)')
    axes[0].set_title('Channel 0 - Raw Data (first 5 seconds)')
    axes[0].set_ylabel('ADC Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Channel 0 NEO
    axes[1].plot(time_ms[:5000], ch0_neo[:5000], 'r-', linewidth=0.5, alpha=0.7, label='Channel 0 NEO')
    axes[1].axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
    axes[1].scatter(ch0_detections[ch0_detections < 5000], 
                   ch0_neo[ch0_detections[ch0_detections < 5000]], 
                   color='red', s=10, label='Detections')
    axes[1].set_title(f'Channel 0 - NEO (first 5 seconds, {len(ch0_detections[ch0_detections < 5000])} detections)')
    axes[1].set_ylabel('NEO Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Channel 1 data
    axes[2].plot(time_ms[:5000], ch1_data[:5000], 'b-', linewidth=0.5, alpha=0.7, label='Channel 1 ADC')
    axes[2].axhline(y=32768, color='g', linestyle='--', alpha=0.5, label='Midpoint (32768)')
    axes[2].set_title('Channel 1 - Raw Data (first 5 seconds)')
    axes[2].set_ylabel('ADC Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Channel 1 NEO
    axes[3].plot(time_ms[:5000], ch1_neo[:5000], 'r-', linewidth=0.5, alpha=0.7, label='Channel 1 NEO')
    axes[3].axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
    axes[3].scatter(ch1_detections[ch1_detections < 5000], 
                   ch1_neo[ch1_detections[ch1_detections < 5000]], 
                   color='red', s=10, label='Detections')
    axes[3].set_title(f'Channel 1 - NEO (first 5 seconds, {len(ch1_detections[ch1_detections < 5000])} detections)')
    axes[3].set_ylabel('NEO Value')
    axes[3].set_xlabel('Time (ms)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path("seizures") / f"{log_base}_diagnostic.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic plot saved to {output_file}")
    print(f"\nChannel 0 statistics:")
    print(f"  Max NEO: {np.max(ch0_neo):.0f}")
    print(f"  Mean NEO: {np.mean(ch0_neo):.0f}")
    print(f"  Std NEO: {np.std(ch0_neo):.0f}")
    print(f"  Samples above threshold: {len(ch0_detections)} ({100*len(ch0_detections)/len(ch0_data):.2f}%)")
    if len(ch0_detections) > 0:
        print(f"  First detection at: {ch0_detections[0]} ms")
        print(f"  Last detection at: {ch0_detections[-1]} ms")
        # Check for gaps
        gaps = np.diff(ch0_detections)
        print(f"  Average gap between detections: {np.mean(gaps):.1f} ms")
        print(f"  Max gap: {np.max(gaps)} ms")
    
    print(f"\nChannel 1 statistics:")
    print(f"  Max NEO: {np.max(ch1_neo):.0f}")
    print(f"  Mean NEO: {np.mean(ch1_neo):.0f}")
    print(f"  Std NEO: {np.std(ch1_neo):.0f}")
    print(f"  Samples above threshold: {len(ch1_detections)} ({100*len(ch1_detections)/len(ch1_data):.2f}%)")
    if len(ch1_detections) > 0:
        print(f"  First detection at: {ch1_detections[0]} ms")
        print(f"  Last detection at: {ch1_detections[-1]} ms")
        # Check for gaps
        gaps = np.diff(ch1_detections)
        print(f"  Average gap between detections: {np.mean(gaps):.1f} ms")
        print(f"  Max gap: {np.max(gaps)} ms")

if __name__ == "__main__":
    main()
