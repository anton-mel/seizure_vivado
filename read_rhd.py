#!/usr/bin/env python3
"""
Read Intan RHD2000 .rhd files.
Uses Intan's official read_rhd library if available, otherwise implements basic parser.
"""

import sys
import struct
import numpy as np
from pathlib import Path

# Try to import Intan's official read_rhd library
try:
    # Check if IntanRHX app has the library
    sys.path.insert(0, '/Applications/IntanRHX.app/Contents/Resources')
    import read_rhd as intan_read_rhd
    HAS_INTAN_LIB = True
except ImportError:
    HAS_INTAN_LIB = False

def read_rhd_file(rhd_path):
    """
    Read Intan RHD2000 .rhd file and extract amplifier data.
    
    Returns:
        dict with keys:
            - 'amplifier_data': numpy array of shape (num_channels, num_samples) with uint16 ADC codes
            - 'sample_rate': float, samples per second
            - 'num_channels': int
            - 'num_samples': int
    """
    rhd_path = Path(rhd_path)
    if not rhd_path.exists():
        raise FileNotFoundError(f"RHD file not found: {rhd_path}")
    
    if HAS_INTAN_LIB and intan_read_rhd:
        # Use Intan's official library
        try:
            # Try different possible function names
            if hasattr(intan_read_rhd, 'read_file'):
                result = intan_read_rhd.read_file(str(rhd_path))
            elif hasattr(intan_read_rhd, 'read_rhd_file'):
                result = intan_read_rhd.read_rhd_file(str(rhd_path))
            else:
                raise AttributeError("No read function found in Intan library")
            
            amplifier_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
            sample_rate = result.get('frequency_parameters', {}).get('amplifier_sample_rate', 20000.0)
            
            return {
                'amplifier_data': amplifier_data,
                'sample_rate': sample_rate,
                'num_channels': amplifier_data.shape[0],
                'num_samples': amplifier_data.shape[1]
            }
        except Exception as e:
            print(f"Warning: Intan library failed: {e}. Trying manual parsing...")
    
    # Manual parsing fallback
    return read_rhd_manual(rhd_path)

def read_rhd_manual(rhd_path):
    """
    Manual RHD file parser (fallback if official library not available).
    This is a simplified parser - may not work for all RHD file versions.
    """
    with open(rhd_path, 'rb') as f:
        # Read magic number
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x0D691F3A:
            raise ValueError(f"Invalid RHD file format. Magic: 0x{magic:08X}")
        
        # Read version
        version = struct.unpack('<H', f.read(2))[0]
        
        # Skip to data section (simplified - actual format is more complex)
        # For now, raise error suggesting to use IntanRHX export
        raise NotImplementedError(
            "Manual RHD parsing not fully implemented. "
            "Please use IntanRHX software to export data to a compatible format, "
            "or ensure Intan's read_rhd library is available."
        )
