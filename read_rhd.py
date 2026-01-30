#!/usr/bin/env python3
"""
Read Intan RHD2000 .rhd files.
Uses Intan's official importrhdutilities.py to read RHD files.
"""

import sys
import numpy as np
from pathlib import Path

# Import the official Intan RHD loader from root directory
_rhd_loader_path = Path(__file__).parent / "importrhdutilities.py"
if not _rhd_loader_path.exists():
    raise FileNotFoundError(f"Intan RHD loader not found at {_rhd_loader_path}")

# Add the directory to path and import
sys.path.insert(0, str(_rhd_loader_path.parent))
try:
    import importrhdutilities as rhd_loader
except ImportError as e:
    raise ImportError(f"Failed to import Intan RHD loader: {e}")

def read_rhd_file(rhd_path):
    """
    Read Intan RHD2000 .rhd file and extract amplifier data.
    
    Uses the official Intan importrhdutilities.py loader.
    
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
    
    # Use Intan's official loader
    try:
        result, data_present = rhd_loader.load_file(str(rhd_path))
        
        if not data_present:
            raise ValueError("RHD file contains no amplifier data")
        
        # Extract amplifier data from result dict
        # The data is already in volts, but we need ADC codes
        # According to importrhdutilities.py line 1074-1075:
        # voltage = 0.195 * (adc_code - 32768)
        # So: adc_code = (voltage / 0.195) + 32768
        
        amplifier_data_volts = result['amplifier_data']  # Shape: (num_channels, num_samples) in Volts
        
        # Convert back to ADC codes (uint16)
        amplifier_data_adc = np.round((amplifier_data_volts / 0.195) + 32768).astype(np.uint16)
        
        # Get sample rate
        sample_rate = result.get('sample_rate', 20000.0)
        
        return {
            'amplifier_data': amplifier_data_adc,
            'sample_rate': sample_rate,
            'num_channels': amplifier_data_adc.shape[0],
            'num_samples': amplifier_data_adc.shape[1]
        }
            
    except Exception as e:
        raise RuntimeError(f"Failed to read RHD file: {e}")
