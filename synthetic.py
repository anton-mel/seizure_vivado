import math
import random
import time

NUM_CHANNELS = 32
SAMPLES_PER_CHANNEL = 128
TIMESTAMP_BYTES = 8
DATA_BYTES = NUM_CHANNELS * SAMPLES_PER_CHANNEL

# Seizure averaging
AVERAGE_WINDOW_SIZE = 8
AVERAGE_STEP_SIZE = 3

# Seizure visualization
# Parameters to find left and right boundaries
WINDOW_TIMEOUT = 200
TRANSITION_COUNT = 10

class NeuralSynthSource:
    
    def __init__(self, sample_rate, n_units=2, seed=None, enable_seizures=True):
        self.sample_rate = sample_rate
        self.n_units = n_units
        self.t_step_ms = 1000.0 / sample_rate
        self.t_ms = 0.0
        self.enable_seizures = enable_seizures
        
        # Constants from Intan SDK
        self.noise_rms_level_uv = 5.0
        self.spike_refractory_period_ms = 5.0
        self.lfp_frequency_hz = 2.3
        self.lfp_modulation_hz = 0.5
        
        # Seizure parameters (added by me)
        self.seizure_start_time_ms = None
        self.seizure_duration_ms = 6000.0  # 6 seconds
        self.seizure_freq_hz = 2.5  # 2.5 Hz low-frequency seizure activity
        self.seizure_amplitude_uv = 500.0  # Very high amplitude seizure 
        self.seizure_probability = 0.01  # 1% chance per second (lower chance = more dramatic)
        
        # Initialize random state
        if seed is not None:
            random.seed(seed)
        
        # Initialize spike parameters for each unit (added by me)
        self.spike_amplitude = []
        self.spike_duration_ms = []
        self.spike_rate_hz = []
        self.firing = []
        self.spike_time_ms = []
        
        for i in range(n_units):
            # Spike amplitude: -200 to -500 µV (negative going spikes)
            self.spike_amplitude.append(random.uniform(-500.0, -200.0))
            # Spike duration: 0.3 to 1.7 ms
            self.spike_duration_ms.append(random.uniform(0.3, 1.7))
            # Spike rate: 0.1 to 50 Hz (log-uniform distribution)
            self.spike_rate_hz.append(self._log_uniform(0.1, 50.0))
            # Status tracking
            self.firing.append(False)
            self.spike_time_ms.append(0.0)
    
    def _log_uniform(self, min_val, max_val):
        """Generate log-uniform random value"""
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        return math.exp(random.uniform(log_min, log_max))
    
    def _gaussian_noise(self):
        """Generate Gaussian noise using Central Limit Theorem"""
        gaussian_n = 6
        r = 0.0
        for _ in range(gaussian_n):
            r += random.uniform(-1.0, 1.0)
        # Scale factor: sqrt(3.0) / sqrt(n)
        gaussian_scale_factor = math.sqrt(3.0) / math.sqrt(gaussian_n)
        r *= gaussian_scale_factor
        return r
    
    def _lfp_voltage(self):
        """Generate LFP (Local Field Potential) voltage"""
        # Modulated amplitude: 100-180 µV
        amplitude = 100.0 + 80.0 * math.sin(2.0 * math.pi * (self.t_ms / 1000.0) * self.lfp_modulation_hz)
        # LFP frequency: 2.3 Hz
        return amplitude * math.sin(2.0 * math.pi * (self.t_ms / 1000.0) * self.lfp_frequency_hz)
    
    def _next_spike_voltage(self, unit):
        """Generate spike voltage for a specific unit"""
        if self.spike_time_ms[unit] < self.spike_duration_ms[unit]:
            # Exponentially decaying sine wave
            amplitude = self.spike_amplitude[unit] * math.exp(-2.0 * self.spike_time_ms[unit])
            phase = 2.0 * math.pi * self.spike_time_ms[unit] / self.spike_duration_ms[unit]
            result = amplitude * math.sin(phase)
            self.spike_time_ms[unit] += self.t_step_ms
            return result
        elif self.spike_time_ms[unit] < self.spike_duration_ms[unit] + self.spike_refractory_period_ms:
            # Refractory period
            self.spike_time_ms[unit] += self.t_step_ms
            return 0.0
        else:
            # End of refractory period, unit can fire again
            self.firing[unit] = False
            self.spike_time_ms[unit] = 0.0
            return 0.0
    
    def _seizure_voltage(self):
        """Generate seizure voltage - high amplitude, low frequency bursts"""
        if self.seizure_start_time_ms is None:
            # Random chance to start a seizure
            if random.random() < self.seizure_probability * self.t_step_ms / 1000.0:
                self.seizure_start_time_ms = self.t_ms
            return 0.0
        
        # Check if seizure is ongoing
        elapsed = self.t_ms - self.seizure_start_time_ms
        if elapsed < self.seizure_duration_ms:
            # Active seizure: low frequency (2.5 Hz) high amplitude bursts
            time_in_seizure = elapsed / 1000.0  # Convert to seconds
            seizure_signal = self.seizure_amplitude_uv * math.sin(2.0 * math.pi * self.seizure_freq_hz * time_in_seizure)
            return seizure_signal
        else:
            # Seizure ended
            self.seizure_start_time_ms = None
            return 0.0
    
    def next_sample(self):
        """Generate next sample in microvolts"""
        # Start with Gaussian noise
        result = self.noise_rms_level_uv * self._gaussian_noise()
        
        # Add spike voltages for firing units
        for unit in range(self.n_units):
            if self.firing[unit]:
                result += self._next_spike_voltage(unit)
            else:
                # Probability of starting a spike
                spike_modulation_factor = (1000.0 - (self.t_ms % 1000.0)) / 1000.0
                probability = spike_modulation_factor * self.spike_rate_hz[unit] * self.t_step_ms / 1000.0
                
                if random.random() < probability:
                    self.firing[unit] = True
        
        # Add LFP modulation
        result += self._lfp_voltage()
        
        # Add seizure activity if enabled
        if self.enable_seizures:
            result += self._seizure_voltage()
        
        # Advance time
        self.t_ms += self.t_step_ms
        
        return result
    
    def reset(self):
        """Reset to initial state"""
        self.t_ms = 0.0
        self.seizure_start_time_ms = None
        for unit in range(self.n_units):
            self.firing[unit] = False
            self.spike_time_ms[unit] = 0.0

def generate_data(num_channels=32, samples_per_channel=5000, sample_rate=1000.0, enable_seizures=True):
    """Generate realistic synthetic neural data quantized to 8-bit (0–255).

    NOTE: This function is kept for backward compatibility with older tests.
    New FPGA/Verilog tests use 16-bit Intan-style ADC codes generated by
    generate_data_intan16() below.
    """
    data_bytes = num_channels * samples_per_channel
    packet = bytearray(data_bytes)
    
    sources = []
    for channel in range(num_channels):
        # Use channel number + current time for truly random but reproducible per channel
        random_seed = channel + int(time.time() * 1000) if enable_seizures else channel
        sources.append(NeuralSynthSource(sample_rate, n_units=2, seed=random_seed, enable_seizures=enable_seizures))
    
    for channel in range(num_channels):
        for sample in range(samples_per_channel):
            voltage_uv = sources[channel].next_sample()
            
            # Legacy 8-bit quantization: map from approximately
            # -6389.76 µV to +6389.57 µV into 0–255.
            quantized = int((voltage_uv + 6389.76) * 255.0 / 12779.33)
            quantized = max(0, min(255, quantized))
            
            packet[channel * samples_per_channel + sample] = quantized
    
    # 32 channels × 128 samples × 1 byte 
    # = 4096 bytes = 4 KB
    return packet


def generate_data_intan16(num_channels=32, samples_per_channel=5000, sample_rate=1000.0, enable_seizures=True):
    """Generate synthetic neural data directly as 16-bit Intan-style ADC codes.

    Mapping:
        code16 = round(voltage_uv / 0.195) + 32768
        clipped to [0, 65535]
    """
    import numpy as np

    total_samples = num_channels * samples_per_channel
    data16 = np.zeros(total_samples, dtype=np.uint16)
    
    sources = []
    for channel in range(num_channels):
        random_seed = channel + int(time.time() * 1000) if enable_seizures else channel
        sources.append(NeuralSynthSource(sample_rate, n_units=2, seed=random_seed, enable_seizures=enable_seizures))
    
    for channel in range(num_channels):
        for sample in range(samples_per_channel):
            voltage_uv = sources[channel].next_sample()
            
            # Convert microvolts to 16-bit ADC code (Intan style)
            code16 = int(round(voltage_uv / 0.195)) + 32768
            if code16 < 0:
                code16 = 0
            elif code16 > 65535:
                code16 = 65535
            
            idx = channel * samples_per_channel + sample
            data16[idx] = code16
    
    return data16

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def visualize(data, samples, duration_label, filename):
        _, axes = plt.subplots(4, 1, figsize=(15, 10))
        for channel in range(4):
            start_idx = channel * samples
            channel_data = data[start_idx:start_idx + samples]
            time_ms = [(i * 1000.0 / 1000.0) for i in range(samples)]
            
            axes[channel].plot(time_ms, channel_data, linewidth=0.5, alpha=0.7)
            axes[channel].set_title(f'Channel {channel + 1} - {duration_label}')
            axes[channel].set_xlabel('Time (ms)')
            axes[channel].set_ylabel('ADC Value (0-255)')
            axes[channel].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
    
    # Run for 0.1 second without seizures
    samples_01sec = 100  # 1kHz aka 100 samples
    data_normal_01sec = generate_data(num_channels=32, samples_per_channel=samples_01sec, 
                                                       sample_rate=1000.0, enable_seizures=False)
    visualize(data_normal_01sec, samples_01sec, "Normal Neural Data (0.1 second at 1kHz)", "synthetic_normal_0.1sec.png")

    # Run for 1 minute without seizures
    samples_1min = 60000 # 1kHz aka 60000 samples
    data_normal = generate_data(num_channels=32, samples_per_channel=samples_1min, 
                                                 sample_rate=1000.0, enable_seizures=False)
    visualize(data_normal, samples_1min, "Normal Neural Data (1 minute at 1kHz)", "synthetic_normal_1min.png")
    
    # Run for 1 minute with seizures
    data_seizure = generate_data(num_channels=32, samples_per_channel=samples_1min, 
                                                  sample_rate=1000.0, enable_seizures=True)
    visualize(data_seizure, samples_1min, "Neural Data with Seizures (1 minute at 1kHz)", "synthetic_seizure_1min.png")
    
    print("\nDone!")
