"""
Speed estimation module for train axle speed sensor (toothed wheel).
Toothed wheel: 90 teeth, Wheel diameter: 0.85 m.
"""
import numpy as np
from src.config import SPEED_SENSOR_TEETH, WHEEL_CIRCUMFERENCE

def estimate_speed_from_pulse(pulse_signal, sampling_rate=10000):
    """
    Computes train speed from binary/toothed-wheel pulse signal.
    
    Parameters:
        pulse_signal: 1D array of 0/1 values (or noisy analog pulses)
        sampling_rate: sampling frequency in Hz (default 10,000)
        
    Returns:
        dict with:
            speed_mps: speed in meters per second
            speed_kmh: speed in kilometers per hour
            rot_freq_hz: wheel rotational frequency in Hz
            pulse_count: number of rising transitions
            speed_stability: std dev of speed across 4 sub-windows
    """
    # Binarize if not strictly binary
    if pulse_signal.dtype != np.bool_ and pulse_signal.dtype != np.int_:
        threshold = (np.max(pulse_signal) + np.min(pulse_signal)) / 2.0
        binary_pulse = (pulse_signal > threshold).astype(np.int8)
    else:
        binary_pulse = (pulse_signal > 0).astype(np.int8)
        
    # Count rising edges (0 -> 1)
    diff = np.diff(binary_pulse)
    rising_edges = np.sum(diff > 0)
    
    # Duration in seconds
    duration = len(pulse_signal) / sampling_rate
    
    # Calculate rotational frequency (rev/s)
    rot_freq = rising_edges / (SPEED_SENSOR_TEETH * duration)
    
    # Linear speed (m/s)
    speed_mps = rot_freq * WHEEL_CIRCUMFERENCE
    speed_kmh = speed_mps * 3.6
    
    # Calculate stability across 4 quarters
    quarter_len = len(pulse_signal) // 4
    sub_speeds = []
    for q in range(4):
        sub_diff = diff[q * quarter_len : (q + 1) * quarter_len]
        sub_edges = np.sum(sub_diff > 0)
        sub_duration = quarter_len / sampling_rate
        sub_v = (sub_edges / (SPEED_SENSOR_TEETH * sub_duration)) * WHEEL_CIRCUMFERENCE
        sub_speeds.append(sub_v)
        
    speed_stability = float(np.std(sub_speeds))
    acceleration = float(sub_speeds[-1] - sub_speeds[0]) # m/s change over 1s
    
    return {
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_kmh),
        "rot_freq_hz": float(rot_freq),
        "pulse_count": int(rising_edges),
        "speed_stability": speed_stability,
        "acceleration": acceleration
    }

def freq_to_wavelength(freq_hz, speed_mps):
    """
    Converts temporal frequency (Hz) to spatial corrugation wavelength (meters):
    lambda = v / f
    """
    if freq_hz <= 0 or speed_mps <= 0:
        return 0.0
    return speed_mps / freq_hz

def wavelength_to_freq(wavelength_m, speed_mps):
    """
    Converts spatial corrugation wavelength (meters) to temporal frequency (Hz):
    f = v / lambda
    """
    if wavelength_m <= 0:
        return 0.0
    return speed_mps / wavelength_m
