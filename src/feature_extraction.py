"""
Physics-informed feature extraction module for Rail Corrugation condition monitoring.
Extracts speed, time-domain, frequency-domain, spatial differential, and wavelength features.
"""
import numpy as np
from scipy import signal
from scipy import stats
from src.config import (
    SAMPLING_RATE, SPEED_SENSOR_TEETH, WHEEL_CIRCUMFERENCE,
    SIDE_I_POSITIONS, SIDE_II_POSITIONS, NUM_CARS, FREQ_BANDS, WAVELENGTH_BINS
)
from src.speed_estimator import estimate_speed_from_pulse

def compute_time_stats(x):
    """
    Computes time-domain statistical indicators for a 1D or 2D array (samples x channels).
    """
    mean = np.mean(x, axis=0)
    rms = np.sqrt(np.mean(x**2, axis=0))
    peak = np.max(np.abs(x), axis=0)
    p2p = np.ptp(x, axis=0)
    kurt = stats.kurtosis(x, axis=0, fisher=True)
    skew = stats.skew(x, axis=0)
    
    # Avoid zero division
    eps = 1e-9
    crest_factor = peak / (rms + eps)
    shape_factor = rms / (np.mean(np.abs(x), axis=0) + eps)
    impulse_factor = peak / (np.mean(np.abs(x), axis=0) + eps)
    margin_factor = peak / ((np.mean(np.sqrt(np.abs(x) + eps), axis=0))**2 + eps)
    
    return {
        "rms": float(np.mean(rms)),
        "peak": float(np.mean(peak)),
        "p2p": float(np.mean(p2p)),
        "kurt": float(np.mean(kurt)),
        "skew": float(np.mean(skew)),
        "crest": float(np.mean(crest_factor)),
        "shape": float(np.mean(shape_factor)),
        "impulse": float(np.mean(impulse_factor)),
        "margin": float(np.mean(margin_factor)),
        "max_rms": float(np.max(rms)),
        "max_peak": float(np.max(peak)),
        "max_kurt": float(np.max(kurt))
    }

def extract_features_from_df(df, filename=""):
    """
    Extracts complete physics-informed feature dictionary from a 1-second dataframe.
    """
    features = {}
    if filename:
        features["filename"] = filename
        
    # 1. Speed Features
    speed_col = df.iloc[:, 0].values
    speed_info = estimate_speed_from_pulse(speed_col, SAMPLING_RATE)
    for k, v in speed_info.items():
        features[k] = v
        
    v_mps = max(speed_info["speed_mps"], 1.0) # avoid div by zero
    
    # Identify column groups
    col_names = list(df.columns)
    
    # Map columns to cars, positions, and sensor types
    # Col index 1 to 128
    car_pos_map = {}
    for c_idx in range(1, len(col_names)):
        c_name = col_names[c_idx]
        is_vib = "Vibration" in c_name
        is_shock = "Shock" in c_name
        # extract car and pos
        # Name pattern: "Vibration of bearing in position {pos} of car {car}"
        # Extract numbers
        parts = c_name.split()
        pos = int(parts[5])
        car = int(parts[8])
        car_pos_map[c_idx] = (car, pos, is_vib, is_shock)
        
    # Group indices
    s1_vib_idx = [idx for idx, (car, pos, is_vib, is_shock) in car_pos_map.items() if is_vib and pos in SIDE_I_POSITIONS]
    s2_vib_idx = [idx for idx, (car, pos, is_vib, is_shock) in car_pos_map.items() if is_vib and pos in SIDE_II_POSITIONS]
    s1_shock_idx = [idx for idx, (car, pos, is_vib, is_shock) in car_pos_map.items() if is_shock and pos in SIDE_I_POSITIONS]
    s2_shock_idx = [idx for idx, (car, pos, is_vib, is_shock) in car_pos_map.items() if is_shock and pos in SIDE_II_POSITIONS]
    
    # Extract matrices: shape (10000, 32)
    s1_vib = df.iloc[:, s1_vib_idx].values
    s2_vib = df.iloc[:, s2_vib_idx].values
    s1_shock = df.iloc[:, s1_shock_idx].values
    s2_shock = df.iloc[:, s2_shock_idx].values
    
    # 2. Time-Domain Stats for Side I vs Side II
    s1_vib_stats = compute_time_stats(s1_vib)
    s2_vib_stats = compute_time_stats(s2_vib)
    s1_shock_stats = compute_time_stats(s1_shock)
    s2_shock_stats = compute_time_stats(s2_shock)
    
    for k, v in s1_vib_stats.items():
        features[f"s1_vib_{k}"] = v
    for k, v in s2_vib_stats.items():
        features[f"s2_vib_{k}"] = v
    for k, v in s1_shock_stats.items():
        features[f"s1_shock_{k}"] = v
    for k, v in s2_shock_stats.items():
        features[f"s2_shock_{k}"] = v
        
    # 3. Bilateral Differential (Side Contrast) Ratios
    eps = 1e-8
    features["diff_rms"] = s1_vib_stats["rms"] - s2_vib_stats["rms"]
    features["sdi_rms"] = (s1_vib_stats["rms"] - s2_vib_stats["rms"]) / (s1_vib_stats["rms"] + s2_vib_stats["rms"] + eps)
    features["ratio_rms"] = s1_vib_stats["rms"] / (s2_vib_stats["rms"] + eps)
    
    features["diff_max_rms"] = s1_vib_stats["max_rms"] - s2_vib_stats["max_rms"]
    features["sdi_max_rms"] = (s1_vib_stats["max_rms"] - s2_vib_stats["max_rms"]) / (s1_vib_stats["max_rms"] + s2_vib_stats["max_rms"] + eps)
    
    features["diff_peak"] = s1_vib_stats["peak"] - s2_vib_stats["peak"]
    features["sdi_peak"] = (s1_vib_stats["peak"] - s2_vib_stats["peak"]) / (s1_vib_stats["peak"] + s2_vib_stats["peak"] + eps)
    
    features["diff_kurt"] = s1_vib_stats["kurt"] - s2_vib_stats["kurt"]
    features["sdi_kurt"] = (s1_vib_stats["kurt"] - s2_vib_stats["kurt"]) / (abs(s1_vib_stats["kurt"]) + abs(s2_vib_stats["kurt"]) + eps)
    
    # Vibration to shock ratio
    features["vib_shock_ratio_s1"] = s1_vib_stats["rms"] / (s1_shock_stats["rms"] + eps)
    features["vib_shock_ratio_s2"] = s2_vib_stats["rms"] / (s2_shock_stats["rms"] + eps)
    features["vib_shock_ratio_diff"] = features["vib_shock_ratio_s1"] - features["vib_shock_ratio_s2"]

    # 4. Per-Car Spatial Contrast
    car_diffs = []
    car_sdis = []
    car_s1_anomalies = 0
    car_s2_anomalies = 0
    
    for car in range(1, NUM_CARS + 1):
        c_s1_vib_idx = [idx for idx, (c, p, is_vib, is_shock) in car_pos_map.items() if is_vib and c == car and p in SIDE_I_POSITIONS]
        c_s2_vib_idx = [idx for idx, (c, p, is_vib, is_shock) in car_pos_map.items() if is_vib and c == car and p in SIDE_II_POSITIONS]
        
        c_s1_mat = df.iloc[:, c_s1_vib_idx].values
        c_s2_mat = df.iloc[:, c_s2_vib_idx].values
        
        c_s1_rms = np.sqrt(np.mean(c_s1_mat**2))
        c_s2_rms = np.sqrt(np.mean(c_s2_mat**2))
        
        diff_c = c_s1_rms - c_s2_rms
        sdi_c = diff_c / (c_s1_rms + c_s2_rms + eps)
        
        car_diffs.append(diff_c)
        car_sdis.append(sdi_c)
        
        features[f"car{car}_s1_rms"] = float(c_s1_rms)
        features[f"car{car}_s2_rms"] = float(c_s2_rms)
        features[f"car{car}_sdi"] = float(sdi_c)
        
        if sdi_c > 0.15:
            car_s1_anomalies += 1
        elif sdi_c < -0.15:
            car_s2_anomalies += 1
            
    features["max_car_diff_s1"] = float(np.max(car_diffs))
    features["max_car_diff_s2"] = float(-np.min(car_diffs))
    features["max_car_sdi_s1"] = float(np.max(car_sdis))
    features["max_car_sdi_s2"] = float(-np.min(car_sdis))
    features["car_s1_anomaly_count"] = car_s1_anomalies
    features["car_s2_anomaly_count"] = car_s2_anomalies
    features["car_sdi_std"] = float(np.std(car_sdis))
    
    # 5. Spectral Band Powers & Spatial Wavelength Analysis
    # Welch PSD on average signal per side
    s1_vib_mean_signal = np.mean(s1_vib, axis=1) # (10000,)
    s2_vib_mean_signal = np.mean(s2_vib, axis=1)
    
    freqs, psd_s1 = signal.welch(s1_vib_mean_signal, fs=SAMPLING_RATE, nperseg=1024)
    freqs, psd_s2 = signal.welch(s2_vib_mean_signal, fs=SAMPLING_RATE, nperseg=1024)
    
    total_psd_s1 = np.sum(psd_s1) + eps
    total_psd_s2 = np.sum(psd_s2) + eps
    
    features["total_psd_s1"] = float(total_psd_s1)
    features["total_psd_s2"] = float(total_psd_s2)
    features["sdi_total_psd"] = float((total_psd_s1 - total_psd_s2) / (total_psd_s1 + total_psd_s2))
    
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        bp_s1 = np.sum(psd_s1[band_mask])
        bp_s2 = np.sum(psd_s2[band_mask])
        
        features[f"psd_s1_{band_name}"] = float(bp_s1)
        features[f"psd_s2_{band_name}"] = float(bp_s2)
        features[f"sdi_psd_{band_name}"] = float((bp_s1 - bp_s2) / (bp_s1 + bp_s2 + eps))
        features[f"ratio_psd_{band_name}"] = float(bp_s1 / (bp_s2 + eps))
        
    # 6. Spatial Wavelength Energy Bins: lambda = v / f
    # Convert frequency array to physical wavelength (m)
    wavelengths = np.zeros_like(freqs)
    non_zero = freqs > 0
    wavelengths[non_zero] = v_mps / freqs[non_zero]
    
    for wl_name, (wl_min, wl_max) in WAVELENGTH_BINS:
        wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if np.any(wl_mask):
            wl_energy_s1 = np.sum(psd_s1[wl_mask])
            wl_energy_s2 = np.sum(psd_s2[wl_mask])
        else:
            wl_energy_s1 = 0.0
            wl_energy_s2 = 0.0
            
        features[f"wl_s1_{wl_name}"] = float(wl_energy_s1)
        features[f"wl_s2_{wl_name}"] = float(wl_energy_s2)
        features[f"sdi_wl_{wl_name}"] = float((wl_energy_s1 - wl_energy_s2) / (wl_energy_s1 + wl_energy_s2 + eps))

    return features
