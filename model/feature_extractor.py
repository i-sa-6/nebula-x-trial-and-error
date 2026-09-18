"""Feature extraction for Rail Vehicle Door Cycle Telemetry.

Extracts physics-informed electrical, kinematic, and statistical features
from door-opening and door-closing cycles. Every feature is computed
strictly within the isolated segment window [t_start, t_end] to ensure
zero temporal data leakage.
"""

import math
from typing import Any, Dict, List


def extract_cycle_features(seg_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract physics-based features from a single door movement cycle.

    All calculations are strictly isolated to the provided cycle rows.
    No future data, past cycles, or global statistics are accessed.

    Args:
        seg_rows: List of row dictionaries representing a single door cycle.

    Returns:
        Dict mapping feature names to numeric float values.
    """
    n_rows = len(seg_rows)
    if n_rows == 0:
        raise ValueError("Cannot extract features from an empty segment.")

    dt = 0.02  # 50 Hz sampling rate (20 ms interval)
    duration_sec = n_rows * dt

    # 1. Parse raw signals
    currents = [float(r.get("Motor current(mA)", 0.0)) for r in seg_rows]
    voltages = [float(r.get("Motor Voltage(10mV)", 0.0)) for r in seg_rows]
    emfs = [float(r.get("Motor electrodynamic force", 0.0)) for r in seg_rows]
    positions = [float(r.get("Door leaf position", 0.0)) for r in seg_rows]

    # Operation identification
    start_pos = positions[0]
    end_pos = positions[-1]
    pos_delta = end_pos - start_pos
    opening_flags = sum(1 for r in seg_rows if r.get("Door is opening") == "1")
    closing_flags = sum(1 for r in seg_rows if r.get("Door is closing") == "1")
    is_open_op = 1.0 if (opening_flags > closing_flags or end_pos > start_pos) else 0.0

    # 2. Mid-stroke definition (25% to 75% of movement duration)
    # This represents the constant velocity regime where motor torque primarily
    # balances mechanical sliding friction (acceleration torque ≈ 0).
    q1 = int(n_rows * 0.25)
    q3 = int(n_rows * 0.75)
    if q3 <= q1:
        q1, q3 = 0, n_rows

    mid_currents = currents[q1:q3]
    mid_voltages = voltages[q1:q3]
    mid_emfs = emfs[q1:q3]

    # 3. Current Statistics
    cur_mean = sum(currents) / n_rows
    cur_max = max(currents)
    cur_min = min(currents)
    cur_var = sum((x - cur_mean) ** 2 for x in currents) / n_rows
    cur_std = math.sqrt(cur_var)
    cur_rms = math.sqrt(sum(x ** 2 for x in currents) / n_rows)

    sorted_currents = sorted(currents)
    cur_median = sorted_currents[n_rows // 2]
    cur_p25 = sorted_currents[int(n_rows * 0.25)]
    cur_p75 = sorted_currents[int(n_rows * 0.75)]
    cur_iqr = cur_p75 - cur_p25

    # Mid-stroke current metrics (vital for abnormal friction detection)
    mid_cur_mean = sum(mid_currents) / len(mid_currents)
    mid_cur_max = max(mid_currents)
    mid_cur_var = sum((x - mid_cur_mean) ** 2 for x in mid_currents) / len(mid_currents)
    mid_cur_std = math.sqrt(mid_cur_var)
    cur_integral = sum(currents) * dt  # Electric charge Q (mA * s)

    # 4. Voltage & Back-EMF Statistics
    volt_mean = sum(voltages) / n_rows
    volt_mid_mean = sum(mid_voltages) / len(mid_voltages)
    volt_max = max(voltages)

    emf_mean = sum(emfs) / n_rows
    emf_mid_mean = sum(mid_emfs) / len(mid_emfs)
    emf_max = max(emfs)
    emf_to_current_ratio = emf_mid_mean / (mid_cur_mean + 1e-4)

    # 5. Electrical Power & Energy Consumption
    # Power = Voltage * Current (proportional to instantaneous electrical power)
    powers = [v * c for v, c in zip(voltages, currents)]
    mid_powers = powers[q1:q3]
    power_mean = sum(powers) / n_rows
    power_mid_mean = sum(mid_powers) / len(mid_powers)
    power_max = max(powers)
    total_energy = sum(powers) * dt  # Work performed (proportional to Joules)

    # 6. Kinematic & Geometric features
    mean_velocity = abs(pos_delta) / (duration_sec + 1e-4)

    return {
        "is_open_op": is_open_op,
        "n_rows": float(n_rows),
        "duration_sec": duration_sec,
        "cur_mean": cur_mean,
        "cur_std": cur_std,
        "cur_max": cur_max,
        "cur_min": cur_min,
        "cur_rms": cur_rms,
        "cur_median": cur_median,
        "cur_iqr": cur_iqr,
        "mid_cur_mean": mid_cur_mean,
        "mid_cur_max": mid_cur_max,
        "mid_cur_std": mid_cur_std,
        "cur_integral": cur_integral,
        "volt_mean": volt_mean,
        "volt_mid_mean": volt_mid_mean,
        "volt_max": volt_max,
        "emf_mean": emf_mean,
        "emf_mid_mean": emf_mid_mean,
        "emf_max": emf_max,
        "emf_to_current_ratio": emf_to_current_ratio,
        "power_mean": power_mean,
        "power_mid_mean": power_mid_mean,
        "power_max": power_max,
        "total_energy": total_energy,
        "mean_velocity": mean_velocity,
        "start_pos": start_pos,
        "end_pos": end_pos,
    }

