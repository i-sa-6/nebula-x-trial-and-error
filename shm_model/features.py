import numpy as np
import rainflow


def extract_features(df):
    stress = df["stress"]

    features = {
        "mean": stress.mean(),
        "std": stress.std(),
        "min": stress.min(),
        "max": stress.max(),
        "range": stress.max() - stress.min(),
        "rms": np.sqrt(np.mean(stress**2)),
        "median": stress.median(),
        "p95": stress.quantile(0.95),
        "p99": stress.quantile(0.99),
    }

    cycles = list(rainflow.count_cycles(stress.to_numpy()))

    if len(cycles) > 0:
        cycle_ranges = np.array([cycle[0] for cycle in cycles])
        cycle_counts = np.array([cycle[1] for cycle in cycles])

        features["rf_total_cycles"] = np.sum(cycle_counts)
        features["rf_mean_range"] = np.average(
            cycle_ranges,
            weights=cycle_counts
        )
        features["rf_max_range"] = np.max(cycle_ranges)
        features["rf_p95_range"] = np.percentile(cycle_ranges, 95)

        features["rf_sum_range2"] = np.sum(
            cycle_counts * cycle_ranges**2
        )
        features["rf_sum_range3"] = np.sum(
            cycle_counts * cycle_ranges**3
        )
        features["rf_sum_range4"] = np.sum(
            cycle_counts * cycle_ranges**4
        )

    else:
        features["rf_total_cycles"] = 0
        features["rf_mean_range"] = 0
        features["rf_max_range"] = 0
        features["rf_p95_range"] = 0
        features["rf_sum_range2"] = 0
        features["rf_sum_range3"] = 0
        features["rf_sum_range4"] = 0

    return features