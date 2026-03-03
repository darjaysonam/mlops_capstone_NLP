"""
Simple data drift detection using statistical comparison
"""

from scipy.stats import ks_2samp


def detect_drift(reference_data, new_data, threshold=0.05):
    """
    Uses Kolmogorov-Smirnov test
    """

    stat, p_value = ks_2samp(reference_data, new_data)

    drift_detected = p_value < threshold

    return {"p_value": p_value, "drift_detected": drift_detected}
