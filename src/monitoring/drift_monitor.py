import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.logger import setup_logger

logger = setup_logger("drift_monitor")

class DriftMonitor:
    """Monitors distribution drift for input features and tracks model performance over time."""
    
    def __init__(self, psi_threshold: float = 0.2):
        self.psi_threshold = psi_threshold

    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between baseline and current data distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant distribution shift (Drift Detected!)
        """
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]
        
        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(baseline, percentiles)
        buckets[0] = -np.inf
        buckets[-1] = np.inf
        
        # Remove duplicate bin boundaries
        buckets = np.unique(buckets)
        if len(buckets) < 2:
            return 0.0

        b_counts, _ = np.histogram(baseline, bins=buckets)
        c_counts, _ = np.histogram(current, bins=buckets)

        b_pct = b_counts / len(baseline)
        c_pct = c_counts / len(current)

        # Replace zeros with tiny constant to prevent log(0) or division by zero
        b_pct = np.where(b_pct == 0, 0.0001, b_pct)
        c_pct = np.where(c_pct == 0, 0.0001, c_pct)

        psi_val = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
        return round(float(psi_val), 4)

    def analyze_dataset_drift(self, df_baseline: pd.DataFrame, df_current: pd.DataFrame, num_features: List[str]) -> Dict[str, Any]:
        """Runs drift calculation across all numeric features."""
        logger.info("Running feature drift analysis against baseline dataset...")
        drift_results = {}
        drift_detected_count = 0

        for col in num_features:
            if col in df_baseline.columns and col in df_current.columns:
                base_vals = pd.to_numeric(df_baseline[col], errors="coerce").fillna(0).values
                curr_vals = pd.to_numeric(df_current[col], errors="coerce").fillna(0).values
                
                psi = self.calculate_psi(base_vals, curr_vals)
                is_drifted = psi >= self.psi_threshold
                if is_drifted:
                    drift_detected_count += 1
                    
                drift_results[col] = {
                    "psi_score": psi,
                    "drift_detected": is_drifted,
                    "baseline_mean": round(float(np.mean(base_vals)), 2),
                    "current_mean": round(float(np.mean(curr_vals)), 2)
                }

        summary = {
            "total_features_monitored": len(num_features),
            "drifted_features_count": drift_detected_count,
            "overall_drift_status": "ALERT_RETRAIN_RECOMMENDED" if drift_detected_count > 0 else "STABLE",
            "feature_details": drift_results
        }
        logger.info(f"Drift analysis complete. Overall Status: {summary['overall_drift_status']}")
        return summary
