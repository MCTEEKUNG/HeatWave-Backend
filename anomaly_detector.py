"""
Heatwave AI — Anomaly Detector
Triggers heavy computation only when anomalies are detected.
Normal mode: use cached AI prediction (fast, cheap).
Anomaly mode: recompute with full latest data (accurate).
"""
import numpy as np
import pandas as pd
import config


class AnomalyDetector:
    """
    Detects weather anomalies that may indicate incoming heatwaves.
    Uses Z-score + rate-of-change based detection.
    
    Triggers:
    - Rapid temperature increase (>2 std dev above rolling mean)
    - NDVI sudden drop (>1.5 std dev below rolling mean)
    - Humidity rapid drop
    - Wind speed sudden increase
    """

    def __init__(self, z_threshold=2.0, rate_threshold=1.5, lookback=30):
        self.z_threshold = z_threshold
        self.rate_threshold = rate_threshold
        self.lookback = lookback

    def detect(self, df, verbose=True):
        """
        Check the latest data point for anomalies.
        Returns: dict with anomaly flags and details.
        """
        if len(df) < self.lookback + 1:
            return {"is_anomaly": False, "reason": "Insufficient data", "triggers": []}

        latest = df.iloc[-1]
        history = df.iloc[-(self.lookback + 1):-1]

        triggers = []

        # --- 1. Temperature Spike ---
        if "T2M_MAX" in df.columns:
            temp_mean = history["T2M_MAX"].mean()
            temp_std = history["T2M_MAX"].std() + 1e-8
            temp_z = (latest["T2M_MAX"] - temp_mean) / temp_std

            if temp_z > self.z_threshold:
                triggers.append({
                    "feature": "T2M_MAX",
                    "type": "spike",
                    "z_score": round(temp_z, 2),
                    "value": round(latest["T2M_MAX"], 2),
                    "mean": round(temp_mean, 2),
                    "detail": f"Temp {latest['T2M_MAX']:.1f}°C is {temp_z:.1f}σ above "
                              f"30-day mean ({temp_mean:.1f}°C)",
                })

        # --- 2. Temperature Rate of Change ---
        if "T2M_MAX" in df.columns and len(df) >= 4:
            temp_3d_ago = df.iloc[-4]["T2M_MAX"]
            temp_change = latest["T2M_MAX"] - temp_3d_ago
            temp_change_std = history["T2M_MAX"].diff().std() + 1e-8
            rate_z = temp_change / (temp_change_std * 3)

            if rate_z > self.rate_threshold:
                triggers.append({
                    "feature": "T2M_MAX",
                    "type": "rapid_increase",
                    "rate": round(temp_change, 2),
                    "detail": f"Temp rose {temp_change:.1f}°C in 3 days "
                              f"(rate z={rate_z:.1f}σ)",
                })

        # --- 3. NDVI Drop ---
        if "NDVI" in df.columns:
            ndvi_mean = history["NDVI"].mean()
            ndvi_std = history["NDVI"].std() + 1e-8
            ndvi_z = (ndvi_mean - latest["NDVI"]) / ndvi_std  # Inverted: drop is bad

            if ndvi_z > self.rate_threshold:
                triggers.append({
                    "feature": "NDVI",
                    "type": "drop",
                    "z_score": round(ndvi_z, 2),
                    "value": round(latest["NDVI"], 4),
                    "mean": round(ndvi_mean, 4),
                    "detail": f"NDVI {latest['NDVI']:.4f} is {ndvi_z:.1f}σ below "
                              f"30-day mean ({ndvi_mean:.4f})",
                })

        # --- 4. Humidity Drop ---
        if "RH2M" in df.columns:
            rh_mean = history["RH2M"].mean()
            rh_std = history["RH2M"].std() + 1e-8
            rh_z = (rh_mean - latest["RH2M"]) / rh_std

            if rh_z > self.z_threshold:
                triggers.append({
                    "feature": "RH2M",
                    "type": "drop",
                    "z_score": round(rh_z, 2),
                    "value": round(latest["RH2M"], 1),
                    "detail": f"Humidity {latest['RH2M']:.1f}% dropped "
                              f"{rh_z:.1f}σ below 30-day mean ({rh_mean:.1f}%)",
                })

        # --- 5. Wind Speed Spike ---
        if "WS10M" in df.columns:
            ws_mean = history["WS10M"].mean()
            ws_std = history["WS10M"].std() + 1e-8
            ws_z = (latest["WS10M"] - ws_mean) / ws_std

            if ws_z > self.z_threshold:
                triggers.append({
                    "feature": "WS10M",
                    "type": "spike",
                    "z_score": round(ws_z, 2),
                    "value": round(latest["WS10M"], 1),
                    "detail": f"Wind {latest['WS10M']:.1f} m/s is {ws_z:.1f}σ "
                              f"above mean ({ws_mean:.1f} m/s)",
                })

        is_anomaly = len(triggers) > 0

        if verbose and is_anomaly:
            print(f"\n⚠️  ANOMALY DETECTED — {len(triggers)} trigger(s):")
            for t in triggers:
                print(f"    [{t['feature']}] {t['detail']}")
        elif verbose:
            print("  ✅ No anomalies detected. Using cached prediction.")

        return {
            "is_anomaly": is_anomaly,
            "n_triggers": len(triggers),
            "triggers": triggers,
            "severity": self._calc_severity(triggers),
        }

    def _calc_severity(self, triggers):
        """Calculate overall severity from triggers."""
        if not triggers:
            return "none"
        max_z = max(
            t.get("z_score", t.get("rate", 0)) for t in triggers
        )
        if max_z > 3.0 or len(triggers) >= 3:
            return "critical"
        elif max_z > 2.0 or len(triggers) >= 2:
            return "high"
        else:
            return "moderate"
