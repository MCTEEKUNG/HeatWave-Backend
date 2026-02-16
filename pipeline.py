"""
Heatwave AI v2 — Pipeline Orchestrator
End-to-end: Fetch → Cache → Train/Predict → Anomaly → Alert

Usage:
    python pipeline.py --mode train --model lstm
    python pipeline.py --mode predict
    python pipeline.py --mode full --model transformer
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import config
from fetch_data import fetch_nasa_power, generate_ndvi_proxy, clean_data
from cache_manager import CacheManager
from anomaly_detector import AnomalyDetector


def run_fetch(cache):
    """Fetch data and populate cache."""
    print("\n📡 Step 1: Fetching satellite data...")
    
    # Check if cache has data and only fetch new
    latest = cache.get_latest_date()
    if latest:
        print(f"  Cache has data up to {latest}")
        cached_df = cache.get_cached_data()
        if len(cached_df) > 100:
            print(f"  Using {len(cached_df)} cached records.")
            return cached_df

    # Full fetch
    from fetch_data import main as fetch_main
    fetch_main()

    df = pd.read_csv(config.DATA_FILE, index_col="date", parse_dates=True)
    cache.store_data(df)
    return df


def run_train(model_type, df=None):
    """Train the specified model."""
    print(f"\n🧠 Step 2: Training [{model_type.upper()}] model...")

    if df is None:
        df = pd.read_csv(config.DATA_FILE, index_col="date", parse_dates=True)

    from train_model import label_heatwaves, train_rf, train_lstm, train_transformer

    df = label_heatwaves(df)

    if model_type == "rf":
        train_rf(df)
    elif model_type == "lstm":
        train_lstm(df)
    elif model_type == "transformer":
        train_transformer(df)


def run_predict(model_type, cache):
    """Run prediction with anomaly detection."""
    print(f"\n🔮 Step 3: Prediction [{model_type.upper()}]")

    # Fetch recent data for prediction
    import requests
    end = datetime.now()
    start = end - timedelta(days=30)

    params = {
        "parameters": ",".join(config.NASA_POWER_PARAMS),
        "community": "AG",
        "longitude": config.LONGITUDE,
        "latitude": config.LATITUDE,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    print("  Fetching recent 30 days of weather data...")
    response = requests.get(config.NASA_POWER_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    parameters = data["properties"]["parameter"]

    records = {}
    for param_name, values in parameters.items():
        for date_str, value in values.items():
            if date_str not in records:
                records[date_str] = {}
            records[date_str][param_name] = value if value != -999.0 else np.nan

    df_recent = pd.DataFrame.from_dict(records, orient="index")
    df_recent.index = pd.to_datetime(df_recent.index, format="%Y%m%d")
    df_recent.index.name = "date"
    df_recent = df_recent.sort_index()
    df_recent = df_recent.interpolate(method="linear", limit_direction="both")

    # Add derived features
    df_recent["temp_range"] = df_recent["T2M_MAX"] - df_recent["T2M_MIN"]
    df_recent["heat_index_approx"] = (
        df_recent["T2M"] + 0.33 * df_recent["RH2M"] / 100 * 6.105
        * np.exp(17.27 * df_recent["T2M"] / (237.7 + df_recent["T2M"]))
    )

    # NDVI proxy
    doy = df_recent.index.dayofyear
    seasonal = 0.35 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)
    rain_30 = df_recent["PRECTOTCORR"].rolling(window=min(30, len(df_recent)), min_periods=1).sum()
    rain_norm = (rain_30 - rain_30.min()) / (rain_30.max() - rain_30.min() + 1e-8)
    df_recent["NDVI"] = (seasonal * 0.6 + rain_norm * 0.3).clip(0.05, 0.8)

    df_recent.dropna(inplace=True)

    # --- Anomaly Detection ---
    detector = AnomalyDetector(
        z_threshold=config.ANOMALY_Z_THRESHOLD,
        rate_threshold=config.ANOMALY_RATE_THRESHOLD,
        lookback=config.ANOMALY_LOOKBACK,
    )
    anomaly_result = detector.detect(df_recent)

    # --- Prediction ---
    latest_date = df_recent.index[-1].strftime("%Y-%m-%d")

    # Check cache first (if no anomaly)
    if not anomaly_result["is_anomaly"]:
        cached = cache.get_cached_prediction(latest_date)
        if cached:
            print(f"\n  Using cached prediction for {latest_date}")
            return _display_result(latest_date, cached["probability"],
                                    df_recent.iloc[-1], anomaly_result, cached=True)

    # Run model prediction
    probability = _model_predict(model_type, df_recent)

    # Risk level
    risk_level, advice = _get_risk_level(probability)

    # Cache the prediction
    cache.store_prediction(latest_date, probability, risk_level, model_type,
                           is_anomaly=anomaly_result["is_anomaly"])

    # Generate Map
    try:
        from visualize_map import render_heatwave_map
        map_path = render_heatwave_map(
            risk_level=risk_level,
            probability=probability,
            weather_data=df_recent.iloc[-1].to_dict(),
            date_str=latest_date
        )
        print(f"  🗺️  Map saved: {map_path}")
    except Exception as e:
        print(f"  ⚠️  Map generation failed: {e}")

    return _display_result(latest_date, probability, df_recent.iloc[-1], anomaly_result)


def _model_predict(model_type, df_recent):
    """Run prediction using the specified model."""
    from train_model import get_raw_features
    feature_cols = get_raw_features(df_recent)

    if model_type == "lstm" and os.path.exists(config.LSTM_MODEL_FILE):
        from models.lstm_model import LSTMTrainer
        trainer = LSTMTrainer.load(config.LSTM_MODEL_FILE)
        X = df_recent[feature_cols].values
        if len(X) >= trainer.seq_len:
            window = X[-trainer.seq_len:]
            prob = trainer.predict_single(window)
            return prob

    elif model_type == "transformer" and os.path.exists(config.TRANSFORMER_MODEL_FILE):
        from models.transformer_model import TransformerTrainer
        trainer = TransformerTrainer.load(config.TRANSFORMER_MODEL_FILE)
        X = df_recent[feature_cols].values
        if len(X) >= trainer.seq_len:
            window = X[-trainer.seq_len:]
            prob = trainer.predict_single(window)
            return prob

    elif os.path.exists(config.MODEL_FILE):
        import joblib
        pkg = joblib.load(config.MODEL_FILE)
        model = pkg["model"]
        scaler = pkg["scaler"]
        features = pkg["features"]

        # Build features with lags
        df_feat = df_recent.copy()
        for col in feature_cols:
            for lag in range(1, config.LAG_DAYS + 1):
                df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)
        for col in ["T2M_MAX", "PRECTOTCORR", "WS10M", "RH2M"]:
            if col in df_feat.columns:
                df_feat[f"{col}_roll3"] = df_feat[col].rolling(3).mean()
                df_feat[f"{col}_roll7"] = df_feat[col].rolling(config.ROLLING_WINDOW).mean()
        df_feat["month"] = df_feat.index.month
        df_feat["day_of_year"] = df_feat.index.dayofyear
        df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
        df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
        if "T2M_MAX" in df_feat.columns:
            df_feat["temp_trend_3d"] = df_feat["T2M_MAX"] - df_feat["T2M_MAX"].shift(3)
        df_feat.dropna(inplace=True)

        if not df_feat.empty:
            latest = df_feat.iloc[[-1]]
            missing = [f for f in features if f not in latest.columns]
            for m in missing:
                latest[m] = 0
            X = latest[features]
            X_scaled = scaler.transform(X)
            try:
                prob = model.predict_proba(X_scaled)[0][1]
            except Exception:
                prob = float(model.predict(X_scaled)[0])
            return prob

    print("  ⚠️  No trained model found. Using simple threshold.")
    return 0.0


def _get_risk_level(probability):
    if probability >= 0.8:
        return "CRITICAL", "Extreme heatwave likely. Take immediate precautions."
    elif probability >= 0.6:
        return "HIGH", "High risk of heatwave. Stay hydrated and avoid outdoor activity."
    elif probability >= 0.4:
        return "MEDIUM", "Moderate heatwave risk. Monitor conditions."
    else:
        return "LOW", "Low heatwave risk. Normal conditions expected."


def _display_result(date, probability, latest_data, anomaly_result, cached=False):
    risk_level, advice = _get_risk_level(probability)

    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    icon = icons.get(risk_level, "⚪")

    print(f"\n{'=' * 50}")
    print(f"  📅 Date: {date}")
    print(f"  🌡️  Max Temp: {latest_data.get('T2M_MAX', 'N/A'):.1f} C")
    print(f"  💧 Rainfall: {latest_data.get('PRECTOTCORR', 'N/A'):.1f} mm")
    print(f"  💨 Wind: {latest_data.get('WS10M', 'N/A'):.1f} m/s")
    print(f"  💦 Humidity: {latest_data.get('RH2M', 'N/A'):.1f} %")
    print(f"  🌿 NDVI: {latest_data.get('NDVI', 'N/A'):.4f}")
    print(f"{'=' * 50}")
    print(f"  Heatwave Probability: {probability:.1%}")
    print(f"  Risk Level: {icon} {risk_level}")
    print(f"  {advice}")
    if cached:
        print(f"  📦 (from cache)")
    if anomaly_result["is_anomaly"]:
        print(f"  ⚠️  Anomaly severity: {anomaly_result['severity']}")
    print(f"{'=' * 50}")

    return {"date": date, "probability": probability, "risk_level": risk_level, "advice": advice}


def main():
    parser = argparse.ArgumentParser(description="Heatwave AI v2 Pipeline")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["fetch", "train", "predict", "full"],
                        help="Pipeline mode")
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE,
                        choices=["rf", "lstm", "transformer"],
                        help="Model type")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  HEATWAVE AI v2 — Pipeline [{args.mode.upper()}]")
    print(f"  Model: {args.model.upper()} | Bangkok (5km)")
    print("=" * 60)

    cache = CacheManager(config.CACHE_DB)

    try:
        if args.mode in ("fetch", "full"):
            run_fetch(cache)

        if args.mode in ("train", "full"):
            run_train(args.model)

        if args.mode in ("predict", "full"):
            run_predict(args.model, cache)

    finally:
        cache.close()

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
