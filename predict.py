"""
Heatwave AI — Prediction Script
Load trained model, fetch latest weather data, and output heatwave risk.
"""
import requests
import pandas as pd
import numpy as np
import joblib
import config
import os
from datetime import datetime, timedelta


def fetch_recent_data(days=14):
    """Fetch recent weather data from NASA POWER for prediction."""
    end = datetime.now()
    start = end - timedelta(days=days)

    params = {
        "parameters": ",".join(config.NASA_POWER_PARAMS),
        "community": "AG",
        "longitude": config.LONGITUDE,
        "latitude": config.LATITUDE,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    print(f"Fetching recent {days} days of weather data...")
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

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index()
    df = df.interpolate(method="linear", limit_direction="both")

    # Add derived features
    df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]
    df["heat_index_approx"] = df["T2M"] + 0.33 * df["RH2M"] / 100 * 6.105 * np.exp(
        17.27 * df["T2M"] / (237.7 + df["T2M"])
    )

    # NDVI proxy
    doy = df.index.dayofyear
    seasonal = 0.35 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)
    rain_30 = df["PRECTOTCORR"].rolling(window=min(30, len(df)), min_periods=1).sum()
    rain_norm = (rain_30 - rain_30.min()) / (rain_30.max() - rain_30.min() + 1e-8)
    df["NDVI"] = (seasonal * 0.6 + rain_norm * 0.3).clip(0.05, 0.8)

    return df


def engineer_prediction_features(df):
    """Create the same features used during training."""
    feature_cols = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "WS10M",
                    "RH2M", "PS", "NDVI", "temp_range", "heat_index_approx"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    for col in feature_cols:
        for lag in range(1, config.LAG_DAYS + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in ["T2M_MAX", "PRECTOTCORR", "WS10M", "RH2M"]:
        if col in df.columns:
            df[f"{col}_roll3"] = df[col].rolling(3).mean()
            df[f"{col}_roll7"] = df[col].rolling(config.ROLLING_WINDOW).mean()

    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "T2M_MAX" in df.columns:
        df["temp_trend_3d"] = df["T2M_MAX"] - df["T2M_MAX"].shift(3)

    df.dropna(inplace=True)
    return df


def get_risk_level(probability):
    """Convert probability to risk level."""
    if probability >= 0.8:
        return "🔴 CRITICAL", "Extreme heatwave likely. Take immediate precautions."
    elif probability >= 0.6:
        return "🟠 HIGH", "High risk of heatwave. Stay hydrated and avoid outdoor activity."
    elif probability >= 0.4:
        return "🟡 MEDIUM", "Moderate heatwave risk. Monitor conditions."
    else:
        return "🟢 LOW", "Low heatwave risk. Normal conditions expected."


def predict():
    """Run heatwave prediction."""
    if not os.path.exists(config.MODEL_FILE):
        print(f"Error: {config.MODEL_FILE} not found. Run train_model.py first.")
        exit(1)

    # Load model package
    pkg = joblib.load(config.MODEL_FILE)
    model = pkg["model"]
    scaler = pkg["scaler"]
    feature_names = pkg["features"]

    print("=" * 60)
    print("  HEATWAVE AI — Prediction")
    print("=" * 60)

    # Fetch recent data
    df = fetch_recent_data(days=14)
    df = engineer_prediction_features(df)

    if df.empty:
        print("Not enough data for prediction.")
        return

    # Use the latest row
    latest = df.iloc[[-1]]
    latest_date = latest.index[0]

    # Align features
    missing = [f for f in feature_names if f not in latest.columns]
    for m in missing:
        latest[m] = 0  # Fill missing features with 0

    X = latest[feature_names]
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]
    try:
        probability = model.predict_proba(X_scaled)[0][1]
    except Exception:
        probability = float(prediction)

    risk_level, advice = get_risk_level(probability)

    # Output
    print(f"\n📅 Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"🌡️  Max Temp: {latest['T2M_MAX'].values[0]:.1f} °C")
    print(f"💧 Rainfall: {latest['PRECTOTCORR'].values[0]:.1f} mm")
    print(f"💨 Wind: {latest['WS10M'].values[0]:.1f} m/s")
    print(f"💦 Humidity: {latest['RH2M'].values[0]:.1f} %")
    print(f"\n{'='*40}")
    print(f"  Heatwave Probability: {probability:.1%}")
    print(f"  Risk Level: {risk_level}")
    print(f"  {advice}")
    print(f"{'='*40}")

    return {
        "date": latest_date.strftime("%Y-%m-%d"),
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "advice": advice,
        "prediction": int(prediction),
    }


if __name__ == "__main__":
    result = predict()
