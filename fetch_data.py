"""
Heatwave AI — Data Fetcher
Fetches daily weather data from NASA POWER API (free, no auth)
and generates NDVI proxy for Bangkok area.
"""
import requests
import pandas as pd
import numpy as np
import config
import sys

def fetch_nasa_power():
    """Fetch daily weather data from NASA POWER API."""
    params = {
        "parameters": ",".join(config.NASA_POWER_PARAMS),
        "community": "AG",
        "longitude": config.LONGITUDE,
        "latitude": config.LATITUDE,
        "start": config.START_DATE,
        "end": config.END_DATE,
        "format": "JSON",
    }

    print(f"Fetching NASA POWER data...")
    print(f"  Location: ({config.LATITUDE}, {config.LONGITUDE})")
    print(f"  Period: {config.START_DATE} — {config.END_DATE}")
    print(f"  Parameters: {', '.join(config.NASA_POWER_PARAMS)}")

    try:
        response = requests.get(config.NASA_POWER_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        sys.exit(1)

    data = response.json()

    if "properties" not in data or "parameter" not in data["properties"]:
        print("Unexpected API response format.")
        print(data.get("message", data.get("header", "Unknown error")))
        sys.exit(1)

    parameters = data["properties"]["parameter"]

    # Build DataFrame
    records = {}
    for param_name, values in parameters.items():
        for date_str, value in values.items():
            if date_str not in records:
                records[date_str] = {}
            # NASA POWER uses -999 for missing values
            records[date_str][param_name] = value if value != -999.0 else np.nan

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index()

    print(f"  Retrieved {len(df)} daily records.")
    return df


def generate_ndvi_proxy(df):
    """
    Generate a realistic NDVI proxy for Bangkok.
    Bangkok NDVI follows a seasonal pattern:
    - Rainy season (May-Oct): Higher NDVI (~0.4-0.6)  
    - Dry season (Nov-Apr): Lower NDVI (~0.2-0.4)
    We use precipitation as a signal + seasonal sine wave.
    """
    print("Generating NDVI proxy from precipitation + seasonality...")

    # Day of year normalized to [0, 2*pi]
    doy = df.index.dayofyear
    seasonal = 0.35 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)

    # Rainfall influence: rolling 30-day sum of precipitation
    if "PRECTOTCORR" in df.columns:
        rain_30d = df["PRECTOTCORR"].rolling(window=30, min_periods=1).sum()
        rain_norm = (rain_30d - rain_30d.min()) / (rain_30d.max() - rain_30d.min() + 1e-8)
        ndvi = seasonal * 0.6 + rain_norm * 0.3 + np.random.normal(0, 0.02, len(df))
    else:
        ndvi = seasonal + np.random.normal(0, 0.02, len(df))

    df["NDVI"] = ndvi.clip(0.05, 0.8)
    return df


def clean_data(df):
    """Handle missing values and add derived features."""
    print("Cleaning data...")

    # Interpolate missing values
    before_na = df.isna().sum().sum()
    df = df.interpolate(method="linear", limit_direction="both")
    after_na = df.isna().sum().sum()
    print(f"  Filled {before_na - after_na} missing values via interpolation.")

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    # Add derived features
    df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]  # Daily temperature range
    df["heat_index_approx"] = df["T2M"] + 0.33 * df["RH2M"] / 100 * 6.105 * np.exp(
        17.27 * df["T2M"] / (237.7 + df["T2M"])
    )  # Simplified heat index

    print(f"  Final dataset: {len(df)} records, {len(df.columns)} features.")
    return df


def main():
    print("=" * 60)
    print("  HEATWAVE AI — Data Fetcher")
    print("=" * 60)

    # Step 1: Fetch from NASA POWER
    df = fetch_nasa_power()

    # Step 2: Generate NDVI proxy
    df = generate_ndvi_proxy(df)

    # Step 3: Clean and add derived features
    df = clean_data(df)

    # Step 4: Save
    df.to_csv(config.DATA_FILE)
    print(f"\nData saved to {config.DATA_FILE}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPreview:")
    print(df.head())
    print(f"\nStatistics:")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
