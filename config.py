# ==============================
# Heatwave AI v2 — Configuration
# ==============================
import os

# Bangkok center coordinates
LATITUDE = 13.7563
LONGITUDE = 100.5018

# Bangkok 5km bounding box (for map overlay)
BBOX = {
    "north": 13.7788,
    "south": 13.7338,
    "east": 100.5243,
    "west": 100.4793,
}

# NASA POWER API
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMS = [
    "T2M",           # Temperature at 2m (C) — daily mean
    "T2M_MAX",       # Max temperature at 2m (C)
    "T2M_MIN",       # Min temperature at 2m (C)
    "PRECTOTCORR",   # Precipitation corrected (mm/day)
    "WS10M",         # Wind speed at 10m (m/s)
    "RH2M",          # Relative humidity at 2m (%)
    "PS",            # Surface pressure (kPa)
    "ALLSKY_SFC_SW_DWN",  # Solar radiation (kW-hr/m2/day)
]

# Date range for training data
START_DATE = "20190101"
END_DATE = "20231231"

# Heatwave definition
HEATWAVE_PERCENTILE = 90
HEATWAVE_CONSECUTIVE_DAYS = 3

# Model settings
MODEL_TYPE = "lstm"               # Options: "rf", "lstm", "transformer"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Temporal model settings
SEQ_LEN = 14                     # Sliding window size (days of history)
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Legacy RF settings
LAG_DAYS = 7
ROLLING_WINDOW = 7

# Anomaly detection
ANOMALY_Z_THRESHOLD = 2.0
ANOMALY_RATE_THRESHOLD = 1.5
ANOMALY_LOOKBACK = 30

# ==============================
# Output directories
# ==============================
OUTPUT_DIR = "output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")

# Create directories
for d in [DATA_DIR, MODELS_DIR, CHARTS_DIR, MAPS_DIR]:
    os.makedirs(d, exist_ok=True)

# Output file paths
DATA_FILE = os.path.join(DATA_DIR, "bangkok_heatwave_data.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "heatwave_model.joblib")
LSTM_MODEL_FILE = os.path.join(MODELS_DIR, "heatwave_lstm.pt")
TRANSFORMER_MODEL_FILE = os.path.join(MODELS_DIR, "heatwave_transformer.pt")
CACHE_DB = os.path.join(DATA_DIR, "heatwave_cache.db")

# Risk level colors (RGBA)
RISK_COLORS = {
    "LOW":      (0, 180, 0, 64),      # Green, 25% opacity
    "MEDIUM":   (255, 200, 0, 89),     # Yellow, 35% opacity
    "HIGH":     (255, 120, 0, 115),    # Orange, 45% opacity
    "CRITICAL": (220, 0, 0, 140),      # Red, 55% opacity
}
