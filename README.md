# Heatwave Prediction AI v2 🔥

Advanced AI system that predicts heatwave events in Bangkok using temporal deep learning models and NASA satellite weather data.

## Architecture

```
Satellite Data (NASA POWER API)
    ↓
Data Cleaning + Derived Features (temp_range, heat_index)
    ↓
Low-cost Preprocessing + Caching (SQLite)
    ↓
AI Prediction Model (Transformer / LSTM / RF+XGBoost)
    ↓
Anomaly Detector (5 Z-score triggers)
    ↓  Normal → use cached prediction (fast)
    ↓  Anomaly → full recompute (accurate)
    ↓
Risk Map Output (OSM overlay PNG)
    ↓
🟢 LOW / 🟡 MEDIUM / 🟠 HIGH / 🔴 CRITICAL
```

## Features

- **3 AI Models**: Transformer (F1: 0.96), LSTM (F1: 0.93), Random Forest + XGBoost (F1: 0.93)
- **Temporal Modeling**: 14-day sliding window learns `heatwave(t) = f(weather(t-1..t-14))`
- **5-Trigger Anomaly Detection**: Temperature spike, rapid temperature increase, NDVI drop, humidity drop, wind speed spike
- **Smart Caching**: SQLite-based weather feature & prediction cache with incremental updates
- **Derived Features**: Heat index approximation, daily temperature range, lag features, rolling averages
- **NDVI Proxy**: Realistic vegetation index generated from precipitation + seasonal patterns
- **OSM Map Visualization**: Risk level color overlay on OpenStreetMap of Bangkok
- **NASA POWER API**: Free, no authentication required — 8 weather parameters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch 5 years of satellite data (2019-2023)
python fetch_data.py

# Step 2: Train a model (choose: transformer, lstm, rf)
python train_model.py --model transformer

# Step 3: Run prediction + generate risk map
python pipeline.py --mode predict --model transformer
```

### Full Pipeline (all steps at once)
```bash
python pipeline.py --mode full --model transformer
```

### Standalone Prediction (RF model only)
```bash
python predict.py
```

## Project Structure

```
heatwave-ai/
├── config.py                # All settings & hyperparameters
├── fetch_data.py            # NASA POWER API data fetcher + NDVI proxy
├── train_model.py           # Model training (--model rf/lstm/transformer)
├── predict.py               # Standalone RF prediction script
├── pipeline.py              # Full pipeline orchestrator (fetch/train/predict/full)
├── visualize_map.py         # OSM risk map generator (Bangkok basemap + overlay)
├── cache_manager.py         # SQLite weather feature & prediction cache
├── anomaly_detector.py      # Z-score anomaly trigger system (5 triggers)
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── lstm_model.py        # LSTM temporal model (PyTorch)
│   └── transformer_model.py # Temporal Transformer model (PyTorch)
└── output/
    ├── data/                # CSV data + SQLite cache
    ├── models/              # Saved model files (.pt, .joblib)
    ├── charts/              # Confusion matrices, feature importance
    └── maps/                # Heatwave risk map PNGs
```

## Models

| Model | F1 Score | Training Time | Best For |
|---|---|---|---|
| **Transformer** | **0.96** ✅ | ~60s | Best accuracy, seasonal patterns |
| LSTM | 0.93 | ~30s | Temporal sequences, balanced |
| RF + XGBoost | 0.93 | ~5s | Fast, interpretable |

## Pipeline Modes

```bash
python pipeline.py --mode <MODE> --model <MODEL>
```

| Mode | Description |
|---|---|
| `fetch` | Download satellite data only |
| `train` | Train model only |
| `predict` | Predict + anomaly check + map |
| `full` | All of the above |

| Model | Description |
|---|---|
| `transformer` | Temporal Transformer (recommended) |
| `lstm` | LSTM recurrent network |
| `rf` | Random Forest + XGBoost ensemble |

## Anomaly Detection

The `AnomalyDetector` monitors 5 weather triggers using Z-score and rate-of-change analysis over a 30-day lookback window:

| # | Trigger | Condition | Description |
|---|---|---|---|
| 1 | **Temperature Spike** | Z > 2.0 | T2M_MAX exceeds 2σ above 30-day mean |
| 2 | **Rapid Temp Increase** | Rate Z > 1.5 | Temperature rose sharply over 3 days |
| 3 | **NDVI Drop** | Z > 1.5 | Vegetation index dropped below 1.5σ |
| 4 | **Humidity Drop** | Z > 2.0 | Relative humidity dropped 2σ below mean |
| 5 | **Wind Speed Spike** | Z > 2.0 | Wind speed exceeds 2σ above mean |

**Severity Levels**: `none` → `moderate` (1 trigger) → `high` (2+ triggers or Z > 2.0) → `critical` (3+ triggers or Z > 3.0)

## Configuration

Edit `config.py` to customize:

### Location & Data

| Setting | Default | Description |
|---|---|---|
| `LATITUDE` | `13.7563` | Bangkok center latitude |
| `LONGITUDE` | `100.5018` | Bangkok center longitude |
| `BBOX` | 5km box | Bounding box for map overlay |
| `START_DATE` | `"20190101"` | Training data start date |
| `END_DATE` | `"20231231"` | Training data end date |

### Heatwave Definition

| Setting | Default | Description |
|---|---|---|
| `HEATWAVE_PERCENTILE` | `90` | Top N% temperature = heatwave |
| `HEATWAVE_CONSECUTIVE_DAYS` | `3` | Consecutive hot days to qualify |

### Model Hyperparameters

| Setting | Default | Description |
|---|---|---|
| `MODEL_TYPE` | `"lstm"` | Default model: `"transformer"`, `"lstm"`, `"rf"` |
| `SEQ_LEN` | `14` | Sliding window size (days) |
| `EPOCHS` | `50` | Training epochs for deep learning |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `0.001` | Optimizer learning rate |
| `LSTM_HIDDEN` | `64` | LSTM hidden layer size |
| `LSTM_LAYERS` | `2` | Number of LSTM layers |
| `TRANSFORMER_D_MODEL` | `64` | Transformer embedding dimension |
| `TRANSFORMER_NHEAD` | `4` | Transformer attention heads |
| `TRANSFORMER_LAYERS` | `2` | Number of Transformer encoder layers |

### Anomaly Detection

| Setting | Default | Description |
|---|---|---|
| `ANOMALY_Z_THRESHOLD` | `2.0` | Z-score trigger for spike/drop anomalies |
| `ANOMALY_RATE_THRESHOLD` | `1.5` | Z-score trigger for rate-of-change anomalies |
| `ANOMALY_LOOKBACK` | `30` | Rolling window size (days) for anomaly baseline |

## NASA POWER Parameters

8 daily weather variables fetched from satellite data:

| Parameter | Description |
|---|---|
| `T2M` | Temperature at 2m — daily mean (°C) |
| `T2M_MAX` | Max temperature at 2m (°C) |
| `T2M_MIN` | Min temperature at 2m (°C) |
| `PRECTOTCORR` | Precipitation corrected (mm/day) |
| `WS10M` | Wind speed at 10m (m/s) |
| `RH2M` | Relative humidity at 2m (%) |
| `PS` | Surface pressure (kPa) |
| `ALLSKY_SFC_SW_DWN` | Solar radiation (kW-hr/m²/day) |

**Source**: [NASA POWER API](https://power.larc.nasa.gov/) — Free, no authentication required

## Risk Levels

| Level | Probability | Color | Action |
|---|---|---|---|
| 🟢 LOW | < 40% | Green (25% opacity) | Normal conditions |
| 🟡 MEDIUM | 40-60% | Yellow (35% opacity) | Monitor conditions |
| 🟠 HIGH | 60-80% | Orange (45% opacity) | Stay hydrated, avoid outdoors |
| 🔴 CRITICAL | > 80% | Red (55% opacity) | Take immediate precautions |

## Dependencies

```
requests        # HTTP client for NASA API
pandas          # Data manipulation
numpy           # Numerical computing
scikit-learn    # ML preprocessing & Random Forest
xgboost         # Gradient boosting ensemble
matplotlib      # Plotting & chart generation
seaborn         # Statistical visualization
joblib          # Model serialization
torch           # PyTorch (LSTM & Transformer)
Pillow          # Image processing for OSM tiles
```

## License

MIT
