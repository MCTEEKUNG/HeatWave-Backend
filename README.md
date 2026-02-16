# Heatwave Prediction AI v2 🔥

Advanced AI system to predict heatwave events using temporal deep learning models.

## Architecture
```
NASA POWER → Cache → AI Model (LSTM/Transformer/RF) → Anomaly Detector → Alert
```

## Quick Start

```bash
pip install -r requirements.txt

# Fetch satellite data (NASA POWER API — free, no auth)
python fetch_data.py

# Train model (choose: lstm, transformer, rf)
python train_model.py --model lstm
python train_model.py --model transformer

# Run prediction
python pipeline.py --mode predict --model transformer

# Full pipeline (fetch + train + predict)
python pipeline.py --mode full --model lstm
```

## Models

| Model | F1 Score | Best For |
|---|---|---|
| Random Forest + XGBoost | 0.93 | Fast, interpretable |
| LSTM (2-layer) | 0.93 | Temporal patterns |
| Temporal Transformer | **0.96** | Long-range dependencies |

## Key Features
- **Anomaly Detection**: Only recomputes on temperature spikes / NDVI drops
- **SQLite Cache**: Precomputed features + cached predictions
- **Temporal Modeling**: 14-day sliding window learns past → future patterns
- **Risk Levels**: 🟢 LOW → 🟡 MEDIUM → 🟠 HIGH → 🔴 CRITICAL
