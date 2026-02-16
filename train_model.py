"""
Heatwave AI v2 — Model Training
Supports: Random Forest (rf), LSTM (lstm), Transformer (transformer)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
import argparse


def load_data():
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: {config.DATA_FILE} not found. Run fetch_data.py first.")
        exit(1)
    df = pd.read_csv(config.DATA_FILE, index_col="date", parse_dates=True)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def label_heatwaves(df):
    threshold = df["T2M_MAX"].quantile(config.HEATWAVE_PERCENTILE / 100)
    print(f"Heatwave Threshold: T2M_MAX > {threshold:.2f} C (top {100 - config.HEATWAVE_PERCENTILE}%)")
    df["hot_day"] = (df["T2M_MAX"] > threshold).astype(int)
    consecutive = df["hot_day"].rolling(
        window=config.HEATWAVE_CONSECUTIVE_DAYS,
        min_periods=config.HEATWAVE_CONSECUTIVE_DAYS
    ).sum()
    df["is_heatwave"] = (consecutive >= config.HEATWAVE_CONSECUTIVE_DAYS).astype(int)
    hw_count = df["is_heatwave"].sum()
    print(f"Heatwave days: {hw_count} / {len(df)} ({hw_count/len(df)*100:.1f}%)")
    return df


def get_raw_features(df):
    """Get the base feature columns for temporal models."""
    base_cols = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "WS10M",
                 "RH2M", "PS", "ALLSKY_SFC_SW_DWN", "NDVI", "temp_range", "heat_index_approx"]
    return [c for c in base_cols if c in df.columns]


# ============================================================
# Random Forest + XGBoost (Legacy — with manual lag features)
# ============================================================
def train_rf(df):
    print("\n" + "=" * 50)
    print("Training Random Forest + XGBoost Ensemble")
    print("=" * 50)

    feature_cols = get_raw_features(df)

    # Create lag & rolling features
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

    exclude = ["is_heatwave", "hot_day"]
    features = [c for c in df.columns if c not in exclude]
    X = df[features]
    y = df["is_heatwave"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced",
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                  scale_pos_weight=scale_pos, random_state=config.RANDOM_STATE,
                                  eval_metric="logloss")
    xgb_clf.fit(X_train_s, y_train)

    ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb_clf)], voting="soft")
    ensemble.fit(X_train_s, y_train)

    ens_pred = ensemble.predict(X_test_s)
    print(classification_report(y_test, ens_pred, target_names=["Normal", "Heatwave"]))

    plot_results(y_test, ens_pred, rf.feature_importances_, features, "RF+XGB")

    pkg = {"model": ensemble, "scaler": scaler, "features": features,
           "threshold_temp": df["T2M_MAX"].quantile(config.HEATWAVE_PERCENTILE / 100)}
    joblib.dump(pkg, config.MODEL_FILE)
    print(f"Model saved to {config.MODEL_FILE}")


# ============================================================
# LSTM Temporal Model
# ============================================================
def train_lstm(df):
    print("\n" + "=" * 50)
    print("Training LSTM Temporal Model")
    print("=" * 50)

    from models.lstm_model import LSTMTrainer

    feature_cols = get_raw_features(df)
    X = df[feature_cols].values
    y = df["is_heatwave"].values.astype(np.float32)

    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    trainer = LSTMTrainer(
        n_features=len(feature_cols),
        seq_len=config.SEQ_LEN,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
    )
    trainer.fit(X_train, y_train, X_test, y_test)

    # Evaluate
    preds_prob = trainer.predict(X_test)
    preds = (preds_prob > 0.5).astype(int)
    y_eval = y_test[config.SEQ_LEN:]  # Shifted by seq_len

    if len(preds) != len(y_eval):
        min_len = min(len(preds), len(y_eval))
        preds = preds[:min_len]
        y_eval = y_eval[:min_len]

    print(classification_report(y_eval.astype(int), preds, target_names=["Normal", "Heatwave"]))

    plot_results(y_eval.astype(int), preds, None, feature_cols, "LSTM")

    trainer.save(config.LSTM_MODEL_FILE)


# ============================================================
# Transformer Temporal Model
# ============================================================
def train_transformer(df):
    print("\n" + "=" * 50)
    print("Training Temporal Transformer Model")
    print("=" * 50)

    from models.transformer_model import TransformerTrainer

    feature_cols = get_raw_features(df)
    X = df[feature_cols].values
    y = df["is_heatwave"].values.astype(np.float32)

    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    trainer = TransformerTrainer(
        n_features=len(feature_cols),
        seq_len=config.SEQ_LEN,
        d_model=config.TRANSFORMER_D_MODEL,
        nhead=config.TRANSFORMER_NHEAD,
        num_layers=config.TRANSFORMER_LAYERS,
        lr=config.LEARNING_RATE * 0.5,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
    )
    trainer.fit(X_train, y_train, X_test, y_test)

    preds_prob = trainer.predict(X_test)
    preds = (preds_prob > 0.5).astype(int)
    y_eval = y_test[config.SEQ_LEN:]

    if len(preds) != len(y_eval):
        min_len = min(len(preds), len(y_eval))
        preds = preds[:min_len]
        y_eval = y_eval[:min_len]

    print(classification_report(y_eval.astype(int), preds, target_names=["Normal", "Heatwave"]))

    plot_results(y_eval.astype(int), preds, None, feature_cols, "Transformer")

    trainer.save(config.TRANSFORMER_MODEL_FILE)


# ============================================================
# Visualization
# ============================================================
def plot_results(y_true, y_pred, importances, features, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Normal", "Heatwave"],
                yticklabels=["Normal", "Heatwave"])
    plt.title(f"Heatwave Prediction — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace('+', '_')}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved.")

    if importances is not None:
        top_n = min(15, len(features))
        top_idx = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_idx)), importances[top_idx], color="#e74c3c")
        plt.yticks(range(len(top_idx)), [features[i] for i in top_idx])
        plt.title(f"Top {top_n} Feature Importances — {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{model_name.lower().replace('+', '_')}.png", dpi=150)
        plt.close()
        print(f"Feature importance saved.")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Heatwave AI — Train Model")
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE,
                        choices=["rf", "lstm", "transformer"],
                        help="Model type to train")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  HEATWAVE AI — Training [{args.model.upper()}]")
    print("=" * 60)

    df = load_data()
    df = label_heatwaves(df)

    if args.model == "rf":
        train_rf(df)
    elif args.model == "lstm":
        train_lstm(df)
    elif args.model == "transformer":
        train_transformer(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
