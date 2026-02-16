"""
Heatwave AI — Temporal Fusion Transformer
Multi-head self-attention over time steps for heatwave prediction.
Captures long-range seasonal patterns better than LSTM.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader
from models.lstm_model import HeatwaveSequenceDataset


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time steps."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerModel(nn.Module):
    """
    Transformer encoder for time-series heatwave prediction.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1) — heatwave probability
    """

    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=128, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)           # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)           # Add positional encoding
        x = self.transformer_encoder(x)   # Self-attention
        # Use mean pooling over sequence dimension
        x = x.mean(dim=1)                # → (batch, d_model)
        return self.classifier(x).squeeze(-1)


class TransformerTrainer:
    """Handles training, evaluation, and prediction for the Transformer model."""

    def __init__(self, n_features, seq_len=14, d_model=64, nhead=4, 
                 num_layers=2, lr=0.0005, epochs=50, batch_size=32, device=None):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TemporalTransformerModel(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.scaler_mean = None
        self.scaler_std = None

    def _normalize(self, X, fit=False):
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Transformer model."""
        X_norm = self._normalize(X_train, fit=True)
        train_ds = HeatwaveSequenceDataset(X_norm, y_train, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None:
            X_val_norm = self._normalize(X_val)
            val_ds = HeatwaveSequenceDataset(X_val_norm, y_val, self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        print(f"Training Transformer on {self.device} | {self.epochs} epochs | "
              f"seq_len={self.seq_len} | features={X_train.shape[1]}")

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss_str = ""
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_batch)
                        val_loss += self.criterion(pred, y_batch).item()
                val_loss /= len(val_loader)
                val_loss_str = f" | val_loss: {val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"train_loss: {train_loss:.4f}{val_loss_str}")

        if best_state:
            self.model.load_state_dict(best_state)
            print(f"  Restored best model (val_loss: {best_val_loss:.4f})")

    def predict(self, X):
        X_norm = self._normalize(X)
        self.model.eval()
        ds = HeatwaveSequenceDataset(X_norm, np.zeros(len(X_norm)), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                preds.extend(pred.cpu().numpy())
        return np.array(preds)

    def predict_single(self, X_window):
        X_norm = self._normalize(
            X_window.reshape(1, *X_window.shape) if X_window.ndim == 2 else X_window
        )
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_norm).to(self.device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            pred = self.model(x)
        return pred.cpu().item()

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "config": {
                "n_features": self.model.input_proj.in_features,
                "d_model": self.model.input_proj.out_features,
                "nhead": self.model.transformer_encoder.layers[0].self_attn.num_heads,
                "num_layers": len(self.model.transformer_encoder.layers),
                "seq_len": self.seq_len,
            }
        }, path)
        print(f"Transformer model saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        trainer = cls(
            n_features=cfg["n_features"],
            seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.scaler_mean = checkpoint["scaler_mean"]
        trainer.scaler_std = checkpoint["scaler_std"]
        return trainer
