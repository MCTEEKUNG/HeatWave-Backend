"""
Heatwave AI — LSTM Temporal Model
Learns: heatwave(t) = f(features(t-1..t-N))
Uses sliding windows over time-series weather data.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class HeatwaveSequenceDataset(Dataset):
    """Converts time-series DataFrame into sliding window sequences."""

    def __init__(self, X, y, seq_len=14):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_seq, y_target


class LSTMHeatwaveModel(nn.Module):
    """
    LSTM architecture for heatwave prediction.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1) — heatwave probability
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.classifier(last_hidden)
        return out.squeeze(-1)


class LSTMTrainer:
    """Handles training, evaluation, and prediction for the LSTM model."""

    def __init__(self, n_features, seq_len=14, hidden_size=64, num_layers=2,
                 lr=0.001, epochs=50, batch_size=32, device=None):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LSTMHeatwaveModel(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.scaler_mean = None
        self.scaler_std = None

    def _normalize(self, X, fit=False):
        """Z-score normalization."""
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model."""
        X_train_norm = self._normalize(X_train, fit=True)
        
        train_ds = HeatwaveSequenceDataset(X_train_norm, y_train, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None:
            X_val_norm = self._normalize(X_val)
            val_ds = HeatwaveSequenceDataset(X_val_norm, y_val, self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        print(f"Training LSTM on {self.device} | {self.epochs} epochs | "
              f"seq_len={self.seq_len} | features={X_train.shape[1]}")

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
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

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            print(f"  Restored best model (val_loss: {best_val_loss:.4f})")

    def predict(self, X):
        """Predict heatwave probabilities."""
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
        """Predict from a single window (seq_len, features)."""
        X_norm = self._normalize(X_window.reshape(1, *X_window.shape) 
                                 if X_window.ndim == 2 else X_window)
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_norm).to(self.device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            pred = self.model(x)
        return pred.cpu().item()

    def save(self, path):
        """Save model + normalization params."""
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "config": {
                "n_features": self.model.lstm.input_size,
                "hidden_size": self.model.lstm.hidden_size,
                "num_layers": self.model.lstm.num_layers,
                "seq_len": self.seq_len,
            }
        }, path)
        print(f"LSTM model saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        """Load a saved LSTM model."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        
        trainer = cls(
            n_features=cfg["n_features"],
            seq_len=cfg["seq_len"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.scaler_mean = checkpoint["scaler_mean"]
        trainer.scaler_std = checkpoint["scaler_std"]
        return trainer
