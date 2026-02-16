"""
Heatwave AI — Cache Manager
SQLite-based feature cache for precomputed features.
Supports incremental updates (only fetch/compute new days).
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import config


DB_PATH = "heatwave_cache.db"


class CacheManager:
    """Manages a SQLite cache of precomputed weather features."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_cache (
                date TEXT PRIMARY KEY,
                T2M REAL, T2M_MAX REAL, T2M_MIN REAL,
                PRECTOTCORR REAL, WS10M REAL, RH2M REAL, PS REAL,
                ALLSKY_SFC_SW_DWN REAL, NDVI REAL,
                temp_range REAL, heat_index_approx REAL,
                is_heatwave INTEGER DEFAULT NULL,
                prediction REAL DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions_cache (
                date TEXT PRIMARY KEY,
                probability REAL,
                risk_level TEXT,
                model_type TEXT,
                is_anomaly INTEGER DEFAULT 0,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get_latest_date(self):
        """Get the most recent date in cache."""
        cursor = self.conn.execute(
            "SELECT MAX(date) FROM weather_cache"
        )
        result = cursor.fetchone()[0]
        return result

    def get_cached_data(self, start_date=None, end_date=None):
        """Retrieve cached data as DataFrame."""
        query = "SELECT * FROM weather_cache"
        conditions = []
        params = []

        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date"
        df = pd.read_sql(query, self.conn, parse_dates=["date"])
        if not df.empty:
            df.set_index("date", inplace=True)
        return df

    def store_data(self, df):
        """Store DataFrame into cache (upsert)."""
        if df.empty:
            return

        df_to_store = df.copy()
        if df_to_store.index.name == "date":
            df_to_store = df_to_store.reset_index()

        df_to_store["date"] = pd.to_datetime(df_to_store["date"]).dt.strftime("%Y-%m-%d")

        # Get existing columns in the table
        cursor = self.conn.execute("PRAGMA table_info(weather_cache)")
        table_cols = {row[1] for row in cursor.fetchall()}

        # Only include columns that exist in the table
        valid_cols = [c for c in df_to_store.columns if c in table_cols]
        df_to_store = df_to_store[valid_cols]

        for _, row in df_to_store.iterrows():
            cols = list(row.index)
            vals = list(row.values)
            placeholders = ",".join(["?"] * len(cols))
            col_names = ",".join(cols)
            update_set = ",".join([f"{c}=excluded.{c}" for c in cols if c != "date"])

            self.conn.execute(
                f"INSERT INTO weather_cache ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT(date) DO UPDATE SET {update_set}",
                vals,
            )
        self.conn.commit()
        print(f"  Cached {len(df_to_store)} records to {self.db_path}")

    def store_prediction(self, date, probability, risk_level, model_type, is_anomaly=False):
        """Cache a prediction result."""
        self.conn.execute(
            "INSERT OR REPLACE INTO predictions_cache "
            "(date, probability, risk_level, model_type, is_anomaly) "
            "VALUES (?, ?, ?, ?, ?)",
            (date, probability, risk_level, model_type, int(is_anomaly)),
        )
        self.conn.commit()

    def get_cached_prediction(self, date):
        """Get a cached prediction for a specific date."""
        cursor = self.conn.execute(
            "SELECT probability, risk_level, model_type, is_anomaly "
            "FROM predictions_cache WHERE date = ?",
            (date,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "probability": row[0],
                "risk_level": row[1],
                "model_type": row[2],
                "is_anomaly": bool(row[3]),
            }
        return None

    def needs_update(self, target_date):
        """Check if we need to fetch new data up to target_date."""
        latest = self.get_latest_date()
        if latest is None:
            return True
        return target_date > latest

    def close(self):
        self.conn.close()
