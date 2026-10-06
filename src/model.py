"""A small NumPy/pandas ridge-regression model for freight-rate prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
    "date_ordinal", "month", "day_of_week", "day_of_month", "day_of_year",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs without using load_id or the target."""
    data = frame.copy()
    # December scenario rows intentionally omit coordinates and market/quote signals.
    # Keep those feature slots and impute them from the labeled training data.
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "market_index", "quote_signal"]:
        if column not in data:
            data[column] = np.nan
    for column in ["pickup", "delivery", "equipment"]:
        if column not in data:
            data[column] = "Unknown"
    date = pd.to_datetime(data["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("date contains invalid or missing values")
    data["date_ordinal"] = date.map(pd.Timestamp.toordinal).astype(float)
    data["month"] = date.dt.month.astype(float)
    data["day_of_week"] = date.dt.dayofweek.astype(float)
    data["day_of_month"] = date.dt.day.astype(float)
    data["day_of_year"] = date.dt.dayofyear.astype(float)
    numeric = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric["log_distance"] = np.log1p(numeric["distance"].clip(lower=0))
    numeric["log_weight"] = np.log1p(numeric["weight"].clip(lower=0))
    numeric["distance_x_market"] = numeric["distance"] * numeric["market_index"]
    numeric["distance_x_quote"] = numeric["distance"] * numeric["quote_signal"]
    categorical = data[CATEGORICAL_COLUMNS].fillna("Unknown").astype(str)
    result = pd.concat([numeric, pd.get_dummies(categorical, dtype=float)], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass
class FreightRateModel:
    """Standardized ridge regression trained against log(positive rate)."""
    alpha: float = 10.0
    feature_columns: list[str] | None = None
    medians: pd.Series | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    coefficients: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "FreightRateModel":
        features = make_features(frame)
        self.feature_columns = list(features.columns)
        self.medians = features.median().fillna(0.0)
        matrix = features.fillna(self.medians).to_numpy(dtype=float)
        self.means = matrix.mean(axis=0)
        self.scales = matrix.std(axis=0)
        self.scales[self.scales == 0] = 1.0
        matrix = (matrix - self.means) / self.scales
        design = np.column_stack([np.ones(len(matrix)), matrix])
        y = np.log(np.maximum(pd.to_numeric(target).to_numpy(dtype=float), 1e-6))

        # Solve the regularized normal equations; leave the intercept unpenalized.
        gram = design.T @ design
        penalty = np.eye(gram.shape[0]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(gram + penalty, design.T @ y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if any(value is None for value in (self.feature_columns, self.medians, self.means, self.scales, self.coefficients)):
            raise RuntimeError("model must be fit or loaded before prediction")
        features = make_features(frame).reindex(columns=self.feature_columns, fill_value=0.0)
        matrix = features.fillna(self.medians).to_numpy(dtype=float)
        matrix = (matrix - self.means) / self.scales
        log_prediction = self.coefficients[0] + matrix @ self.coefficients[1:]
        return np.maximum(np.exp(np.clip(log_prediction, -20, 20)), 0.01)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path: str | Path) -> "FreightRateModel":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, FreightRateModel):
            raise TypeError("model artifact has an unexpected type")
        return model
