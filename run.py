"""Validate the model, fit on all labeled rows, and create submission files."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.model import FreightRateModel


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-9)) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train_path = args.data_dir / "train-test.csv"
    train = pd.read_csv(train_path)
    required = {"posted_rate", "date", "load_id", "pickup", "delivery", "distance", "equipment", "weight", "market_index", "quote_signal"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"training data is missing columns: {sorted(missing)}")
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    train["posted_rate"] = pd.to_numeric(train["posted_rate"], errors="raise")
    if (train["posted_rate"] <= 0).any():
        raise ValueError("posted_rate must be positive")

    # Time-ordered holdout: October is unseen while training uses January–September.
    cutoff = train["date"].max().to_period("M")
    valid_mask = train["date"].dt.to_period("M") == cutoff
    fit_rows, valid_rows = train.loc[~valid_mask], train.loc[valid_mask]
    if fit_rows.empty or valid_rows.empty:
        raise ValueError("need at least two months for chronological validation")
    validation_model = FreightRateModel().fit(fit_rows, fit_rows["posted_rate"])
    scores = regression_metrics(valid_rows["posted_rate"].to_numpy(), validation_model.predict(valid_rows))
    print(f"Chronological validation: train through {fit_rows.date.max():%Y-%m-%d}; holdout {cutoff}")
    print("Metrics (rate units are dollars): " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    # Refit on all labeled data before predicting the November and December rows.
    model = FreightRateModel().fit(train, train["posted_rate"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "freight_rate_model.pkl")

    validation = pd.read_csv(args.data_dir / "validation.csv")
    template = pd.read_csv(args.data_dir / "validation-predictions-template.csv")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("load_id must be unique in validation and the template")
    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("validation IDs do not match template IDs")
    by_id = pd.Series(model.predict(validation), index=validation["load_id"])
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].map(by_id)
    if submission["predicted_rate"].isna().any() or (submission["predicted_rate"] <= 0).any():
        raise ValueError("invalid or missing validation predictions")
    submission.to_csv("validation_predictions.csv", index=False, float_format="%.2f")

    december_path = args.data_dir / "december-chart-inputs.csv"
    december = pd.read_csv(december_path)
    december["predicted_rate"] = model.predict(december)
    december_output = args.output_dir / "december_predictions.csv"
    december.to_csv(december_output, index=False, float_format="%.2f")
    print("Wrote validation_predictions.csv and " + str(december_output))
    print("Next: python score.py --predictions validation_predictions.csv --december-predictions " + str(december_output))


if __name__ == "__main__":
    main()
