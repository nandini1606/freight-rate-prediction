# Freight Rate Prediction Challenge

## Goal

Estimate `posted_rate` for freight loads from their lane, equipment, distance, weight, date, market index, and quote signal. The supplied labeled file has 48,000 rows from 2025-01-01 through 2025-10-31. The assessment asks for predictions for 12,000 November loads plus a 31-day December scenario.

## Approach

1. Check columns, dates, target positivity, missing values, and identifier uniqueness.
2. Hold out the latest month in the labeled data (October) for validation. This tests a later time period and avoids randomly mixing future rows into the training set.
3. Train a ridge regression model on the log of `posted_rate`. Features include numeric load characteristics, date parts, lane/equipment one-hot indicators, and a few simple distance interactions. Missing numeric features are imputed with medians learned from the training fold; unknown categories are ignored.
4. Report MAE, RMSE, and MAPE for the October holdout. Refit on all labeled rows before final prediction.
5. Fill the provided validation template by `load_id`, predict each December row, and use `score.py` to validate deliverables and create the required chart.

## Project files

```text
freight-rate-prediction/
├── data/
│   ├── train-test.csv                       # provided labeled development data
│   ├── validation.csv                       # provided November loads
│   ├── validation-predictions-template.csv   # required output row order
│   └── december-chart-inputs.csv             # 31 fixed-scenario input rows
├── src/
│   ├── __init__.py
│   └── model.py                              # feature engineering and ridge model
├── requirements.txt
├── run.py                                    # validation + final predictions
├── score.py                                  # supplied assessment scorer
├── validation_predictions.csv                # generated submission file
└── README.md
```

Put the four supplied CSV files in `data/` using the exact filenames shown above. They are ignored by Git because they are assessment inputs; include the code and instructions in the public repository, and only include data if the assessment permits it.

## Setup and run

```bash

python -m pip install -r requirements.txt
python run.py
python score.py --predictions validation_predictions.csv --december-predictions model/december_predictions.csv
```

The run prints chronological validation metrics, writes `validation_predictions.csv`, and saves the fitted model and `december_predictions.csv` in `artifacts/`. It leaves the original December input file unchanged. The scorer writes `scorer_results/candidate_december.png`.

## Output checks and submission

The validation CSV must have exactly `load_id,predicted_rate`, one positive prediction for each of the 12,000 template IDs. The December CSV must retain its original seven columns and one row per day. Submit the validation CSV, accessible repository, a PDF or DOCX describing the validation and split and containing the scorer chart, and a 2–3 minute Loom walkthrough. The supplied scorer validates file structure and creates the chart; it does not compute the final model score.
