# KSE-100 & MAPL — Stock Price Prediction

Predicting **next-day closing price** for the KSE-100 index (Pakistan Stock
Exchange, 20 years of daily data) and MAPL stock, comparing Linear
Regression, Random Forest, and XGBoost.

## Data
- `KSE100-20years.csv` — daily OHLCV, 2004-01-02 to 2024-08-02 (5,094 rows)
- `MAPL.xlsx` — daily OHLCV, 2006-01-02 to 2024-08-08 (4,529 rows)

## Method
- **Features**: lagged closes (1, 2, 3, 5, 10 days), 1-day return, 5/10/20-day
  moving averages, 10-day rolling volatility, high-low range, volume change.
- **Target**: next day's close price.
- **Split**: chronological — last 15% of each series held out as the test
  set (no shuffling, since this is a time series).
- **Models**: `LinearRegression`, `RandomForestRegressor` (300 trees, depth 8),
  `XGBRegressor` (400 trees, depth 4, lr 0.05).

Run with:
```
python3 analysis.py
```

## Results

### KSE-100 Index
| Model             |     RMSE |     MAE |       R2 |
|:------------------|---------:|--------:|---------:|
| Linear Regression |   962.44 |  413.38 | 0.9927   |
| Random Forest     |  8591.43 | 4085.02 | 0.4165   |
| XGBoost           |  8871.70 | 4241.55 | 0.3778   |

### MAPL Stock
| Model             |    RMSE |     MAE |     R2   |
|:------------------|--------:|--------:|---------:|
| Linear Regression |   30.44 |   18.79 | 0.9941   |
| XGBoost           |  382.17 |  192.04 | 0.0756   |
| Random Forest     |  384.11 |  193.91 | 0.0662   |

**Linear Regression wins by a wide margin on both series.**

## Why Linear Regression beats the tree models here

This is a real, worth-noting finding rather than an odd fluke:

- `Close_lag1` (yesterday's close) is by far the strongest predictor of
  tomorrow's close, and the true relationship is close to linear —
  `next_close ≈ last_close + small_drift`. Linear Regression captures that
  directly.
- Random Forest and XGBoost split on feature *thresholds* learned from the
  training range. Since both KSE-100 and MAPL trended strongly upward over
  20 years, the test period's prices are **outside the range the trees ever
  saw during training** — tree models cannot extrapolate beyond the min/max
  they were trained on, so they effectively predict values clustered near
  the training price range, producing large systematic errors.
- This is a classic pitfall when applying tree-based models to trending,
  non-stationary time series with a chronological split. It's not that
  XGBoost/Random Forest are "worse" models in general — they're just the
  wrong tool for extrapolating a trending series without detrending
  (e.g. modeling returns instead of price levels, or differencing the
  series first).

## Outputs
`outputs/` contains, per series:
- `*_history.png` — full price history with train/test split marked
- `*_actual_vs_pred.png` — actual vs. predicted close over the test period
- `results_summary.md` — metrics tables

## Caveat
Predicting next-day *price levels* this well (R²≈0.99) is largely an
artifact of lag-1 autocorrelation, not genuine forecasting skill — a naive
"tomorrow = today" baseline would score similarly. For a real trading or
forecasting signal, the more meaningful (and much harder) target is
next-day **return** or **direction**, not price level.
