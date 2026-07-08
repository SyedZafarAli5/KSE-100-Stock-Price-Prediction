"""
KSE-100 Index & MAPL Stock — Next-Day Close Price Prediction
Compares Linear Regression, Random Forest, and XGBoost using lag/technical
features with a time-based (non-shuffled) train/test split.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)


def load_kse100(path="KSE100-20years.csv"):
    df = pd.read_csv(path)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def load_mapl(path="MAPL.xlsx"):
    df = pd.read_excel(path)
    df = df.rename(columns={"DATE": "Date", "OPEN": "Open", "HIGH": "High",
                             "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def engineer_features(df):
    df = df.copy()
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["vol_10"] = df["Close"].rolling(10).std()
    df["hl_range"] = df["High"] - df["Low"]
    df["vol_chg"] = df["Volume"].pct_change(1)
    df["target"] = df["Close"].shift(-1)  # next-day close
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURES = ["Open", "High", "Low", "Close", "Volume",
            "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5", "Close_lag10",
            "ret_1d", "ma_5", "ma_10", "ma_20", "vol_10", "hl_range", "vol_chg"]


def time_split(df, test_frac=0.15):
    n_test = int(len(df) * test_frac)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test


def run_models(train, test, label):
    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8,
                                                 random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42),
    }

    rows = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    results = pd.DataFrame(rows).sort_values("RMSE")

    # Plot: actual vs predicted (best model by RMSE) over test period
    best_name = results.iloc[0]["Model"]
    plt.figure(figsize=(11, 5))
    plt.plot(test["Date"], y_test.values, label="Actual", linewidth=1.6)
    plt.plot(test["Date"], preds[best_name], label=f"Predicted ({best_name})",
              linewidth=1.2, alpha=0.85)
    plt.title(f"{label}: Actual vs Predicted Close (Test Period) — {best_name}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    fname = f"{OUT}/{label.replace(' ', '_')}_actual_vs_pred.png"
    plt.savefig(fname, dpi=130)
    plt.close()

    # Plot: full history close price
    plt.figure(figsize=(11, 4))
    plt.plot(pd.concat([train["Date"], test["Date"]]),
              pd.concat([train["Close"], test["Close"]]), linewidth=0.8)
    plt.axvline(test["Date"].iloc[0], color="red", linestyle="--", linewidth=1,
                label="Train/Test split")
    plt.title(f"{label}: Full Close Price History")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{label.replace(' ', '_')}_history.png", dpi=130)
    plt.close()

    return results, best_name


def main():
    summary_lines = ["# KSE-100 & MAPL — Model Performance Summary\n"]

    print("Processing KSE-100 Index...")
    kse = engineer_features(load_kse100())
    train, test = time_split(kse)
    print(f"  {len(kse)} rows, {kse['Date'].min().date()} -> {kse['Date'].max().date()}")
    results_kse, best_kse = run_models(train, test, "KSE-100 Index")
    print(results_kse.to_string(index=False))
    summary_lines.append("## KSE-100 Index\n")
    summary_lines.append(results_kse.to_markdown(index=False))
    summary_lines.append(f"\n**Best model: {best_kse}**\n")

    print("\nProcessing MAPL Stock...")
    mapl = engineer_features(load_mapl())
    train_m, test_m = time_split(mapl)
    print(f"  {len(mapl)} rows, {mapl['Date'].min().date()} -> {mapl['Date'].max().date()}")
    results_mapl, best_mapl = run_models(train_m, test_m, "MAPL Stock")
    print(results_mapl.to_string(index=False))
    summary_lines.append("## MAPL Stock\n")
    summary_lines.append(results_mapl.to_markdown(index=False))
    summary_lines.append(f"\n**Best model: {best_mapl}**\n")

    with open(f"{OUT}/results_summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nDone. Plots + results_summary.md saved to {OUT}/")


if __name__ == "__main__":
    main()
