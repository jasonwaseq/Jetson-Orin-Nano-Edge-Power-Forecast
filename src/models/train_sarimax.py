import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    # Cyclical hour-of-day encoding + one-hot day-of-week
    hour = idx.hour.values
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    dow = idx.dayofweek  # Mon=0..Sun=6
    dow_oh = pd.get_dummies(dow, prefix="dow", drop_first=True)

    weekend = (dow >= 5).astype(int)

    X = pd.DataFrame(
        {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "weekend": weekend,
        },
        index=idx,
    )
    X = pd.concat([X, dow_oh.set_index(idx)], axis=1)
    return X.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to OPSD time_series_60min_singleindex.csv")
    ap.add_argument("--target", default="DE_load_actual_entsoe_transparency",
                    help="Column name to forecast (e.g., DE_load_actual_entsoe_transparency)")
    ap.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    ap.add_argument("--test_days", type=int, default=30, help="Holdout period length (days)")
    ap.add_argument("--outdir", default="artifacts", help="Where to save model outputs")
    ap.add_argument("--train_days", type=int, default=180, 
		   help="How many most-recent days to use for training (speeds up SARIMAX a lot)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # OPSD includes both utc_timestamp and cet_cest_timestamp; use utc for consistency.
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.set_index("utc_timestamp").sort_index()

    if args.target not in df.columns:
        raise SystemExit(
            f"Target '{args.target}' not found. Example columns include load_actual_* by country."
        )

    y = df[args.target].astype(float).copy()
    y = y.asfreq("h")
    y = y.interpolate(limit=3).dropna()  # light cleanup
    keep_len = (args.train_days + args.test_days) * 24
    y = y.iloc[-keep_len:]

    # Align features
    X = make_time_features(y.index)
    y, X = y.align(X, join="inner", axis=0)

    # Train/test split
    test_len = args.test_days * 24
    if len(y) <= test_len + 7 * 24:
        raise SystemExit("Not enough data after cleaning for the requested test split.")

    y_train, y_test = y.iloc[:-test_len], y.iloc[-test_len:]
    X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]

    # Strong starting config for hourly load
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 24)

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Rolling one-shot forecast over the test window (fast + simple)
    pred = res.get_prediction(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
    yhat = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)  # 95% interval

    rmse = float(np.sqrt(mean_squared_error(y_test, yhat)))
    mae = float(mean_absolute_error(y_test, yhat))

    # Interval coverage (how often actual is inside the 95% band)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    coverage = float(((y_test >= lower) & (y_test <= upper)).mean())

    metrics = {
        "target": args.target,
        "order": order,
        "seasonal_order": seasonal_order,
        "horizon_hours_for_service": args.horizon,
        "test_days": args.test_days,
        "rmse": rmse,
        "mae": mae,
        "interval_95_coverage": coverage,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # Save artifacts
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # statsmodels results pickle
    res.save(outdir / "sarimax_results.pkl")

    # Save a small sample forecast CSV for sanity-checking
    out = pd.DataFrame(
        {
            "y": y_test,
            "yhat": yhat,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    )
    out.to_csv(outdir / "test_forecast.csv")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
