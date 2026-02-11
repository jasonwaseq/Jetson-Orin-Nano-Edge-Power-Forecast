from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import json
import time

from fastapi import FastAPI, Query
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "sarimax_results.pkl"

app = FastAPI(
    title="Edge Power Forecasting Service",
    description="24-hour electricity load forecasting on Jetson Orin Nano",
    version="0.1",
)

# Load model once at startup
model = SARIMAXResults.load(MODEL_PATH)
EXOG_COLS = list(model.model.exog_names)  # exact exog columns used during training

START_TIME = time.time()
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}

class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    ci_lower: float
    ci_upper: float


def make_future_features(start_ts: pd.Timestamp, horizon: int) -> pd.DataFrame:
    idx = pd.date_range(start=start_ts, periods=horizon, freq="H")

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
    X = pd.concat([X, dow_oh.set_index(idx)], axis=1).astype(float)

    X = X.reindex(columns=EXOG_COLS, fill_value=0.0)

    return X


@app.get("/forecast", response_model=List[ForecastPoint])
def forecast(hours: int = Query(24, ge=1, le=168)):
    """
    Forecast future electricity load.

    Returns:
    - timestamp
    - predicted load
    - lower / upper 95% confidence bounds
    """
    last_ts = pd.Timestamp(model.data.dates[-1])
    future_X = make_future_features(last_ts + pd.Timedelta(hours=1), hours)

    pred = model.get_forecast(steps=hours, exog=future_X)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    out = []
    for ts in mean.index:
        out.append(
            ForecastPoint(
                timestamp=ts.isoformat(),
                prediction=float(mean.loc[ts]),
                ci_lower=float(ci.loc[ts].iloc[0]),
                ci_upper=float(ci.loc[ts].iloc[1]),
            )
        )
    return out

@app.get("/health")
def health():
    metrics = load_metrics()
    last_ts = pd.Timestamp(model.data.dates[-1]).isoformat()
    uptime_s = time.time() - START_TIME

    return {
        "status": "ok",
        "model_last_timestamp": last_ts,
        "uptime_seconds": round(uptime_s, 2),
        "metrics": metrics,
    }
