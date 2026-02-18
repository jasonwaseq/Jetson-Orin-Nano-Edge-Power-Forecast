import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import json
import time

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from .config import settings

logger = logging.getLogger("power-forecast")

# ---------------------------------------------------------------------------
# Global state populated during startup
# ---------------------------------------------------------------------------
_model: SARIMAXResults | None = None
_exog_cols: list[str] = []
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SARIMAX model once at startup; fail fast with a clear message."""
    global _model, _exog_cols
    model_path: Path = settings.MODEL_PATH
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            "Run src/models/train_sarimax.py first to generate it."
        )
    logger.info("Loading SARIMAX model from %s …", model_path)
    _model = SARIMAXResults.load(model_path)
    _exog_cols = list(_model.model.exog_names)
    logger.info("Model loaded. Exog columns: %s", _exog_cols)
    yield
    logger.info("Shutting down power-forecast service.")


app = FastAPI(
    title="Edge Power Forecasting Service",
    description="24-hour electricity load forecasting on Jetson Orin Nano",
    version="0.1",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics() -> dict:
    metrics_path: Path = settings.METRICS_PATH
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    ci_lower: float
    ci_upper: float


def make_future_features(start_ts: pd.Timestamp, horizon: int) -> pd.DataFrame:
    # Use lowercase "h" — "H" is deprecated in pandas ≥ 2.2
    idx = pd.date_range(start=start_ts, periods=horizon, freq="h")

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

    # Align to the exact columns the model was trained with
    X = X.reindex(columns=_exog_cols, fill_value=0.0)

    return X


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/forecast", response_model=List[ForecastPoint])
def forecast(hours: int = Query(24, ge=1, le=168)):
    """
    Forecast future electricity load.

    Returns:
    - timestamp
    - predicted load
    - lower / upper 95% confidence bounds
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    logger.info("Forecast request: hours=%d", hours)
    last_ts = pd.Timestamp(_model.data.dates[-1])
    future_X = make_future_features(last_ts + pd.Timedelta(hours=1), hours)

    pred = _model.get_forecast(steps=hours, exog=future_X)
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
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    metrics = load_metrics()
    last_ts = pd.Timestamp(_model.data.dates[-1]).isoformat()
    uptime_s = time.time() - START_TIME

    return {
        "status": "ok",
        "model_last_timestamp": last_ts,
        "uptime_seconds": round(uptime_s, 2),
        "metrics": metrics,
    }
