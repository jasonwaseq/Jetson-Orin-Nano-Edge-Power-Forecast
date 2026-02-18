"""
tests/test_api.py – Unit tests for the power-forecast FastAPI service.

The SARIMAX model is mocked so no .pkl file is required to run the tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers to build a realistic mock model
# ---------------------------------------------------------------------------

def _make_mock_model(n_steps: int = 168):
    """Return a MagicMock that mimics SARIMAXResults well enough for the API."""
    # Build a date range that the model will report as its training history
    dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")

    mock_model = MagicMock()
    mock_model.data.dates = dates
    mock_model.model.exog_names = [
        "hour_sin", "hour_cos", "weekend",
        "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    ]

    def _get_forecast(steps, exog=None):
        future_idx = pd.date_range(
            start=dates[-1] + pd.Timedelta(hours=1),
            periods=steps,
            freq="h",
            tz="UTC",
        )
        forecast_mock = MagicMock()
        forecast_mock.predicted_mean = pd.Series(
            np.random.uniform(40_000, 60_000, size=steps), index=future_idx
        )
        ci = pd.DataFrame(
            {
                "lower y": np.random.uniform(38_000, 40_000, size=steps),
                "upper y": np.random.uniform(60_000, 62_000, size=steps),
            },
            index=future_idx,
        )
        forecast_mock.conf_int.return_value = ci
        return forecast_mock

    mock_model.get_forecast.side_effect = _get_forecast
    return mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with the SARIMAX model mocked out."""
    mock_model = _make_mock_model()

    # Patch SARIMAXResults.load so the lifespan handler doesn't need a real file
    with patch("src.service.api.SARIMAXResults") as mock_cls, \
         patch("src.service.api.settings") as mock_settings:

        # Make MODEL_PATH appear to exist
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_settings.MODEL_PATH = mock_path
        mock_settings.METRICS_PATH.exists.return_value = False

        mock_cls.load.return_value = mock_model

        from src.service.api import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForecastEndpoint:
    def test_default_24_hours(self, client):
        resp = client.get("/forecast?hours=24")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 24
        for point in data:
            assert "timestamp" in point
            assert "prediction" in point
            assert "ci_lower" in point
            assert "ci_upper" in point

    def test_minimum_1_hour(self, client):
        resp = client.get("/forecast?hours=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_maximum_168_hours(self, client):
        resp = client.get("/forecast?hours=168")
        assert resp.status_code == 200
        assert len(resp.json()) == 168

    def test_zero_hours_rejected(self, client):
        resp = client.get("/forecast?hours=0")
        assert resp.status_code == 422  # FastAPI validation error

    def test_over_limit_rejected(self, client):
        resp = client.get("/forecast?hours=169")
        assert resp.status_code == 422  # FastAPI validation error

    def test_prediction_is_numeric(self, client):
        resp = client.get("/forecast?hours=3")
        assert resp.status_code == 200
        for point in resp.json():
            assert isinstance(point["prediction"], float)
            assert isinstance(point["ci_lower"], float)
            assert isinstance(point["ci_upper"], float)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "uptime_seconds" in body
        assert "model_last_timestamp" in body
        assert isinstance(body["uptime_seconds"], float)

    def test_health_metrics_field_present(self, client):
        resp = client.get("/health")
        assert "metrics" in resp.json()
