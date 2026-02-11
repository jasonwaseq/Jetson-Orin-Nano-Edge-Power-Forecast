# Edge Power Forecasting Service

A real-time electricity load forecasting system deployed on an NVIDIA Jetson Orin Nano.
The service produces 24-hour probabilistic forecasts using a seasonal SARIMAX model and
serves results via a low-latency REST API.

## Features
- Seasonal SARIMAX forecasting with calendar-based exogenous features
- 24-hour rolling forecasts with 95% confidence intervals
- FastAPI service with OpenAPI/Swagger UI
- Health endpoint exposing model metadata and metrics
- Benchmarked sub-30 ms p99 inference latency on Jetson Orin Nano

## Model
- Target: National electricity load (OPSD)
- Order: (1,1,1)
- Seasonal order: (1,1,1,24)
- Exogenous features:
  - Hour-of-day (sin/cos)
  - Day-of-week
  - Weekend indicator

## Performance
| Metric | Value |
|------|------|
| RMSE | 5564 MW |
| MAE | 3872 MW |
| 95% CI Coverage | 1.00 |
| p50 Latency | 25.2 ms |
| p99 Latency | 29.7 ms |

## API
- `GET /forecast?hours=24`
- `GET /health`

## Deployment
Runs fully on-device on NVIDIA Jetson Orin Nano with no cloud dependencies.

