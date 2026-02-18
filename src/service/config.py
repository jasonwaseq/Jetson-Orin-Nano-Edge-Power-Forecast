from pathlib import Path
import logging
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    ARTIFACT_DIR: Path = BASE_DIR / "artifacts"
    MODEL_PATH: Path = ARTIFACT_DIR / "sarimax_results.pkl"
    METRICS_PATH: Path = ARTIFACT_DIR / "metrics.json"
    LOG_LEVEL: str = "INFO"

    model_config = ConfigDict(env_file=".env")

settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("power-forecast")
