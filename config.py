"""Global configuration for the project."""
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACT_PREFIX = "credit_risk_model"
MODEL_VERSION_FORMAT = "%Y%m%dT%H%M%S"
DEFAULT_MODEL_NAME = "latest_model.joblib"
# Data filename (unchanged)
DATA_FILENAME = "Risque_data.xls"
# Streamlit
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", 8501))
STREAMLIT_HOST = os.environ.get("STREAMLIT_HOST", "0.0.0.0")
