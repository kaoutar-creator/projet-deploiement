"""Utility helpers for data loading, model versioning and predictions."""
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from .config import MODELS_DIR, ARTIFACT_PREFIX, MODEL_VERSION_FORMAT, DEFAULT_MODEL_NAME


def load_data(path=None):
    """Load data from Excel using the project's filename if path is None."""
    if path is None:
        path = Path.cwd() / "data" / "Risque_data.xls"
    return pd.read_excel(path)


def save_model_versioned(model, name_prefix=ARTIFACT_PREFIX):
    """Save model with timestamped version and update latest symlink (simple duplicate)."""
    ts = datetime.utcnow().strftime(MODEL_VERSION_FORMAT)
    fname = f"{name_prefix}_{ts}.joblib"
    outpath = MODELS_DIR / fname
    joblib.dump(model, outpath)
    # also save a latest copy
    latest = MODELS_DIR / DEFAULT_MODEL_NAME
    joblib.dump(model, latest)
    return outpath


def load_latest_model():
    latest = MODELS_DIR / DEFAULT_MODEL_NAME
    if latest.exists():
        return joblib.load(latest)
    # try to find any model
    candidates = sorted(MODELS_DIR.glob(f"{ARTIFACT_PREFIX}_*.joblib"), reverse=True)
    if candidates:
        return joblib.load(candidates[0])
    return None


def preprocess_X(X, scaler=None, fit=False):
    """Simple numerical scaler pipeline. Returns scaler and transformed X."""
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return scaler, X_scaled
