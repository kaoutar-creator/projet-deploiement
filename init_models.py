"""Helper to import existing artifact into models/ as latest_model.joblib on first run."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
art = ROOT / 'artifacts' / 'credit_risk_model.joblib'
dst_dir = ROOT / 'models'
dst_dir.mkdir(exist_ok=True)
dst = dst_dir / 'latest_model.joblib'
if art.exists() and not dst.exists():
    shutil.copy2(art, dst)
    print('Copied artifact to models/latest_model.joblib')
else:
    print('No artifact copied (already present or missing)')
