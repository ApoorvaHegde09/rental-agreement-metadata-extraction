"""python scripts/serve.py [--port 8000]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaextract.api import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--model", default="models/extractor.joblib")
    a = ap.parse_args()
    create_app(a.model).run(host="0.0.0.0", port=a.port)
