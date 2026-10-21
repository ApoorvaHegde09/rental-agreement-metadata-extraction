"""REST API:  POST /extract (multipart field 'file': .docx or image)  ->  JSON with the six fields."""
import os, tempfile
from pathlib import Path
import joblib
from flask import Flask, jsonify, request
from . import CSV_COLUMNS, FIELDS
from .ingest import IMAGE_EXT


def create_app(model_path="models/extractor.joblib"):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    model = joblib.load(model_path)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/extract")
    def extract():
        f = request.files.get("file")
        if f is None:
            return jsonify(error="send the document as multipart field 'file'"), 400
        ext = Path(f.filename or "").suffix.lower()
        if ext != ".docx" and ext not in IMAGE_EXT:
            return jsonify(error=f"unsupported file type '{ext}'. Upload .docx or an image (png/jpg)."), 415
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
        try:
            res = model.predict_file(tmp.name)
        finally:
            os.unlink(tmp.name)
        conf = res.pop("_confidence", {})
        return jsonify(file=f.filename, fields={CSV_COLUMNS[k]: res.get(k) for k in FIELDS}, confidence=conf)

    return app
