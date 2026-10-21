"""Predict metadata for every document in a folder (or one file); score against a CSV if given.

    python scripts/predict.py --input data/test --truth data/test.csv --out outputs
    python scripts/predict.py --input some_agreement.png
"""
import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaextract import FIELDS, CSV_COLUMNS
from metaextract.data import find_files, load_truth, stem_of
from metaextract.evaluate import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--model", default="models/extractor.joblib")
    ap.add_argument("--truth"); ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    model = joblib.load(a.model)
    inp = Path(a.input)
    files = find_files(inp) if inp.is_dir() else {stem_of(inp): inp}
    preds = {s: model.predict_file(p) for s, p in files.items()}
    rows = [{"File Name": s, **{CSV_COLUMNS[f]: preds[s].get(f) for f in FIELDS}} for s in preds]
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if a.truth:
        truth = load_truth(a.truth)
        strict, per_doc = score(preds, truth); lenient, _ = score(preds, truth, lenient=True)
        print("\nPer-field recall (strict exact match):\n", strict.to_string())
        print("\nPer-field recall (lenient: case/punctuation-insensitive names):\n", lenient.to_string())
        (out / "test_metrics.json").write_text(json.dumps({"strict": strict.to_dict("index"), "lenient": lenient.to_dict("index"), "per_document": per_doc}, indent=2))


if __name__ == "__main__":
    main()
