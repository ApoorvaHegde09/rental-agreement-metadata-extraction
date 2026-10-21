"""Train the extractor on data/train and report leave-one-document-out recall on the training set.

    python scripts/train.py --data data --out models
"""
import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaextract import FIELDS, CSV_COLUMNS
from metaextract.data import find_files, load_truth, load_texts, reconcile_labels
from metaextract.evaluate import score
from metaextract.model import Extractor, prepare, analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="models")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(exist_ok=True, parents=True)

    files = find_files(data / "train")
    truth = load_truth(data / "train.csv")
    texts = load_texts(files, out / "train_text_cache.json")            # docx text / OCR, cached
    truth = reconcile_labels(texts, truth)                               # repairs swapped CSV rows
    docs = {s: (texts[s], truth[s]) for s in files if s in truth}
    skipped = [s for s in files if s not in truth]
    print(f"{len(docs)} labelled training documents; unlabelled (skipped): {skipped}")

    prep = {s: prepare(*docs[s]) for s in docs}
    ana = {s: analyze(docs[s][0]) for s in docs}
    # near-duplicate documents (page-1 vs full version of the same contract) are held out together
    groups = {}
    for s in docs:
        groups.setdefault("6683127" if s.startswith("668312") else s, []).append(s)
    preds = {}
    for g, held in groups.items():
        m = Extractor().fit([prep[s] for s in docs if s not in held])
        for s in held:
            preds[s] = m.predict_analysis(ana[s])
    gt = {s: docs[s][1] for s in docs}
    strict, per_doc = score(preds, gt)
    lenient, _ = score(preds, gt, lenient=True)
    print("\nLeave-one-document-out recall on TRAIN (strict exact match):\n", strict.to_string())
    (out / "loo_metrics.json").write_text(json.dumps({"strict": strict.to_dict("index"), "lenient": lenient.to_dict("index")}, indent=2))
    pd.DataFrame([{"File Name": s, **{CSV_COLUMNS[f]: preds[s].get(f) for f in FIELDS}} for s in preds]).to_csv(out / "loo_predictions.csv", index=False)

    final = Extractor().fit(list(prep.values()))
    joblib.dump(final, out / "extractor.joblib")
    print(f"\nmodel saved -> {out/'extractor.joblib'}")


if __name__ == "__main__":
    main()
