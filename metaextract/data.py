"""Dataset helpers: file discovery, ground-truth loading, cached text extraction, label↔file sanity check."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from . import CSV_COLUMNS
from .ingest import read_document, tokenize
from .weak_labels import name_score

DOC_EXT = {".docx", ".png", ".jpg", ".jpeg"}


def stem_of(path):
    name = Path(path).name
    for ext in sorted(DOC_EXT, key=len, reverse=True):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name[:-4] if name.lower().endswith(".pdf") else name


def find_files(folder):
    return {stem_of(p): p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in DOC_EXT}


def load_truth(csv_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    truth = {}
    for _, r in df.iterrows():
        t = {}
        for key, col in CSV_COLUMNS.items():
            v = str(r[col]).strip()
            t[key] = None if v == "" else (int(float(v)) if key in ("value", "notice") else v)
        truth[stem_of(r["File Name"])] = t
    return truth


def load_texts(files, cache_file=None):
    cache = json.loads(Path(cache_file).read_text()) if cache_file and Path(cache_file).exists() else {}
    for stem, p in files.items():
        if stem not in cache:
            cache[stem] = read_document(p)
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(json.dumps(cache))
    return {s: cache[s] for s in files}


def reconcile_labels(texts, truth, margin=0.25, log=print):
    """Check every CSV row really describes the file it is attached to (party names must appear in that file).
    Rows that fit another file clearly better are re-assigned (optimal one-to-one assignment).
    Needed because train.csv has two rows swapped against their images."""
    stems = [s for s in truth if s in texts]
    words = {s: [t.text for t in tokenize(texts[s])] for s in texts}
    S = np.array([[name_score(words[d], truth[s]["party_one"] or "") + name_score(words[d], truth[s]["party_two"] or "")
                   for d in stems] for s in stems])
    rows, cols = linear_sum_assignment(-S)
    fixed = dict(truth)
    for r, c in zip(rows, cols):
        if r != c and S[r, c] - S[r, r] > margin:
            log(f"[labels] CSV row '{stems[r]}' matches file '{stems[c]}' better "
                f"({S[r, c]:.2f} vs {S[r, r]:.2f}) -> re-assigned")
            fixed[stems[c]] = truth[stems[r]]
    return fixed
