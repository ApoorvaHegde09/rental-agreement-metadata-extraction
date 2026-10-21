"""Per-field recall exactly as defined in the brief: True = exact match, False = mismatch or nothing extracted."""
import pandas as pd
from . import FIELDS


def _canon(v, key, lenient):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if key in ("party_one", "party_two"):
        s = str(v).strip()
        return "".join(ch.lower() for ch in s if ch.isalnum()) if lenient else s
    return v if key in ("value", "notice") else str(v).strip()


def score(preds, truth, lenient=False):
    """preds/truth: {stem: {field: value}}. Docs whose ground truth for a field is empty are skipped for that field."""
    rows, per_doc = [], {}
    for f in FIELDS:
        hit = tot = 0
        for stem, t in truth.items():
            if stem not in preds or t.get(f) is None:
                continue
            ok = _canon(preds[stem].get(f), f, lenient) == _canon(t[f], f, lenient)
            hit += ok; tot += 1
            per_doc.setdefault(stem, {})[f] = bool(ok)
        rows.append({"field": f, "correct": hit, "total": tot, "recall": round(hit / tot, 3) if tot else None})
    return pd.DataFrame(rows).set_index("field"), per_doc
