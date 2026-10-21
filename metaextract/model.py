"""Learned extractor.
Stage 1  token classifier (logistic regression on contextual features) -> per-token probability for every field.
Stage 2  generic parsers only PROPOSE candidate spans (dates, amounts, 'N months'); the trained model RANKS them.
         Names are decoded directly from the token probabilities; roles (party one / two) also come from the model.
"""
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from . import lexicon as lx
from .features import doc_features
from .ingest import read_document, tokenize
from .weak_labels import TAGS, build_tags

_MAX_WIN = 7


def date_candidates(words):
    """Minimal parseable date windows (i, j, date). The parser proposes, the model decides."""
    out, n = [], len(words)
    for i in range(n):
        w = lx.clean(words[i])
        if not (any(c.isdigit() for c in w) or w in lx.MONTHS or w in lx.ORD_WORDS):
            continue
        for j in range(i, min(n, i + _MAX_WIN)):
            d = lx.parse_date(words[i:j + 1])
            if d:
                out.append((i, j, d)); break
    return out


def analyze(text):
    toks = tokenize(text)
    words = [t.text for t in toks]
    return {"toks": toks, "words": words, "feats": doc_features(toks), "dates": date_candidates(words),
            "qty": lx.quantity_windows(words), "amounts": [(i, lx.parse_amount([w])) for i, w in enumerate(words)]}


class Extractor:
    def __init__(self, C=1.0, tag_weight=4.0, min_df=1, end_explicit=0.3):
        self.C, self.tag_weight, self.min_df, self.end_explicit = C, tag_weight, min_df, end_explicit
        self.vec, self.clf = DictVectorizer(), None

    # ---------------- training ----------------
    def fit(self, prepared):
        """prepared: list of (feature_dicts, tags) — one per document (see `prepare`)."""
        df = Counter()
        for feats, _ in prepared:
            df.update({k for f in feats for k in f})
        self.keep = {k for k, c in df.items() if c >= self.min_df or k == "bias"}
        feats = [{k: v for k, v in f.items() if k in self.keep} for fs, _ in prepared for f in fs]
        y = [t for _, tags in prepared for t in tags]
        X = self.vec.fit_transform(feats)
        cw = {t: (1.0 if t == "O" else self.tag_weight) for t in TAGS}
        self.clf = LogisticRegression(C=self.C, max_iter=500, class_weight=cw).fit(X, y)
        return self

    # ---------------- inference ----------------
    def _probs(self, feats):
        X = self.vec.transform([{k: v for k, v in f.items() if k in self.keep} for f in feats])
        P = self.clf.predict_proba(X)
        return {c: P[:, k] for k, c in enumerate(self.clf.classes_)}

    @staticmethod
    def _name_spans(pn, k=6, self_min_grow=0.1):
        """Seeds = most name-like tokens; grow each span while neighbours (gap <= 1) stay name-like."""
        used, spans = set(), []
        for seed in np.argsort(-pn):
            if len(spans) >= k or pn[seed] < 0.02:
                break
            if seed in used:
                continue
            th = max(self_min_grow, 0.5 * pn[seed])
            i = j = int(seed)
            grew = True
            while grew:
                grew = False
                for step in (-1, 1):
                    for g in (1, 2):
                        nb = (i if step < 0 else j) + step * g
                        if 0 <= nb < len(pn) and nb not in used and pn[nb] >= th:
                            if step < 0:
                                i = nb
                            else:
                                j = nb
                            grew = True
                            break
            used.update(range(i, j + 1))
            spans.append((i, j))
        return spans

    def predict_analysis(self, A):
        words = A["words"]
        if not words or self.clf is None:
            return {}
        P = self._probs(A["feats"])
        out = {k: None for k in ("value", "start", "end", "notice", "party_one", "party_two")}
        conf = {}
        mean = lambda tag, i, j: float(P[tag][i:j + 1].mean())

        cands = [(P["VAL"][i], v) for i, v in A["amounts"] if v]
        if cands:
            s, v = max(cands)
            out["value"], conf["value"] = int(v), float(s)
        starts = [(mean("START", i, j), d) for i, j, d in A["dates"]]
        start = None
        if starts:
            s, start = max(starts, key=lambda x: x[0])
            out["start"], conf["start"] = lx.fmt_date(start), float(s)
        ends = [(mean("END", i, j), d) for i, j, d in A["dates"] if start is None or d >= start]
        durs = []
        if start:
            for i, j, n, u in A["qty"]:
                if u in ("month", "year") and 1 <= n <= (120 if u == "month" else 10):
                    durs.append((mean("DUR", i, j), lx.end_from_duration(start, n, u)))
        best_end = max(ends, key=lambda x: x[0]) if ends else None
        best_dur = max(durs, key=lambda x: x[0]) if durs else None
        if best_end and (best_end[0] >= self.end_explicit or not best_dur):
            out["end"], conf["end"] = lx.fmt_date(best_end[1]), float(best_end[0])
        elif best_dur:
            out["end"], conf["end"] = lx.fmt_date(best_dur[1]), float(best_dur[0])
        nots = [(mean("NOTICE", i, j), lx.notice_days(n, u)) for i, j, n, u in A["qty"]
                if u in ("month", "day") and 1 <= n <= (12 if u == "month" else 365)]
        if nots:
            s, d = max(nots)
            out["notice"], conf["notice"] = int(d), float(s)
        # parties: detect name spans from P1+P2 mass, then assign the two roles jointly
        spans = self._name_spans(P["P1"] + P["P2"])
        if spans:
            sc = [(float(P["P1"][i:j + 1].mean()), float(P["P2"][i:j + 1].mean())) for i, j in spans]
            best = None
            for a in range(len(spans)):
                for b in range(len(spans)):
                    if a != b or len(spans) == 1:
                        tot = sc[a][0] + (sc[b][1] if a != b else 0)
                        if best is None or tot > best[0]:
                            best = (tot, a, b)
            _, a, b = best
            text = lambda i, j: " ".join(words[i:j + 1]).strip(" ,.;:")
            out["party_one"], conf["party_one"] = text(*spans[a]), sc[a][0]
            if a != b:
                out["party_two"], conf["party_two"] = text(*spans[b]), sc[b][1]
        out["_confidence"] = conf
        return out

    def predict_text(self, text):
        return self.predict_analysis(analyze(text))

    def predict_file(self, path):
        return self.predict_text(read_document(path))


def prepare(text, truth):
    """Training example for one document: features + weak token tags aligned from the ground truth."""
    A = analyze(text)
    return A["feats"], build_tags(A["toks"], truth)
