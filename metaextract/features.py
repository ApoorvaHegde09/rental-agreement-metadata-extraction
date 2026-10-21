"""Contextual features for every token (word identity, shape, neighbours, directional bags of words, layout)."""
from . import lexicon as lx


def shape(w):
    out = []
    for ch in w:
        c = "X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def doc_features(tokens):
    n = len(tokens)
    low = [lx.clean(t.text) for t in tokens]
    n_lines = max(1, tokens[-1].line + 1) if tokens else 1
    cap = [int(t.text[:1].isupper()) for t in tokens]
    left_run, right_run = [0] * n, [0] * n
    for i in range(n):
        left_run[i] = (left_run[i - 1] + 1) if cap[i] and i > 0 else int(cap[i])
    for i in range(n - 1, -1, -1):
        right_run[i] = (right_run[i + 1] + 1) if cap[i] and i + 1 < n else int(cap[i])
    feats = []
    for i, t in enumerate(tokens):
        w = low[i]
        f = {"bias": 1, "w=" + w: 1, "shape=" + shape(t.text): 1, "pre3=" + w[:3]: 1, "suf3=" + w[-3:]: 1,
             "isdig": int(any(c.isdigit() for c in t.text)),
             "ndig=%d" % min(sum(c.isdigit() for c in t.text), 8): 1, "isup": int(t.text.isupper()),
             "istitle": int(t.text.istitle()), "len=%d" % min(len(w), 12): 1,
             "month": int(w in lx.MONTHS), "numword": int(w in lx.NUM_WORDS), "ordword": int(w in lx.ORD_WORDS),
             "unit=" + str(lx.unit_of(t.text)): 1, "amount": int(lx.parse_amount([t.text]) is not None),
             "pos=%d" % int(10 * i / max(n, 1)): 1, "line=%d" % min(t.line, 12): 1,
             "relline=%d" % int(10 * t.line / n_lines): 1, "col=%d" % min(t.col, 6): 1,
             "eol": int(i + 1 < n and tokens[i + 1].line != t.line) or int(i + 1 == n),
             "bol": int(t.col == 0)}
        f["caprun_l=%d" % min(left_run[i], 5)] = 1
        f["caprun_r=%d" % min(right_run[i], 5)] = 1
        f["endsdot"] = int(t.text.endswith("."))
        f["endscomma"] = int(t.text.rstrip().endswith(","))
        if i > 0:
            f["prev_endsdot"] = int(tokens[i - 1].text.endswith("."))
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            if 0 <= j < n:
                f["w%+d=%s" % (d, low[j])] = 1
                f["s%+d=%s" % (d, shape(tokens[j].text))] = 1
                if abs(d) == 1:
                    f["dig%+d" % d] = int(any(c.isdigit() for c in tokens[j].text))
                    f["month%+d" % d] = int(low[j] in lx.MONTHS)
                    f["unit%+d=%s" % (d, lx.unit_of(tokens[j].text))] = 1
        if i > 0:
            f["bi-=" + low[i - 1] + "|" + w] = 1
        if i + 1 < n:
            f["bi+=" + w + "|" + low[i + 1]] = 1
        for j in range(max(0, i - 12), i):
            f["L:" + low[j]] = 1
        for j in range(i + 1, min(n, i + 13)):
            f["R:" + low[j]] = 1
        feats.append(f)
    return feats
