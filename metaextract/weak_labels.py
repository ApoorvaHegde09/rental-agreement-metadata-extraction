"""Training-time only: turn the CSV ground truth into per-token tags by *aligning* each value with the text
(fuzzy string similarity for names; parse-and-compare for numbers/dates/durations).
This produces the supervision for the model; it is never used at prediction time."""
from difflib import SequenceMatcher
from . import lexicon as lx

TAGS = ["O", "VAL", "START", "END", "DUR", "NOTICE", "P1", "P2"]
MAX_DATE_WIN = 7


def _norm(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _parts(date_str):
    try:
        d, m, y = (int(x) for x in str(date_str).strip().split("."))
        return d, m, y
    except Exception:
        return None


def date_windows(words, target):
    """Minimal windows whose parsed date equals `target` (a datetime.date)."""
    hits = []
    n = len(words)
    for i in range(n):
        for j in range(i, min(n, i + MAX_DATE_WIN)):
            win = words[i:j + 1]
            if not any(ch.isdigit() for w in win for ch in w):
                continue
            if lx.parse_date(win) == target:
                if j > i and lx.parse_date(words[i + 1:j + 1]) == target:   # not minimal on the left
                    break
                hits.append((i, j)); break
    return hits


def name_windows(words, name, thresh=0.85):
    target = _norm(name)
    if not target:
        return []
    nw = max(1, len(name.split()))
    cands = []
    for L in range(max(1, nw - 1), nw + 3):
        for i in range(0, len(words) - L + 1):
            r = SequenceMatcher(None, _norm("".join(words[i:i + L])), target).ratio()
            if r >= thresh:
                cands.append((r, -L, i, i + L - 1))
    cands.sort(reverse=True)
    chosen, used = [], set()
    for r, _, i, j in cands:
        if not used.intersection(range(i, j + 1)):
            chosen.append((i, j)); used.update(range(i, j + 1))
    return chosen


def build_tags(tokens, truth):
    words = [t.text for t in tokens]
    tags = ["O"] * len(words)

    def mark(spans, tag):
        for i, j in spans:
            for k in range(i, j + 1):
                if tags[k] == "O":
                    tags[k] = tag

    # parties
    p1 = name_windows(words, truth.get("party_one") or "")
    mark(p1, "P1")
    p2 = name_windows(words, truth.get("party_two") or "")
    mark(p2, "P2")
    # rent
    if truth.get("value"):
        mark([(i, i) for i, w in enumerate(words) if lx.parse_amount([w]) == int(truth["value"])], "VAL")
    # dates
    start = None
    if truth.get("start"):
        p = _parts(truth["start"])
        if p:
            try:
                from datetime import date
                start = date(p[2], p[1], p[0])
                mark(date_windows(words, start), "START")
            except ValueError:
                pass
    end_parts = _parts(truth.get("end")) if truth.get("end") else None
    if end_parts:
        try:
            from datetime import date
            mark(date_windows(words, date(end_parts[2], end_parts[1], end_parts[0])), "END")
        except ValueError:
            pass            # calendar-invalid label such as 31.11.2009: cannot be located literally
    # lease duration: 'N months' such that start + N months ≈ end (month/year granularity)
    if start and end_parts:
        for k, u_i, n, unit in lx.quantity_windows(words):
            if unit in ("month", "year"):
                e = lx.end_from_duration(start, n, unit)
                if (e.month, e.year) == (end_parts[1], end_parts[2]):
                    mark([(k, u_i)], "DUR")
    # notice period: '<n> month(s)/day(s)' whose day-equivalent equals the label and that sits next to 'notice'
    if truth.get("notice"):
        for k, u_i, n, unit in lx.quantity_windows(words):
            if unit in ("month", "day") and lx.notice_days(n, unit) == int(truth["notice"]):
                ctx = words[max(0, k - 8):u_i + 9]
                if any("notice" in w.lower() for w in ctx):
                    mark([(k, u_i)], "NOTICE")
    return tags


def name_score(words, name):
    """Best fuzzy-match ratio of `name` anywhere in the document (used to sanity-check label↔file assignment)."""
    target = _norm(name)
    if not target:
        return 0.0
    nw = max(1, len(name.split())); best = 0.0
    for L in range(max(1, nw - 1), nw + 3):
        for i in range(0, len(words) - L + 1):
            best = max(best, SequenceMatcher(None, _norm("".join(words[i:i + L])), target).ratio())
    return best
