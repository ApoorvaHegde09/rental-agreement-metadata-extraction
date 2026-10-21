"""Small language resources (number words, months, ordinals) + value parsers.
These are *vocabulary / unit-conversion tables*, not extraction rules: they never decide
WHICH span of a document is the answer — the trained model does that."""
import calendar
from datetime import date, datetime, timedelta
from dateutil import parser as dparser

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "ninety": 90})
ORD_WORDS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
ORD_WORDS.update({"twenty-first": 21, "twentyfirst": 21, "thirtieth": 30, "thirty-first": 31})
MONTHS = {m.lower() for m in list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]}
UNITS = ("month", "day", "year", "week")
STRIP = "()[]{}<>.,;:'\"`‘’“”*|!?"


def clean(tok):
    return tok.strip(STRIP + " ").lower()


def number_of(tok):
    """int value of a token like '11', '(11)', '2{Two}', 'eleven' — else None."""
    t = clean(tok)
    if t in NUM_WORDS:
        return NUM_WORDS[t]
    digits = ""
    for ch in t:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def unit_of(tok):
    t = clean(tok)
    for u in UNITS:
        if t.startswith(u):
            return u
    return None


def quantity_windows(tokens):
    """All (start_idx, end_idx, n, unit) 'number … unit' windows (e.g. 'three months', '11 (months)')."""
    out = []
    for u_i, tk in enumerate(tokens):
        u = unit_of(tk if isinstance(tk, str) else tk.text)
        if not u:
            continue
        for k in range(u_i - 1, max(u_i - 4, -1), -1):
            n = number_of(tokens[k] if isinstance(tokens[k], str) else tokens[k].text)
            if n is not None:
                out.append((k, u_i, n, u)); break
    return out


def parse_quantity(words):
    """'three month advance' -> (3, 'month') using the first number→unit pair in the span."""
    q = quantity_windows(list(words))
    return (q[0][2], q[0][3]) if q else None


def _ordinals_to_digits(words):
    out = []
    for w in words:
        c = clean(w)
        out.append(f"{ORD_WORDS[c]}th" if c in ORD_WORDS else w)
    return out


def parse_date(words):
    """Fully-specified (day, month, year) date inside the span, else None."""
    text = " ".join(_ordinals_to_digits(words))
    if not any(ch.isdigit() for ch in text):
        return None
    try:
        a = dparser.parse(text, dayfirst=True, fuzzy=True, default=datetime(2001, 1, 1))
        b = dparser.parse(text, dayfirst=True, fuzzy=True, default=datetime(2002, 2, 2))
    except (ValueError, OverflowError):
        return None
    if a != b or not (1990 <= a.year <= 2035):      # a component was missing / implausible year
        return None
    return a.date()


def parse_amount(words):
    """First numeric run in the span: 'Rs.4,000.00' -> 4000, 'Rs.9000/-(Rupees' -> 9000. OCR junk -> None."""
    for w in words:
        s, i = w, 0
        while i < len(s) and not s[i].isdigit():
            i += 1
        j = i
        while j < len(s) and (s[j].isdigit() or s[j] in ",."):
            j += 1
        num = s[i:j].rstrip(",.")
        if not num:
            continue
        num = num.replace(",", "")
        if num.count(".") > 1:
            continue
        try:
            return int(round(float(num)))
        except ValueError:
            continue
    return None


def add_months(d, n):
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def end_from_duration(start, n, unit):
    if unit == "month":
        return add_months(start, n) - timedelta(days=1)
    if unit == "year":
        return add_months(start, 12 * n) - timedelta(days=1)
    if unit == "week":
        return start + timedelta(weeks=n) - timedelta(days=1)
    return start + timedelta(days=n) - timedelta(days=1)


def notice_days(n, unit):
    return {"day": n, "week": 7 * n, "month": 30 * n, "year": 365 * n}[unit]


def fmt_date(d):
    return d.strftime("%d.%m.%Y")
