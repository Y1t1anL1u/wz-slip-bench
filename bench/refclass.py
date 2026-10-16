"""
refclass.py — 参考类(reference class)定义与条件基率表。
runner.py(lookup / warm2 模板)和 group_stats.py 共用,保证分组口径一致。
"""
import re
import pandas as pd

# ---- 类型:按优先级首个命中(互斥),事故类优先,再工程类,最后通用 ----
TYPE_RULES = [
    ("incident",     r"emergency|pothole|flood|storm|washout|congestion|weather|\bevent\b|crash|accident|disabled|debris"),
    ("mowing/trees", r"\bmow|tree|vegetation|brush|litter"),
    ("culvert",      r"culvert"),
    ("paving",       r"paving|resurfac|milling|asphalt|\bpatch"),
    ("pipe/utility", r"\bpipe|utility|water ?line|sewer|gas ?line|fiber|cable"),
    ("signal/sign",  r"signal|\bsigns?\b|striping|marking"),
    ("guardrail",    r"guardrail|guard rail|barrier"),
    ("bridge",       r"\bbridge"),
    ("ramp",         r"\bramp"),
    ("shoulder",     r"shoulder"),
    ("moving",       r"\bmoving\b|mobile"),
    ("construction", r"construction"),
    ("maintenance",  r"maintenance"),
]
_TYPE_RE = [(k, re.compile(p)) for k, p in TYPE_RULES]

RECUR_RE = re.compile(
    r"nightly|night ?time|overnight|each night|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*(?:-|to|through|thru)\s*(?:mon|tue|wed|thu|fri|sat|sun)|"
    r"weekday|weeknight|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:-|to|through|thru|until)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    r"\bthrough\b|\bthru\b|daily|recurring|intermittent"
)

WINDOW_BINS = [-1e9, 1, 7, 30, 1e9]
WINDOW_LABELS = ["<=1d", "2-7d", "8-30d", ">30d"]


def type_of(desc):
    d = (desc or "").lower()
    for k, rx in _TYPE_RE:
        if rx.search(d):
            return k
    return "other"


def window_bin(first_seen, promised):
    fs = pd.to_datetime(first_seen, errors="coerce", utc=True)
    pe = pd.to_datetime(promised, errors="coerce", utc=True)
    if pd.isna(fs) or pd.isna(pe):
        return "unknown"
    d = (pe - fs).total_seconds() / 86400
    for lo, hi, lab in zip(WINDOW_BINS[:-1], WINDOW_BINS[1:], WINDOW_LABELS):
        if lo < d <= hi:
            return lab
    return "unknown"


def annotate(df):
    """给 DataFrame 加 rc_type / rc_window / rc_recurring / rc_hard 列。"""
    df = df.copy()
    df["rc_type"] = df.input_description.map(type_of)
    fs = pd.to_datetime(df.input_first_seen, errors="coerce", utc=True)
    pe = pd.to_datetime(df.input_promised_end, errors="coerce", utc=True)
    wd = (pe - fs).dt.total_seconds() / 86400
    df["rc_window"] = pd.cut(wd, WINDOW_BINS, labels=WINDOW_LABELS).astype(object).fillna("unknown")
    df["rc_recurring"] = df.input_description.fillna("").str.lower().str.contains(RECUR_RE)
    # 困难层:窗口 >1 天 且 非事故/割草类
    df["rc_hard"] = (df.rc_window != "<=1d") & (~df.rc_type.isin(["incident", "mowing/trees"]))
    return df


def _stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return {"n": int(len(s)),
            "ontime": float((s.abs() <= 1).mean() * 100),
            "gt7": float((s > 7).mean() * 100),
            "gt30": float((s > 30).mean() * 100),
            "p50": float(s.median()), "p75": float(s.quantile(.75)),
            "p90": float(s.quantile(.90)), "p95": float(s.quantile(.95))}


class RefTable:
    """条件基率表:先查 (window, type),n 不足退到 window,再退到全体。"""

    def __init__(self, pool, min_n=30):
        pool = annotate(pool)
        self.min_n = min_n
        self.all = _stats(pool.label_slip_days)
        self.by_wt = {}
        for (w, t), g in pool.groupby(["rc_window", "rc_type"]):
            st = _stats(g.label_slip_days)
            if st and st["n"] >= min_n:
                self.by_wt[(w, t)] = st
        self.by_w = {}
        for w, g in pool.groupby("rc_window"):
            st = _stats(g.label_slip_days)
            if st and st["n"] >= min_n:
                self.by_w[w] = st

    def lookup(self, window, typ):
        if (window, typ) in self.by_wt:
            return self.by_wt[(window, typ)], f"window={window}, type={typ}"
        if window in self.by_w:
            return self.by_w[window], f"window={window} (type={typ} 样本不足,退到窗口层)"
        return self.all, "全体 (退到总体层)"


# ---------------------------------------------------------------------------
# kNN 文本索引(供 conformal.py 复用;runner.py 内有等价实现)
# ---------------------------------------------------------------------------
def _clean_desc(s):
    return re.sub(r"\btims\b", " ", (s or "").lower())


def build_knn(pool, k=30):
    """按窗口桶建 TF-IDF 最近邻索引。返回 {window: (vec, nn, slips, qids)}。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    pa = annotate(pool)
    pa = pa[pa.rc_window != "unknown"].copy()
    pa["_desc"] = pa.input_description.map(_clean_desc)
    out = {}
    for w, g in pa.groupby("rc_window"):
        if len(g) < k + 5:
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(g._desc)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
        out[w] = (vec, nn, pd.to_numeric(g.label_slip_days, errors="coerce").values,
                  g.qid.astype(str).values)
    return out


def knn_neighbors(KNN, window, desc, qid=None, k=30):
    """返回近邻 slip 数组(排除自身 qid)。窗口无索引返回 None。"""
    if window not in KNN:
        return None
    vec, nn, slips, qids = KNN[window]
    dist, idx = nn.kneighbors(vec.transform([_clean_desc(desc)]))
    idx = idx[0]
    if qid is not None:
        idx = [j for j in idx if qids[j] != str(qid)]
    idx = idx[:k]
    s = pd.Series(slips[idx]).dropna()
    return s
