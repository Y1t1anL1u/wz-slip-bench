#!/usr/bin/env python3
"""
group_stats.py — 按描述特征分组,看滑移分布是否分得开(零成本,不调 API)

目的:验证"条件基率"是否值得注入 prompt。如果各组的准时比例 / P90 差异明显,
      warm_v2 就按组给基率;如果分不开,这条路不用走。

分组维度(每条记录可同时属于多组,各维度独立统计):
  A. 描述是否含周期排期语言(nightly / Mon-Fri / 7pm-6am / through ...)
  B. 项目类型关键词(bridge / culvert / pipe / paving / resurfacing / ...)
  C. posted 窗口长度(promised - first_seen):≤1d / 2-7d / 8-30d / >30d
  D. A × C 交叉(周期语言 且 短窗口,是 warm v1 最容易推错的那类)

用法:
  python group_stats.py
  python group_stats.py --in C:/wzdx/out/task_c_slip_nc_marked.parquet
"""
import argparse, re
import pandas as pd
import os as _os
DATA = _os.environ.get("WZDX_DIR", "C:/wzdx")  # root holding out/ and cache/; override with env WZDX_DIR


ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=f"{DATA}/out/task_c_slip_nc_marked.parquet")
ap.add_argument("--min-n", type=int, default=30, help="组内样本少于该值不打印(默认30)")
args = ap.parse_args()

df = pd.read_parquet(args.inp)
df = df[(~df.meta_is_incident) & (~df.is_censored)].copy()
df["slip"] = pd.to_numeric(df.label_slip_days, errors="coerce")
df = df.dropna(subset=["slip"])
df["desc"] = df.input_description.fillna("").str.lower()
fs = pd.to_datetime(df.input_first_seen, errors="coerce", utc=True)
pe = pd.to_datetime(df.input_promised_end, errors="coerce", utc=True)
df["window_d"] = (pe - fs).dt.total_seconds() / 86400

# ---- A. 周期排期语言 ----
RECUR = re.compile(
    r"nightly|night ?time|overnight|each night|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*(?:-|to|through|thru)\s*(?:mon|tue|wed|thu|fri|sat|sun)|"
    r"weekday|weeknight|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:-|to|through|thru|until)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    r"\bthrough\b|\bthru\b|daily|recurring|intermittent"
)
df["recurring"] = df.desc.str.contains(RECUR)

# ---- B. 项目类型(与 refclass.py 同口径:按优先级首个命中,互斥) ----
from refclass import type_of, TYPE_RULES
df["rc_type"] = df.input_description.map(type_of)
TYPES = {k: None for k, _ in TYPE_RULES}
TYPES["other"] = None
for k in TYPES:
    df[f"type_{k}"] = df.rc_type == k

# ---- C. 窗口分桶 ----
df["window_bin"] = pd.cut(df.window_d, [-1e9, 1, 7, 30, 1e9], labels=["≤1d", "2-7d", "8-30d", ">30d"])


def stats(s):
    s = s.dropna()
    if len(s) == 0:
        return None
    return {
        "n": len(s),
        "ontime±1d": (s.abs() <= 1).mean() * 100,
        ">7d": (s > 7).mean() * 100,
        ">30d": (s > 30).mean() * 100,
        "P50": s.median(), "P75": s.quantile(.75),
        "P90": s.quantile(.90), "P95": s.quantile(.95),
    }


def row(label, s):
    st = stats(s)
    if st is None or st["n"] < args.min_n:
        return
    print(f"  {label:<28} n={st['n']:>5}  准时±1d {st['ontime±1d']:5.1f}%  >7d {st['>7d']:5.1f}%  "
          f">30d {st['>30d']:5.1f}%  | P50 {st['P50']:+5.1f}  P75 {st['P75']:+5.0f}  "
          f"P90 {st['P90']:+5.0f}  P95 {st['P95']:+5.0f}")


HDR = "-" * 118
print(f"有效 work zone 样本: {len(df)}")
print(f"\n[全体]"); print(HDR); row("all", df.slip)

print(f"\n[A] 周期排期语言"); print(HDR)
row("recurring=True",  df[df.recurring].slip)
row("recurring=False", df[~df.recurring].slip)

print(f"\n[B] 项目类型(互斥,按优先级首个命中)"); print(HDR)
for k in TYPES:
    row(k, df[df[f"type_{k}"]].slip)
none_type = df.rc_type == "other"

print(f"\n[C] posted 窗口 = promised − first_seen"); print(HDR)
for b in ["≤1d", "2-7d", "8-30d", ">30d"]:
    row(f"window {b}", df[df.window_bin == b].slip)

print(f"\n[D] 周期语言 × 窗口"); print(HDR)
for rec in [True, False]:
    for b in ["≤1d", "2-7d", "8-30d", ">30d"]:
        row(f"recurring={rec} window {b}", df[(df.recurring == rec) & (df.window_bin == b)].slip)

print(f"\n[E] 项目类型 × 窗口(困难层视角)"); print(HDR)
for k in TYPES:
    for b in ["2-7d", "8-30d", ">30d"]:
        row(f"{k} × {b}", df[(df[f"type_{k}"]) & (df.window_bin == b)].slip)

print(f"\n[F] 项目类型 × 周期语言(只看 recurring=True)"); print(HDR)
for k in TYPES:
    row(f"{k} & recurring", df[df[f"type_{k}"] & df.recurring].slip)

# ---- 附:未命中类型关键词的描述里,高频词 top 30(用于补关键词) ----
from collections import Counter
STOP = set("the a an and or of to in on at for from with by is are be will this that lane lanes road rd hwy nc sr i us closed closure closures".split())
cnt = Counter()
for d in df.loc[none_type, "desc"]:
    for w in re.findall(r"[a-z]{4,}", d):
        if w not in STOP:
            cnt[w] += 1
print(f"\n[附] type=other 组 n={int(none_type.sum())} 的高频词 top 30:")
print("  " + ", ".join(f"{w}({c})" for w, c in cnt.most_common(30)))

print("\n判断标准:如果某组 P90 明显高于全体 (+18) 且准时比例仍不低,说明"
      "\"点估计留 posted、尾部放 p90\"这套分层是有依据的;各组 P90 都差不多则条件基率无用。")
