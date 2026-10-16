#!/usr/bin/env python3
"""
compare_runs.py — 把多个 seed 的两组结果按 qid 配对后合并,做一次总的配对检验。

用法:
  python compare_runs.py --a "C:/wzdx/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" ^
                         --b "C:/wzdx/out/pilot_nomodel_knn30_v1_hard_wzonly_seed*.parquet" ^
                         --label-a sonnet_warm3 --label-b knn30
输出:合并后 n、点估计胜率、误差差值中位、Wilcoxon/符号检验 p、pinball q50/q90、覆盖率、宽度。
只统计 work zone 且 is_censored=False 且 (若有) is_stale=False 的题。
"""
import argparse, glob, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="A 组结果 parquet 的 glob")
ap.add_argument("--b", required=True, help="B 组结果 parquet 的 glob")
ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
ap.add_argument("--tie", type=float, default=0.5, help="点估计误差差值 |Δ| 小于该天数算平(默认0.5)")
args = ap.parse_args()


def load(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise SystemExit(f"没有匹配到文件: {pattern}")
    df = pd.concat([pd.read_parquet(f).assign(_src=f) for f in fs], ignore_index=True)
    df = df[~df.qid.astype(str).str.startswith("ncnoslip_")]
    if "is_censored" in df: df = df[~df.is_censored.astype(bool)]
    if "is_stale" in df: df = df[~df.is_stale.astype(bool)]
    for c in ["label_actual_end", "input_promised_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    def pdate(s):
        s = s.astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce", utc=True)
    df["pred_dt"] = pdate(df.pred_end)
    df["p90_dt"] = pdate(df.p90_end) if "p90_end" in df else pd.NaT
    df = df.dropna(subset=["pred_dt", "label_actual_end"]).drop_duplicates("qid", keep="first")
    return df, len(fs)


A, na = load(args.a); B, nb = load(args.b)
m = A.merge(B[["qid", "pred_dt", "p90_dt"]], on="qid", suffixes=("_a", "_b"))
print(f"{args.label_a}: {na} 个文件  {args.label_b}: {nb} 个文件  配对 qid n={len(m)}")

D = lambda x, y: (x - y).dt.total_seconds() / 86400
y = D(m.label_actual_end, m.input_promised_end)           # 真实 slip
ea = D(m.pred_dt_a, m.label_actual_end).abs()
eb = D(m.pred_dt_b, m.label_actual_end).abs()
diff = ea - eb
win = (diff < -args.tie).mean(); lose = (diff > args.tie).mean()

def pinball(y, yhat, q):
    d = y - yhat
    return float(np.maximum(q * d, (q - 1) * d).mean())

pa50 = pinball(y, D(m.pred_dt_a, m.input_promised_end), 0.5)
pb50 = pinball(y, D(m.pred_dt_b, m.input_promised_end), 0.5)

# 检验
try:
    from scipy.stats import wilcoxon, binomtest
    nz = diff[diff.abs() > 1e-9]
    p_w = wilcoxon(nz).pvalue if len(nz) >= 10 else float("nan")
    k = int((diff < -args.tie).sum()); n2 = k + int((diff > args.tie).sum())
    p_sign = binomtest(k, n2, 0.5).pvalue if n2 > 0 else float("nan")
except Exception:
    p_w = p_sign = float("nan")

print(f"\n[点估计] {args.label_a} 更准 {win*100:.0f}%  {args.label_b} 更准 {lose*100:.0f}%  平 {(1-win-lose)*100:.0f}%")
print(f"  误差中位: {args.label_a} {ea.median():.1f} 天  {args.label_b} {eb.median():.1f} 天  | 误差 P90: {ea.quantile(.9):.1f} vs {eb.quantile(.9):.1f}")
print(f"  配对差值(A−B)中位 {diff.median():+.2f} 天  均值 {diff.mean():+.2f} 天")
print(f"  Wilcoxon p={p_w:.3f}  符号检验 p={p_sign:.3f}")
print(f"  pinball q50: {args.label_a} {pa50:.2f}  {args.label_b} {pb50:.2f}")

mm = m.dropna(subset=["p90_dt_a", "p90_dt_b"])
if len(mm):
    yy = D(mm.label_actual_end, mm.input_promised_end)
    pa90 = pinball(yy, D(mm.p90_dt_a, mm.input_promised_end), 0.9)
    pb90 = pinball(yy, D(mm.p90_dt_b, mm.input_promised_end), 0.9)
    ca = (mm.label_actual_end <= mm.p90_dt_a).mean(); cb = (mm.label_actual_end <= mm.p90_dt_b).mean()
    wa = D(mm.p90_dt_a, mm.pred_dt_a).median(); wb = D(mm.p90_dt_b, mm.pred_dt_b).median()
    print(f"\n[区间] n={len(mm)}")
    print(f"  pinball q90: {args.label_a} {pa90:.2f}  {args.label_b} {pb90:.2f}")
    print(f"  P90 覆盖率: {args.label_a} {ca*100:.0f}%  {args.label_b} {cb*100:.0f}%  |  宽度中位: {wa:.0f}d vs {wb:.0f}d")

# 按真实 slip 分层看谁在哪一段更好
print(f"\n[按真实 slip 分层] 点估计误差中位 ({args.label_a} / {args.label_b})")
for lo, hi, lab in [(-1e9, 1, "≤1d(准时)"), (1, 7, "1–7d"), (7, 30, "7–30d"), (30, 1e9, ">30d")]:
    s = (y > lo) & (y <= hi)
    if s.sum():
        print(f"  {lab:<10} n={int(s.sum()):>3}  {ea[s].median():6.1f} / {eb[s].median():6.1f}")
