#!/usr/bin/env python3
"""
conformal.py — 决策层:是否为一条工区记录发出"预计延期 > τ 天"的提醒,并用 split conformal
给误报率(FPR)一个分布无关的上界 α。

决策与保证
  对每条记录算一个 nonconformity score s(越大越"像会延期")。
  在校准集里只取真实未延期(slip ≤ τ)的记录,取它们 s 的 (1−α) 分位(带有限样本修正)作阈值 q。
  测试时 s > q 才发提醒。保证:P(发提醒 | 真实未延期) ≤ α,即误报率 ≤ α(可交换性假设下)。
  这对应 CommCP 的"只在校准后置信度过线时才发消息、错误率 ≤ α"。

评测
  在测试集上报告:实际 FPR(应 ≤ α)、提醒率、召回(真延期里被提醒的比例)、精确率、沉默率。
  不同 score 在同一 α 下比较召回:召回高 = 该 score 信息量大。

score 来源(--source)
  pool : 在 task_c 困难层全体(约 2000 条)上,用 kNN 与参考类现算 score,不用 LLM。样本大,首选。
  pilot: 从 pilot_*.parquet 读(可含 LLM 的 p90/pred/confidence),同时为这些 qid 现算 kNN/参考类 score,
         便于在同一批题上把 LLM score 与非 LLM score 对比。

用法
  python conformal.py --source pool --tau 7 --alpha 0.1
  python conformal.py --source pilot --files f"{DATA}/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" --tau 7
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from refclass import annotate, RefTable, build_knn, knn_neighbors
import os as _os
DATA = _os.environ.get("WZDX_DIR", "C:/wzdx")  # root holding out/ and cache/; override with env WZDX_DIR


ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["pool", "pilot"], default="pool")
ap.add_argument("--marked", default=f"{DATA}/out/task_c_slip_nc_marked.parquet")
ap.add_argument("--files", default=None, help="pilot 模式:结果 parquet 的 glob")
ap.add_argument("--tau", type=float, default=7.0, help="延期阈值(天):slip > τ 视为'会延期'(默认7)")
ap.add_argument("--alpha", type=float, default=0.10, help="误报率上界(默认0.10)")
ap.add_argument("--alphas", default="0.05,0.1,0.15,0.2,0.3", help="扫描的 α 列表")
ap.add_argument("--knn-k", type=int, default=30)
ap.add_argument("--splits", type=int, default=20, help="随机校准/测试划分重复次数(默认20)")
ap.add_argument("--cal-frac", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hard-only", action="store_true", default=True)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ---------- 数据与 score 计算 ----------
marked = pd.read_parquet(args.marked)
valid = marked[(~marked.meta_is_incident) & (~marked.is_censored)].copy()
valid = annotate(valid)
valid["slip"] = pd.to_numeric(valid.label_slip_days, errors="coerce")
valid = valid.dropna(subset=["slip"])

# 观测窗口过滤(与 runner 一致):承诺/实际都在窗口内
pe = pd.to_datetime(valid.input_promised_end, errors="coerce", utc=True)
ae = pd.to_datetime(valid.label_actual_end, errors="coerce", utc=True)
obs_cut = max(pe.max(), ae.max()) - pd.Timedelta(days=30)
valid = valid[(pe < obs_cut) & (ae < obs_cut)]

if args.source == "pool":
    target = valid[valid.rc_hard].copy() if args.hard_only else valid.copy()
    print(f"[pool] 困难层记录 n={len(target)}(来自有效 work zone {len(valid)})")
else:
    fs = sorted(glob.glob(args.files))
    if not fs:
        raise SystemExit(f"没有匹配到文件: {args.files}")
    pil = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    pil = pil[~pil.qid.astype(str).str.startswith("ncnoslip_")]
    if "is_censored" in pil: pil = pil[~pil.is_censored.astype(bool)]
    pil = pil.drop_duplicates("qid")
    def pdate(s):
        s = s.astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce", utc=True)
    pil["pred_dt"] = pdate(pil.pred_end); pil["p90_dt"] = pdate(pil.p90_end) if "p90_end" in pil else pd.NaT
    pil["promised_dt"] = pd.to_datetime(pil.input_promised_end, errors="coerce", utc=True)
    pil["llm_pred_days"] = (pil.pred_dt - pil.promised_dt).dt.total_seconds() / 86400
    pil["llm_p90_days"] = (pil.p90_dt - pil.promised_dt).dt.total_seconds() / 86400
    pil["llm_conf"] = pd.to_numeric(pil.confidence, errors="coerce")
    target = valid.merge(pil[["qid", "llm_pred_days", "llm_p90_days", "llm_conf"]], on="qid", how="inner")
    print(f"[pilot] {len(fs)} 个文件,配对到有效 work zone 记录 n={len(target)}")

# 参考类与 kNN 都用 valid 全体建(查询时排除自身)
REF = RefTable(valid[["qid", "input_description", "input_first_seen", "input_promised_end", "label_slip_days"]], min_n=30)
KNN = build_knn(valid[["qid", "input_description", "input_first_seen", "input_promised_end", "label_slip_days"]], k=args.knn_k)
print(f"kNN 索引窗口桶: {sorted(KNN)}  参考类 (window,type) 组 {len(REF.by_wt)}")

tau = args.tau
rows = []
for _, r in target.iterrows():
    st, _ = REF.lookup(r.rc_window, r.rc_type)
    nb = knn_neighbors(KNN, r.rc_window, r.input_description, qid=r.qid, k=args.knn_k)
    rec = {
        "qid": r.qid, "slip": r.slip, "y": int(r.slip > tau),
        "rc_window": r.rc_window, "rc_type": r.rc_type,
        # 非 LLM scores
        "s_rc_gt7":  st["gt7"] if tau <= 7 else st["gt30"],      # 参考类里 slip>τ 的比例
        "s_rc_p90":  st["p90"],                                     # 参考类 P90 滑移
        "s_knn_frac": float((nb > tau).mean()) if nb is not None and len(nb) else np.nan,  # 近邻里 slip>τ 比例
        "s_knn_p90":  float(nb.quantile(.9)) if nb is not None and len(nb) else np.nan,
        "s_knn_med":  float(nb.median()) if nb is not None and len(nb) else np.nan,
        "s_window_days": (pd.to_datetime(r.input_promised_end, utc=True) - pd.to_datetime(r.input_first_seen, utc=True)).total_seconds() / 86400,
    }
    if args.source == "pilot":
        rec["s_llm_pred"] = r.llm_pred_days
        rec["s_llm_p90"] = r.llm_p90_days
        rec["s_llm_1mconf"] = 100 - r.llm_conf if pd.notna(r.llm_conf) else np.nan
    rows.append(rec)
D = pd.DataFrame(rows)

# ---------- 合成 score:交叉拟合 logistic(训练/预测严格不重叠,避免泄漏进校准) ----------
def add_combined_score(D, feats, name="s_combo", n_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = D[feats].copy()
    ok = X.notna().all(axis=1)
    D[name] = np.nan
    if ok.sum() < 100:
        return D
    Xo, yo = X[ok].values, D.loc[ok, "y"].values
    # 类型 one-hot(只对样本量够的类型)
    typ = pd.get_dummies(D.loc[ok, "rc_type"], prefix="t").astype(float)
    typ = typ.loc[:, typ.sum() >= 30]
    Xo = np.hstack([Xo, typ.values])
    fold = rng.integers(0, n_folds, size=len(Xo))
    pred = np.full(len(Xo), np.nan)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or len(np.unique(yo[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xo[tr], yo[tr])
        pred[te] = clf.predict_proba(Xo[te])[:, 1]
    D.loc[ok, name] = pred
    # 打印全量拟合系数供解释(不用于预测)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xo, yo)
    coefs = clf[-1].coef_[0]
    names = feats + list(typ.columns)
    top = sorted(zip(names, coefs), key=lambda t: -abs(t[1]))[:8]
    print(f"[{name}] 交叉拟合 logistic,特征 {len(names)} 个;标准化系数绝对值 top: "
          + ", ".join(f"{n}={c:+.2f}" for n, c in top))
    return D

BASE_FEATS = ["s_knn_med", "s_knn_frac", "s_knn_p90", "s_rc_gt7", "s_rc_p90", "s_window_days"]
D = add_combined_score(D, BASE_FEATS, "s_combo_knn")
if args.source == "pilot":
    D = add_combined_score(D, BASE_FEATS + ["s_llm_pred", "s_llm_p90"], "s_combo_knn_llm")

score_cols = [c for c in D.columns if c.startswith("s_")]
print(f"\n样本 n={len(D)}  真延期(slip>{tau:g}d)比例 {D.y.mean()*100:.1f}%  (正例 {int(D.y.sum())})")


# ---------- split conformal:控制 FPR ≤ α ----------
def conformal_threshold(s_neg, alpha):
    """在真实未延期的校准 score 上取带修正的 (1−α) 分位。"""
    s = np.sort(np.asarray(s_neg, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf          # 校准样本不足以保证 α:永不提醒
    return s[k - 1]


def evaluate(D, col, alpha, n_splits, cal_frac):
    d = D.dropna(subset=[col])
    out = []
    for _ in range(n_splits):
        idx = rng.permutation(len(d))
        ncal = int(len(d) * cal_frac)
        cal, test = d.iloc[idx[:ncal]], d.iloc[idx[ncal:]]
        q = conformal_threshold(cal.loc[cal.y == 0, col], alpha)
        alert = test[col] > q
        neg, pos = test.y == 0, test.y == 1
        fpr = alert[neg].mean() if neg.any() else np.nan
        tpr = alert[pos].mean() if pos.any() else np.nan
        prec = (test.y[alert] == 1).mean() if alert.any() else np.nan
        out.append(dict(q=q, alert_rate=alert.mean(), fpr=fpr, tpr=tpr, prec=prec))
    o = pd.DataFrame(out)
    return dict(n=len(d), q=o.q.median(), alert=o.alert_rate.mean(), fpr=o.fpr.mean(),
                fpr_max=o.fpr.max(), tpr=o.tpr.mean(), prec=o.prec.mean())


print(f"\n== α={args.alpha:g},τ={tau:g}d,{args.splits} 次随机划分(校准 {args.cal_frac:.0%})平均 ==")
print(f"{'score':<16}{'n':>6}{'阈值':>10}{'提醒率':>8}{'FPR':>8}{'FPR最大':>9}{'召回':>8}{'精确率':>8}")
res_main = {}
for c in score_cols:
    r = evaluate(D, c, args.alpha, args.splits, args.cal_frac)
    res_main[c] = r
    print(f"{c:<16}{r['n']:>6}{r['q']:>10.2f}{r['alert']*100:>7.0f}%{r['fpr']*100:>7.1f}%{r['fpr_max']*100:>8.1f}%"
          f"{r['tpr']*100:>7.0f}%{r['prec']*100:>7.0f}%")
print("  读法:FPR 应 ≤ α(保证成立);同一 α 下召回越高 = 该 score 信息量越大;提醒率越低 = 越'沉默'。")

# ---------- α 扫描:召回 vs α ----------
alphas = [float(a) for a in args.alphas.split(",")]
print(f"\n== 召回率 vs α(τ={tau:g}d)==")
print(f"{'score':<16}" + "".join(f"{f'α={a:g}':>10}" for a in alphas))
for c in score_cols:
    vals = []
    for a in alphas:
        r = evaluate(D, c, a, args.splits, args.cal_frac)
        vals.append(f"{r['tpr']*100:>6.0f}%/{r['fpr']*100:.0f}")
    print(f"{c:<16}" + "".join(f"{v:>10}" for v in vals))
print("  每格 = 召回%/实际FPR%")

# ---------- 基线对照:不校准的固定规则 ----------
print(f"\n== 对照:不做 conformal 的固定规则 ==")
for name, alert in [("posted 窗口>1d 全部提醒", D.s_window_days > 1),
                    ("参考类 gt7≥30%", D.s_rc_gt7 >= 30),
                    ("kNN frac≥0.3", D.s_knn_frac >= 0.3)]:
    neg, pos = D.y == 0, D.y == 1
    print(f"  {name:<24} 提醒率 {alert.mean()*100:5.0f}%  FPR {alert[neg].mean()*100:5.1f}%  召回 {alert[pos].mean()*100:5.0f}%")

out = f"{DATA}/out/conformal_{args.source}_tau{tau:g}.parquet"
D.to_parquet(out, index=False)
print(f"\nscore 表已写: {out}")
