import anthropic, pandas as pd, glob, json, hashlib, os, re, time, argparse
from refclass import RefTable, annotate
import os as _os
DATA = _os.environ.get("WZDX_DIR", "C:/wzdx")  # root holding out/ and cache/; override with env WZDX_DIR


CACHE = f"{DATA}/cache"
os.makedirs(CACHE, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--no-cache", action="store_true",
                help="忽略磁盘缓存,强制重新调用模型(结果仍会写回缓存)")
ap.add_argument("--marked", default=f"{DATA}/out/task_c_slip_nc_marked.parquet",
                help="带 is_censored 列的 task_c 文件")
ap.add_argument("--seed", type=int, default=1,
                help="抽样种子:同时驱动 work zone 抽样、no-slip 抽样和最终 shuffle;"
                     "输出文件名带 seed(默认1)")
ap.add_argument("--top-err", type=int, default=5,
                help="结尾打印有效样本中误差最大的前 N 条(默认5,0 关闭)")
ap.add_argument("--window-margin-days", type=int, default=30,
                help="观测窗口余量:承诺结束日必须早于 obs_end 减去该天数,才认为"
                     "结果已被观测到(默认30)")
ap.add_argument("--noslip-exit-tol-days", type=int, default=7,
                help="no-slip 样本要求记录在承诺日 ±该天数内离开 feed,否则视为僵尸记录(默认7)")
ap.add_argument("--stale-slip-days", type=float, default=30,
                help="僵尸记录判定:promised-first_seen < 1 天 且 slip > 该值(默认30)")
ap.add_argument("--template", choices=["cold", "warm", "warm2", "warm3", "lookup", "knn"], default="cold",
                help="cold: 零信息;warm: 总体基率+文字规律(v1);"
                     "warm2: 注入该记录参考类的条件基率;lookup: 纯查表 baseline;"
                     "knn: 描述 TF-IDF 最近邻 baseline(同窗口桶内),不调模型;"
                     "warm3: 检索增强——把 kNN 近邻及其真实 slip 喂给模型")
ap.add_argument("--temperature", type=float, default=0.0, help="采样温度(默认0,确定性)")
ap.add_argument("--n-wz", type=int, default=25, help="work zone 抽题数(默认25)")
ap.add_argument("--n-noslip", type=int, default=25, help="no-slip 抽题数(默认25;--no-noslip 时为0)")
ap.add_argument("--no-noslip", action="store_true", help="不含 no-slip 半(目标人群=task_c)")
ap.add_argument("--compare-to", default=None,
                help="另一次运行的结果 parquet 路径;按 qid 配对,报告胜率与 pinball loss 差")
ap.add_argument("--knn-show", type=int, default=8, help="warm3 喂给模型的近邻条数(默认8)")
ap.add_argument("--model", default="claude-haiku-4-5",
                help="模型名(默认 claude-haiku-4-5;可换 claude-sonnet-4-6 等)")
ap.add_argument("--knn-k", type=int, default=30, help="knn 模板的近邻数(默认30)")
ap.add_argument("--stratify", action="store_true",
                help="work zone 那半只从困难层抽(窗口>1天 且 非事故/割草类);默认随机")
args = ap.parse_args()
SEED = args.seed
MODEL = args.model
if args.no_noslip:
    args.n_noslip = 0
TEMPLATE_VER = {"cold": "c_cold_v1", "warm": "c_warm_v1",
                "warm2": "c_warm_v2", "lookup": "lookup_v1",
                "knn": f"knn{args.knn_k}_v1",
                "warm3": f"c_warm_v3_k{args.knn_show}"}[args.template]

# ---------- 读带删失标记的 task_c ----------
if os.path.exists(args.marked):
    slip_all = pd.read_parquet(args.marked)
    if "is_censored" not in slip_all.columns:
        print(f"[警告] {args.marked} 缺少 is_censored 列,全部按有效处理")
        slip_all["is_censored"] = False
else:
    print(f"[警告] 找不到 {args.marked},回退到原始文件,is_censored 全置 False")
    slip_all = pd.read_parquet(f"{DATA}/out/task_c_slip_nc.parquet")
    slip_all["is_censored"] = False

# ---------- 读 NC entities,推断观测窗口 ----------
ents = pd.concat(pd.read_parquet(f) for f in
                 glob.glob(f"{DATA}/out/north_carolina/entities_*.parquet"))

# 自动识别"最后观测时间"列名
LAST_SEEN_CANDIDATES = ["last_seen", "last_observed", "last_seen_at", "last_observed_at", "last_update"]
last_seen_col = next((c for c in LAST_SEEN_CANDIDATES if c in ents.columns), None)
if last_seen_col is None:
    print(f"[警告] entities 里找不到最后观测时间列(尝试了 {LAST_SEEN_CANDIDATES}),"
          f"将退化为用 first_seen.max() 当观测窗口结束,且跳过'记录在承诺日附近离开 feed'的检查。")
    print(f"       entities 现有列: {list(ents.columns)}")

for c in ["first_seen", "first_end_date", "last_end_date"] + ([last_seen_col] if last_seen_col else []):
    ents[c] = pd.to_datetime(ents[c], errors="coerce", utc=True)

# entities_*.parquet 可能按快照分文件,同一 entity_id 重复出现;按最后观测保留一条
n_raw = len(ents)
if last_seen_col:
    ents = ents.sort_values(last_seen_col).drop_duplicates("entity_id", keep="last")
else:
    ents = ents.drop_duplicates("entity_id", keep="last")
print(f"entities 去重: {n_raw} 行 -> {len(ents)} 个唯一 entity_id")

obs_end = (ents[last_seen_col].max() if last_seen_col else ents.first_seen.max())
window_cut = obs_end - pd.Timedelta(days=args.window_margin_days)
print(f"观测窗口结束 obs_end={str(obs_end)[:19]}  "
      f"承诺日需早于 {str(window_cut)[:19]} (余量 {args.window_margin_days} 天)")

# ---------- work zone 那半:剔除承诺日超出窗口的(结果未被观测) ----------
slip_all["_promised_dt"] = pd.to_datetime(slip_all.input_promised_end, errors="coerce", utc=True)
slip_all["_actual_dt"] = pd.to_datetime(slip_all.label_actual_end, errors="coerce", utc=True)
wz_pool = slip_all[~slip_all.meta_is_incident]
n_wz_before = len(wz_pool)
wz_pool = wz_pool[(wz_pool._promised_dt < window_cut) & (wz_pool._actual_dt < window_cut)]
print(f"work zone 池: {n_wz_before} -> {len(wz_pool)} (剔除承诺/实际结束超出窗口的 {n_wz_before-len(wz_pool)} 条)")
wz_pool = annotate(wz_pool)
n_hard = int(wz_pool.rc_hard.sum())
if args.stratify:
    print(f"分层抽样: 困难层 {n_hard} 条 / 全池 {len(wz_pool)} 条,work zone {args.n_wz} 题全部从困难层抽")
    slip = wz_pool[wz_pool.rc_hard].sample(args.n_wz, random_state=SEED)
else:
    print(f"随机抽样 (困难层占池子 {n_hard/len(wz_pool)*100:.0f}%);加 --stratify 可只抽困难层")
    slip = wz_pool.sample(args.n_wz, random_state=SEED)

# ---------- no-slip 那半:承诺日在窗口内 + 记录确实在承诺日附近离开 feed ----------
base = ((ents.first_end_date == ents.last_end_date)
        & ents.first_end_date.notna()
        & (~ents.is_incident)
        & (ents.description.fillna("").str.len() >= 10))
n_base = int(base.sum())
in_window = ents.first_end_date < window_cut
cond = base & in_window
n_after_window = int(cond.sum())
if last_seen_col:
    exit_near = (ents[last_seen_col] - ents.first_end_date).abs() <= pd.Timedelta(days=args.noslip_exit_tol_days)
    cond = cond & exit_near
print(f"no-slip 池: 原条件 {n_base} -> 窗口内 {n_after_window} -> "
      f"承诺日附近离开 feed {int(cond.sum())}"
      + ("" if last_seen_col else "  (无 last_seen 列,跳过离开 feed 检查)"))
if args.n_noslip > 0 and int(cond.sum()) < args.n_noslip:
    raise SystemExit(f"[错误] no-slip 合格样本仅 {int(cond.sum())} 条,不足 {args.n_noslip},"
                     f"请放宽 --noslip-exit-tol-days 或 --window-margin-days")
ns = ents[cond].sample(args.n_noslip, random_state=SEED) if args.n_noslip > 0 else ents[cond].iloc[0:0]

noslip = pd.DataFrame({
    "qid": "ncnoslip_" + ns.entity_id.str.replace("|", "_"),
    "input_description": ns.description, "input_road": ns.road,
    "input_direction": ns.direction, "input_first_seen": ns.first_seen,
    "input_promised_end": ns.first_end_date,
    "label_actual_end": ns.last_end_date, "label_slip_days": 0.0,
    "is_censored": False,
})

cols = list(noslip.columns)
exam = pd.concat([slip[cols], noslip]).sample(frac=1, random_state=SEED + 1000)
exam = annotate(exam)

# ---- 条件基率表:用有效 work zone 样本、排除本次考卷里的 qid(防泄漏) ----
_pool = slip_all[(~slip_all.meta_is_incident) & (~slip_all.is_censored)
                 & (~slip_all.qid.isin(exam.qid))]
REF = RefTable(_pool, min_n=30)

# ---- kNN 文本索引(同窗口桶内,TF-IDF 余弦最近邻)----
KNN = None
if args.template in ("knn", "warm3"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    _pa = annotate(_pool)
    _pa = _pa[_pa.rc_window != "unknown"].copy()
    _pa["_desc"] = _pa.input_description.fillna("").str.lower().str.replace(r"\btims\b", " ", regex=True)
    KNN = {}
    for w, g in _pa.groupby("rc_window"):
        if len(g) < args.knn_k + 5:
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(g._desc)
        nn = NearestNeighbors(n_neighbors=args.knn_k, metric="cosine").fit(X)
        KNN[w] = (vec, nn, pd.to_numeric(g.label_slip_days, errors="coerce").values,
                  g.input_description.fillna("").values,
                  pd.to_datetime(g.input_promised_end, errors="coerce", utc=True).values,
                  pd.to_datetime(g.input_first_seen, errors="coerce", utc=True).values)
    print(f"kNN 索引: k={args.knn_k},窗口桶 {sorted(KNN)}")
print(f"参考类表: 基于 {len(_pool)} 条有效 work zone(已排除考卷 qid),"
      f"(window,type) 组 {len(REF.by_wt)} 个,window 组 {len(REF.by_w)} 个")
print(f"组卷 (seed={SEED}, T={args.temperature:g}): {len(exam)} 题 (work zone {len(slip)} + no-slip {len(noslip)});"
      f"其中删失 {int(exam.is_censored.sum())} 题")

PROMPT_COLD = """You are a traffic management center analyst reviewing a WZDx work zone feed.

Record:
- Road: {road} ({direction})
- Description: {desc}
- First observed: {first_seen}
- Currently posted end date: {promised}

Predict the ACTUAL end date of this work zone. The posted date may be accurate, or the work may end earlier or later. Respond with ONLY this JSON:
{{"predicted_end_date": "YYYY-MM-DD", "confidence_0_100": <int>, "reasoning": "<one short sentence>"}}"""

# ---- warm 模板的基率:从 marked 文件的有效 work zone 样本里自动算,不手工写死 ----
_bs = slip_all[(~slip_all.meta_is_incident) & (~slip_all.is_censored)].label_slip_days
_bs = pd.to_numeric(_bs, errors="coerce").dropna()
BASE_RATES = {
    "n": int(len(_bs)),
    "pct_late_7d": float((_bs > 7).mean() * 100),
    "pct_late_30d": float((_bs > 30).mean() * 100),
    "pct_early_1d": float((_bs < -1).mean() * 100),
    "median": float(_bs.median()),
    "p75": float(_bs.quantile(.75)),
    "p90": float(_bs.quantile(.90)),
    "p95": float(_bs.quantile(.95)),
}
if args.template == "warm":
    print("warm 模板基率(来自 marked 文件有效 work zone 样本):"
          f" n={BASE_RATES['n']}  >7d {BASE_RATES['pct_late_7d']:.0f}%  >30d {BASE_RATES['pct_late_30d']:.0f}%"
          f"  提前>1d {BASE_RATES['pct_early_1d']:.0f}%  中位 {BASE_RATES['median']:+.1f}"
          f"  P75 {BASE_RATES['p75']:+.0f}  P90 {BASE_RATES['p90']:+.0f}  P95 {BASE_RATES['p95']:+.0f} 天")

PROMPT_WARM = """You are a traffic management center analyst reviewing a WZDx work zone feed. Your job is to predict when this work zone will ACTUALLY end, which is often not the posted date.

Empirical base rates from this same feed (North Carolina, {n} historical work zones with an observed end):
- Only a minority end on the posted date. {pct_late_7d:.0f}% end more than 7 days AFTER the posted date; {pct_late_30d:.0f}% end more than 30 days after.
- Slip is heavy-tailed: median slip is {median:+.1f} days, but P75 is {p75:+.0f} days, P90 is {p90:+.0f} days, and P95 is {p95:+.0f} days.
- Early completion is rare ({pct_early_1d:.0f}% end more than 1 day early).

Important patterns in this feed:
- Records describing a RECURRING schedule (e.g. "nightly 7pm-6am", "Mon-Fri 9am-3pm", "through Wednesday") are frequently renewed again and again. A precise short schedule in the description is NOT evidence of a short job; these are the records most likely to run for weeks or months past the posted date.
- Posted end dates only a few days after first observation usually reflect the current permit window, not the true project end.
- Bridge, culvert, pipe, paving and multi-phase projects almost always exceed the posted date.
- The posted date is most trustworthy for genuinely one-off events (a single-night incident cleanup, a one-day utility job) with no recurring language.

Record:
- Road: {road} ({direction})
- Description: {desc}
- First observed: {first_seen}
- Currently posted end date: {promised}

Give your best single estimate of the actual end date, AND a conservative P90 date such that you believe there is a 90% chance the work zone has ended by then. Respond with ONLY this JSON:
{{"predicted_end_date": "YYYY-MM-DD", "p90_end_date": "YYYY-MM-DD", "confidence_0_100": <int>, "reasoning": "<one short sentence>"}}"""

PROMPT_WARM2 = """You are a traffic management center analyst reviewing a WZDx work zone feed. Your job is to predict when this work zone will ACTUALLY end.

We have already classified this record into a reference class using two features that dominate slip in this feed: how far the posted end date is from first observation, and the type of work. Here are the empirical outcomes for records in the same class (North Carolina, {rc_n} historical records):
- Reference class: {rc_label}
- {rc_ontime:.0f}% ended within 1 day of the posted date.
- {rc_gt7:.0f}% ended more than 7 days late; {rc_gt30:.0f}% more than 30 days late.
- Slip distribution: median {rc_p50:+.1f} days, P75 {rc_p75:+.0f}, P90 {rc_p90:+.0f}, P95 {rc_p95:+.0f}.

How to use this:
- Your single best estimate should stay close to posted + median slip of the class unless the description gives a concrete reason to move it. Do NOT shift the point estimate later just because the schedule sounds recurring; that pattern is already priced into the class statistics.
- Express tail risk through the P90 date, not the point estimate. Your P90 should be at least posted + the class P90 unless the description clearly indicates a one-off event.
- Precise schedules, day-of-week lists, and time windows in the description are common in this feed and are NOT by themselves evidence of a long project.

Record:
- Road: {road} ({direction})
- Description: {desc}
- First observed: {first_seen}
- Currently posted end date: {promised}

Respond with ONLY this JSON:
{{"predicted_end_date": "YYYY-MM-DD", "p90_end_date": "YYYY-MM-DD", "confidence_0_100": <int>, "reasoning": "<one short sentence>"}}"""

PROMPT_WARM3 = """You are a traffic management center analyst reviewing a WZDx work zone feed. Your job is to predict when this work zone will ACTUALLY end.

Below are the {k} most similar historical records from the same feed (same posted-window bucket: {rc_window}), retrieved by text similarity, each with what actually happened. "slip" is actual end minus posted end, in days.

{neighbors}

Summary of these neighbors: {nb_ontime:.0f}% ended within 1 day of posted; slip median {nb_p50:+.0f}d, P90 {nb_p90:+.0f}d.

Now the record to predict:
- Road: {road} ({direction})
- Description: {desc}
- First observed: {first_seen}
- Currently posted end date: {promised}

Use the neighbors as evidence, but read this record's description carefully: if it indicates a longer-term project (structural rehabilitation, facility closure, multi-phase work, "until further notice", explicit later dates), you may move your estimate well beyond the neighbor median; if it indicates a genuinely one-off short task, you may stay near posted. Give a best estimate and a P90 date (90% chance the work zone has ended by then). Respond with ONLY this JSON:
{{"predicted_end_date": "YYYY-MM-DD", "p90_end_date": "YYYY-MM-DD", "confidence_0_100": <int>, "reasoning": "<one short sentence>"}}"""

PROMPT = {"cold": PROMPT_COLD, "warm": PROMPT_WARM, "warm2": PROMPT_WARM2, "warm3": PROMPT_WARM3,
          "lookup": None, "knn": None}[args.template]

def neighbors_for(r, k_show):
    """返回 (近邻文本块, 近邻 slip Series);窗口桶无索引时返回 (None, None)。"""
    if KNN is None or r.rc_window not in KNN:
        return None, None
    vec, nn, slips, descs, pes, fss = KNN[r.rc_window]
    import re as _re
    d = _re.sub(r"\btims\b", " ", (r.input_description or "").lower())
    dist, idx = nn.kneighbors(vec.transform([d]))
    idx = idx[0]
    s_all = pd.Series(slips[idx]).dropna()
    lines = []
    for j in idx[:k_show]:
        pe = pd.Timestamp(pes[j]); fs = pd.Timestamp(fss[j])
        win = (pe - fs).days if (pd.notna(pe) and pd.notna(fs)) else "?"
        desc = str(descs[j]).replace("\n", " ")[:160]
        sl = slips[j]
        lines.append(f"- posted window {win}d, slip {sl:+.0f}d: \"{desc}\"")
    return "\n".join(lines), s_all

def ref_for(r):
    st, label = REF.lookup(r.rc_window, r.rc_type)
    return st, label

def build_prompt(r):
    kw = dict(road=r.input_road, direction=r.input_direction,
              desc=r.input_description, first_seen=r.input_first_seen,
              promised=r.input_promised_end)
    if args.template == "warm":
        kw.update(BASE_RATES)
    elif args.template == "warm2":
        st, label = ref_for(r)
        kw.update(rc_label=label, rc_n=st["n"], rc_ontime=st["ontime"], rc_gt7=st["gt7"],
                  rc_gt30=st["gt30"], rc_p50=st["p50"], rc_p75=st["p75"],
                  rc_p90=st["p90"], rc_p95=st["p95"])
    elif args.template == "warm3":
        block, s = neighbors_for(r, args.knn_show)
        if block is None:
            st, label = ref_for(r)
            block = "(no text neighbors available for this window bucket; reference class stats: "
            block += f"{label}, on-time {st['ontime']:.0f}%, median {st['p50']:+.0f}d, P90 {st['p90']:+.0f}d)"
            s = pd.Series([st["p50"], st["p90"]])
        kw.update(k=min(args.knn_show, len(s)), rc_window=r.rc_window, neighbors=block,
                  nb_ontime=float((s.abs() <= 1).mean() * 100),
                  nb_p50=float(s.median()), nb_p90=float(s.quantile(.9)))
    return PROMPT.format(**kw)

def knn_answer(r):
    """同窗口桶内描述最近邻:pred = posted + 近邻 slip 中位,p90 = posted + 近邻 slip P90。
    若窗口桶无索引,退到 lookup。"""
    if KNN is None or r.rc_window not in KNN:
        return lookup_answer(r)
    vec, nn, slips, _, _, _ = KNN[r.rc_window]
    d = (r.input_description or "").lower()
    import re as _re
    d = _re.sub(r"\btims\b", " ", d)
    dist, idx = nn.kneighbors(vec.transform([d]))
    s = pd.Series(slips[idx[0]]).dropna()
    pe = pd.to_datetime(r.input_promised_end, errors="coerce", utc=True)
    pred = (pe + pd.Timedelta(days=float(s.median()))).strftime("%Y-%m-%d")
    p90 = (pe + pd.Timedelta(days=float(s.quantile(.9)))).strftime("%Y-%m-%d")
    ontime = float((s.abs() <= 1).mean() * 100)
    return json.dumps({"predicted_end_date": pred, "p90_end_date": p90,
                       "confidence_0_100": int(round(ontime)),
                       "reasoning": f"knn: window={r.rc_window}, k={len(s)}, "
                                    f"mean_cos_dist={float(dist.mean()):.2f}"})

def lookup_answer(r):
    """纯查表:pred = posted + 类 P50,p90 = posted + 类 P90,confidence = 类准时率。"""
    st, label = ref_for(r)
    pe = pd.to_datetime(r.input_promised_end, errors="coerce", utc=True)
    pred = (pe + pd.Timedelta(days=st["p50"])).strftime("%Y-%m-%d")
    p90 = (pe + pd.Timedelta(days=st["p90"])).strftime("%Y-%m-%d")
    return json.dumps({"predicted_end_date": pred, "p90_end_date": p90,
                       "confidence_0_100": int(round(st["ontime"])),
                       "reasoning": f"lookup: {label}"})

client = anthropic.Anthropic()
rows, spent_in, spent_out = [], 0, 0
_TEMP_MODE = {"mode": None}   # "kw" | "extra_body" | "none",首次调用时探测

def _create(prompt):
    """兼容不同 SDK 版本的 temperature 传法。"""
    base = dict(model=MODEL, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
    order = ["kw", "extra_body", "none"] if _TEMP_MODE["mode"] is None else [_TEMP_MODE["mode"]]
    last = None
    for mode in order:
        try:
            if mode == "kw":
                msg = client.messages.create(**base, temperature=args.temperature)
            elif mode == "extra_body":
                msg = client.messages.create(**base, extra_body={"temperature": args.temperature})
            else:
                msg = client.messages.create(**base)
            if _TEMP_MODE["mode"] is None:
                _TEMP_MODE["mode"] = mode
                if mode == "none":
                    print(f"  [警告] 当前 SDK 不支持 temperature 参数,已忽略(实际为服务端默认温度)")
                elif mode == "extra_body":
                    print(f"  [提示] temperature 通过 extra_body 传入")
            return msg
        except TypeError as e:
            if "temperature" in str(e):
                last = e
                continue
            raise
    raise last
n_api, n_cache, n_apifail = 0, 0, 0

for i, (_, r) in enumerate(exam.iterrows()):
    if args.template == "lookup":
        raw = lookup_answer(r)
    elif args.template == "knn":
        raw = knn_answer(r)
    else:
        prompt = build_prompt(r)
        phash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        key = hashlib.sha256(f"{MODEL}|{TEMPLATE_VER}|T{args.temperature:g}|{r.qid}|{phash}".encode()).hexdigest()[:24]
        cf = f"{CACHE}/{key}.json"
        raw = None

    if args.template in ("lookup", "knn"):
        pass
    elif (not args.no_cache) and os.path.exists(cf):
        raw = json.load(open(cf))["text"]
        n_cache += 1
    else:
        try:
            msg = _create(prompt)
            raw = msg.content[0].text
            spent_in += msg.usage.input_tokens
            spent_out += msg.usage.output_tokens
            json.dump({"text": raw}, open(cf, "w"))
            n_api += 1
            time.sleep(0.2)
        except Exception as e:
            n_apifail += 1
            print(f"  [API 失败] qid={r.qid}: {e}")
            raw = None

    reasoning, p90 = None, None
    if raw is None:
        pred, conf = None, None
    else:
        try:
            j = json.loads(re.sub(r"```json|```", "", raw).strip())
            pred, conf = j.get("predicted_end_date"), j.get("confidence_0_100")
            p90 = j.get("p90_end_date")
            reasoning = j.get("reasoning")
        except Exception:
            pred, conf = None, None

    rows.append({"qid": r.qid, "label_actual_end": r.label_actual_end,
                 "input_promised_end": r.input_promised_end,
                 "input_first_seen": r.input_first_seen,
                 "label_slip_days": r.label_slip_days,
                 "is_censored": bool(r.is_censored),
                 "rc_type": r.rc_type, "rc_window": r.rc_window, "rc_hard": bool(r.rc_hard),
                 "pred_end": pred, "p90_end": p90, "confidence": conf,
                 "reasoning": reasoning, "raw": raw})
    if (i + 1) % 10 == 0:
        print(f"...{i+1}/{len(exam)}")

if args.template in ("lookup", "knn"):
    print(f"\n调用来源: 纯 {args.template},未调用模型")
else:
    print(f"\n调用来源: API {n_api}  缓存 {n_cache}  API失败 {n_apifail}")
if args.template not in ("lookup", "knn") and n_api == 0 and n_cache > 0 and not args.no_cache:
    print("  [提示] 本次全部命中缓存 (token=0 属正常)。改 prompt 或加 --no-cache 可强制重算。")
if n_apifail > 0:
    print(f"  [警告] 有 {n_apifail} 题 API 调用失败,其结果为空,已计入解析失败,未污染缓存。")

res = pd.DataFrame(rows)
# 两半来源的时间列类型不同(str vs Timestamp),统一成 UTC datetime 再落盘
for c in ["label_actual_end", "input_promised_end", "input_first_seen"]:
    res[c] = pd.to_datetime(res[c], errors="coerce", utc=True)
_strat = "_hard" if args.stratify else ""
_mtag = "nomodel" if args.template in ("lookup", "knn") else MODEL
_pop = "_wzonly" if args.no_noslip else ""
OUT_PATH = f"{DATA}/out/pilot_{_mtag}_{TEMPLATE_VER}{_strat}{_pop}_seed{SEED}.parquet"
res.to_parquet(OUT_PATH, index=False)

res["actual_dt"] = res.label_actual_end
def _parse_date(s):
    """从模型输出里抽 YYYY-MM-DD 再解析,避免 pandas 按首元素推断格式导致混格式全变 NaT。"""
    s = s.astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce", utc=True)
res["pred_dt"] = _parse_date(res.pred_end)
res["p90_dt"] = _parse_date(res.p90_end)
_nbad = int((res.pred_end.notna() & res.pred_dt.isna()).sum())
if _nbad:
    print(f"[警告] {_nbad} 条 pred_end 无法解析为日期: "
          + ", ".join(res.loc[res.pred_end.notna() & res.pred_dt.isna(), "pred_end"].astype(str).head(5)))
res["promised_dt"] = res.input_promised_end
res["first_seen_dt"] = res.input_first_seen
res["err_days"] = (res.pred_dt - res.actual_dt).dt.total_seconds().abs() / 86400

# 僵尸记录:一天以内的短工,却在 feed 里挂了很久
res["is_stale"] = (((res.promised_dt - res.first_seen_dt).dt.total_seconds() / 86400 < 1)
                   & (res.label_slip_days > args.stale_slip_days))

valid = res[~res.is_censored & ~res.is_stale]
cens = res[res.is_censored]
stale = res[res.is_stale & ~res.is_censored]
ok = valid.dropna(subset=["err_days"])

if valid.pred_end.notna().sum() == 0:
    print(f"\n[中止] 没有任何有效预测(解析失败 {res.pred_end.isna().sum()}/{len(res)}),不生成报告。")
    raise SystemExit(1)

print(f"\n== PILOT 结果 ({MODEL if args.template not in ('lookup','knn') else 'no-model'}, "
      f"template={TEMPLATE_VER}{', stratify=hard' if args.stratify else ''}, seed={SEED}) ==")
print(f"总题数 {len(res)}  有效 {len(valid)}  删失 {len(cens)}  僵尸 {len(stale)}")
print(f"解析失败: {res.pred_end.isna().sum()}/{len(res)} (有效样本内 {valid.pred_end.isna().sum()}/{len(valid)})")

print(f"\n[有效样本 n={len(ok)}] 误差(天): 中位 {ok.err_days.median():.1f}  P90 {ok.err_days.quantile(.9):.1f}")
print(f"[有效样本] 置信度: {valid.confidence.describe()[['mean','min','max']].round(0).to_dict()}")

if len(ok) > 0:
    med_c = ok.confidence.median()
    hi = ok[ok.confidence >= med_c]; lo = ok[ok.confidence < med_c]
    print(f"\n置信度 vs 误差(仅有效样本,以中位置信度 {med_c:.0f} 分组):")
    if len(hi):
        print(f"  高置信({len(hi)}题): 误差中位 {hi.err_days.median():.1f} 天  P90 {hi.err_days.quantile(.9):.1f} 天")
    if len(lo):
        print(f"  低置信({len(lo)}题): 误差中位 {lo.err_days.median():.1f} 天  P90 {lo.err_days.quantile(.9):.1f} 天")

if len(cens) > 0:
    print(f"\n[删失样本 n={len(cens)}] (真实结束未知,不计入误差):")
    print(f"  真实 slip_days: 中位 {cens.label_slip_days.median():+.1f}  "
          f"min {cens.label_slip_days.min():+.1f}  max {cens.label_slip_days.max():+.1f}")
    print(f"  模型解析成功 {cens.pred_end.notna().sum()}/{len(cens)} 题")

if len(stale) > 0:
    print(f"\n[僵尸记录 n={len(stale)}] (短工挂 feed 超 {args.stale_slip_days:g} 天,标签疑为 feed 疏忽,不计入误差):")
    for _, s in stale.iterrows():
        print(f"  {s.qid}  promised-first_seen<1d  slip={s.label_slip_days:+.1f}  pred={s.pred_end}  conf={s.confidence}")

# ---- 按来源拆分:work zone(真延期题)与 no-slip(posted=actual 送分题)分开看 ----
ok = ok.copy()
ok["source"] = ok.qid.astype(str).str.startswith("ncnoslip_").map({True: "noslip", False: "workzone"})
if "source" not in valid.columns:
    valid = valid.copy(); valid["source"] = valid.qid.astype(str).str.startswith("ncnoslip_").map({True: "noslip", False: "workzone"})
print("\n按来源拆分(仅有效样本):")
for tag in ["workzone", "noslip"]:
    g = ok[ok.source == tag]
    if len(g) == 0:
        print(f"  [{tag}] 0 题")
        continue
    line = (f"  [{tag} n={len(g)}] 误差中位 {g.err_days.median():.1f}  "
            f"P90 {g.err_days.quantile(.9):.1f}  "
            f"置信度均值 {g.confidence.mean():.0f}")
    if len(g) >= 4:
        mc = g.confidence.median()
        gh = g[g.confidence >= mc]; gl = g[g.confidence < mc]
        if len(gh) and len(gl):
            line += (f"  | 高置信({len(gh)}) 中位 {gh.err_days.median():.1f} P90 {gh.err_days.quantile(.9):.1f}"
                     f"  低置信({len(gl)}) 中位 {gl.err_days.median():.1f} P90 {gl.err_days.quantile(.9):.1f}")
    print(line)
    if tag == "workzone":
        # 模型是否倾向照抄 posted:预测与承诺日相差 <1 天的比例
        copy_rate = ((g.pred_dt - g.promised_dt).dt.total_seconds().abs() / 86400 < 1).mean()
        true_slip = g.label_slip_days.abs() > 1
        print(f"    照抄 posted 比例 {copy_rate*100:.0f}%  |  真延期(|slip|>1d)题 {int(true_slip.sum())} 题,"
              f"其中照抄 posted 的 {int((true_slip & ((g.pred_dt - g.promised_dt).dt.total_seconds().abs()/86400 < 1)).sum())} 题")
        # 方向命中:真延期 >7d 的题里,模型预测是否晚于 posted 至少 3 天
        late7 = g[g.label_slip_days > 7]
        if len(late7):
            pushed = ((late7.pred_dt - late7.promised_dt).dt.total_seconds() / 86400 >= 3).mean()
            under = ((late7.actual_dt - late7.pred_dt).dt.total_seconds() / 86400).median()
            print(f"    真延期>7d 题 {len(late7)} 题:模型往后推>=3d 的 {pushed*100:.0f}%,"
                  f"仍低估的中位天数 {under:+.1f}")
        # 困难层 / 简单层拆分
        for lab, sub in [("困难层", g[g.rc_hard]), ("简单层", g[~g.rc_hard])]:
            if len(sub):
                line2 = f"    {lab}(n={len(sub)}): 误差中位 {sub.err_days.median():.1f}  P90 {sub.err_days.quantile(.9):.1f}"
                if sub.p90_dt.notna().any():
                    h2 = sub[sub.p90_dt.notna()]
                    line2 += f"  P90覆盖 {(h2.actual_dt <= h2.p90_dt).mean()*100:.0f}%"
                print(line2)
        # P90 覆盖率(warm/warm2/lookup 才有)
        if g.p90_dt.notna().any():
            has = g[g.p90_dt.notna()]
            cover = (has.actual_dt <= has.p90_dt).mean()
            width = ((has.p90_dt - has.pred_dt).dt.total_seconds() / 86400).median()
            print(f"    P90 覆盖率 {cover*100:.0f}% (理想≈90%)  |  [pred, p90] 区间宽度中位 {width:.0f} 天  "
                  f"(n={len(has)})")

# ---- pinball loss(以 slip 天数为标尺:y = actual−posted, q50 用 pred,q90 用 p90)----
def _pinball(y, yhat, q):
    d = y - yhat
    return float(pd.concat([q * d, (q - 1) * d], axis=1).max(axis=1).mean())

wz_ok = ok[ok.source == "workzone"].copy() if "source" in ok.columns else ok.copy()
if len(wz_ok):
    y = (wz_ok.actual_dt - wz_ok.promised_dt).dt.total_seconds() / 86400
    yh50 = (wz_ok.pred_dt - wz_ok.promised_dt).dt.total_seconds() / 86400
    line = f"\n[workzone pinball loss] q50={_pinball(y, yh50, 0.5):.2f}"
    if wz_ok.p90_dt.notna().any():
        m = wz_ok.p90_dt.notna()
        yh90 = (wz_ok.p90_dt[m] - wz_ok.promised_dt[m]).dt.total_seconds() / 86400
        line += f"  q90={_pinball(y[m], yh90, 0.9):.2f}"
    print(line + "  (越小越好)")

# ---- 配对比较 ----
if args.compare_to and len(wz_ok):
    try:
        other = pd.read_parquet(args.compare_to)
        for c in ["label_actual_end", "input_promised_end"]:
            other[c] = pd.to_datetime(other[c], errors="coerce", utc=True)
        other["o_pred"] = _parse_date(other.pred_end)
        other["o_p90"] = _parse_date(other.p90_end) if "p90_end" in other else pd.NaT
        m = wz_ok.merge(other[["qid", "o_pred", "o_p90"]], on="qid", how="inner")
        m = m.dropna(subset=["o_pred"])
        if len(m) == 0:
            print(f"\n[配对] 与 {args.compare_to} 无重叠 qid")
        else:
            e_this = (m.pred_dt - m.actual_dt).dt.total_seconds().abs() / 86400
            e_oth = (m.o_pred - m.actual_dt).dt.total_seconds().abs() / 86400
            diff = e_this - e_oth
            win = (e_this < e_oth - 0.5).mean(); lose = (e_this > e_oth + 0.5).mean()
            y = (m.actual_dt - m.promised_dt).dt.total_seconds() / 86400
            p_this = _pinball(y, (m.pred_dt - m.promised_dt).dt.total_seconds() / 86400, 0.5)
            p_oth = _pinball(y, (m.o_pred - m.promised_dt).dt.total_seconds() / 86400, 0.5)
            print(f"\n[配对 vs {os.path.basename(args.compare_to)}] n={len(m)} (workzone 有效)")
            print(f"  点估计: 本次更准 {win*100:.0f}%  对方更准 {lose*100:.0f}%  平 {(1-win-lose)*100:.0f}%  "
                  f"(阈值 0.5 天);误差差值中位 {diff.median():+.1f} 天")
            print(f"  pinball q50: 本次 {p_this:.2f}  对方 {p_oth:.2f}")
            mm = m.dropna(subset=["p90_dt", "o_p90"])
            if len(mm):
                yy = (mm.actual_dt - mm.promised_dt).dt.total_seconds() / 86400
                p9_this = _pinball(yy, (mm.p90_dt - mm.promised_dt).dt.total_seconds() / 86400, 0.9)
                p9_oth = _pinball(yy, (mm.o_p90 - mm.promised_dt).dt.total_seconds() / 86400, 0.9)
                cov_this = (mm.actual_dt <= mm.p90_dt).mean(); cov_oth = (mm.actual_dt <= mm.o_p90).mean()
                w_this = ((mm.p90_dt - mm.pred_dt).dt.total_seconds() / 86400).median()
                w_oth = ((mm.o_p90 - mm.o_pred).dt.total_seconds() / 86400).median()
                print(f"  pinball q90: 本次 {p9_this:.2f}  对方 {p9_oth:.2f}  |  覆盖 本次 {cov_this*100:.0f}% 对方 {cov_oth*100:.0f}%"
                      f"  |  宽度中位 本次 {w_this:.0f}d 对方 {w_oth:.0f}d")
    except Exception as e:
        print(f"\n[配对] 读取 {args.compare_to} 失败: {e}")

if args.top_err > 0 and len(ok) > 0:
    worst = ok.sort_values("err_days", ascending=False).head(args.top_err)
    print(f"\n[有效样本 误差 Top-{len(worst)}] (定性用:看 promised/actual/pred 哪个离谱)")
    for _, w in worst.iterrows():
        src_tag = "noslip" if str(w.qid).startswith("ncnoslip_") else "workzone"
        span = (w.promised_dt - w.first_seen_dt).days if pd.notna(w.promised_dt) and pd.notna(w.first_seen_dt) else None
        print(f"  err={w.err_days:7.1f}d  conf={w.confidence!s:>4}  [{src_tag}]  {w.qid}")
        print(f"      first_seen={str(w.input_first_seen)[:19]}  promised={str(w.input_promised_end)[:19]}  "
              f"(promised-first_seen={span} d)")
        print(f"      actual   ={str(w.label_actual_end)[:19]}  pred={w.pred_end}"
              + (f"  p90={w.p90_end}" if pd.notna(w.p90_end) else "")
              + f"  slip={w.label_slip_days:+.1f}")
        if w.reasoning:
            print(f"      reasoning: {str(w.reasoning)[:120]}")

_price = {"claude-haiku-4-5": (1e-6, 5e-6), "claude-sonnet-4-6": (3e-6, 15e-6)}.get(MODEL, (3e-6, 15e-6))
print(f"\ntoken消耗: in={spent_in} out={spent_out}  约${spent_in*_price[0]+spent_out*_price[1]:.3f}  ({MODEL})")
print(f"结果已写: {OUT_PATH}")
