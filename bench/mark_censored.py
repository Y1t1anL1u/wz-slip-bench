#!/usr/bin/env python3
"""
mark_censored.py — 标记 WZDx task_c 数据里的右删失样本

背景:
    label_actual_end 里混入了多个 ETL/抓取批次的运行时间戳(如
    2022-06-04 17:11:56/57/58、2022-10-01 04:06:02 等),这些施工区
    是在承诺结束前从 feed 掉了、被系统统一盖了个当前时间戳当"结束",
    并非真实完工。它们的 slip_days 是虚假的,会污染误差均值。

策略:
    不删除,而是新增一列 is_censored 标记为删失。评测时对
    is_censored==True 的样本单独统计,不混入误差均值。

判定规则(v3):
    规则1(默认):高频 ∧ 带秒。
        - 高频:某个精确时间戳出现次数 >= --min-count(默认5)。
        - 带秒:秒数非零或含亚秒(ss!=00 或 microsecond!=0)。
        两条件同时满足才判为批处理污染。整点高频(如 05:00:00)放行,
        视为"多施工段同一整点计划结束"的真实数据。

    规则3(默认):极端提前兜底。
        slip_days < -(--max-early-days)(默认60)一律标记。
        施工区提前两个月以上"结束"物理上不合理,基本等于中途从 feed
        消失(右删失)。此规则不依赖时间戳频次,能兜住批处理时间戳散落
        在相邻几秒、单秒不够高频而被规则1 漏掉的漏网样本(如 17:12:01)。
        注意:只兜"极端提前",不动正的极端值(如 +365 可能是真延期)。

    规则2(可选,--use-second-rule):所有秒数非零的记录一律标记。
        比规则1 更严(不要求高频),可能误伤真实带秒记录,默认关闭。

用法示例:
    python mark_censored.py
    python mark_censored.py --min-count 3
    python mark_censored.py --max-early-days 90
    python mark_censored.py --use-second-rule
    python mark_censored.py --in C:/wzdx/out/task_c_slip_nc.parquet \
                            --out C:/wzdx/out/task_c_slip_nc_marked.parquet
"""
import argparse
import sys
import numpy as np
import pandas as pd
import os as _os
DATA = _os.environ.get("WZDX_DIR", "C:/wzdx")  # root holding out/ and cache/; override with env WZDX_DIR



def has_sub_minute(ts_series, exempt_5959=True):
    """返回布尔 Series:时间戳是否"带秒"(机器时刻特征)。
    整点/整分(ss==00)不算;HH:59:59 是 NCDOT 排班惯例(如 15:59:59、04:59:59),
    exempt_5959=True 时也不算。"""
    sub = (ts_series.dt.second != 0) | (ts_series.dt.microsecond != 0)
    if exempt_5959:
        is5959 = (ts_series.dt.minute == 59) & (ts_series.dt.second == 59) & (ts_series.dt.microsecond == 0)
        sub = sub & ~is5959
    return sub


def _is_machine_ts(ts, exempt_5959=True):
    if ts.second == 0 and ts.microsecond == 0:
        return False
    if exempt_5959 and ts.minute == 59 and ts.second == 59 and ts.microsecond == 0:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="标记 WZDx 右删失样本 (v3: 高频∧带秒 + 极端提前兜底)")
    ap.add_argument("--in", dest="inp",
                    default=f"{DATA}/out/task_c_slip_nc.parquet",
                    help="输入 parquet 路径")
    ap.add_argument("--out", dest="out",
                    default=f"{DATA}/out/task_c_slip_nc_marked.parquet",
                    help="输出 parquet 路径(含 is_censored 列)")
    ap.add_argument("--min-count", type=int, default=5,
                    help="规则1 高频阈值:精确时间戳出现次数达到该值(默认5)")
    ap.add_argument("--max-early-days", type=float, default=60.0,
                    help="规则3 极端提前兜底:slip_days < -该值 一律标记(默认60)")
    ap.add_argument("--no-exempt-5959", action="store_true",
                    help="不把 HH:59:59 视为排班惯例(默认视为惯例、放行)")
    ap.add_argument("--neighbor-seconds", type=int, default=120,
                    help="规则4 邻居:带秒时间戳落在任一规则1命中时间戳 ±该秒数内也标记(默认120;0 关闭)")
    ap.add_argument("--use-second-rule", action="store_true",
                    help="规则2:把所有秒数非零记录都标记(不要求高频,默认关闭,偏严)")
    ap.add_argument("--end-col", default="label_actual_end",
                    help="实际结束日期列名(默认 label_actual_end)")
    ap.add_argument("--slip-col", default="label_slip_days",
                    help="滑移天数列名(默认 label_slip_days)")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    n = len(df)
    if args.end_col not in df.columns:
        sys.exit(f"[错误] 找不到列 {args.end_col};现有列:{list(df.columns)}")

    ae = pd.to_datetime(df[args.end_col], utc=True)
    sub_minute = has_sub_minute(ae, exempt_5959=not args.no_exempt_5959)   # 机器时刻特征

    # --- 频次统计 ---
    vc = ae.value_counts()
    mass_ts = vc[vc >= args.min_count]       # 所有高频时间戳(含整点)
    mass_set = set(mass_ts.index)
    is_mass = ae.isin(mass_set)

    # --- 规则1:高频 ∧ 带秒 ---
    rule1 = is_mass & sub_minute

    # --- 规则3:极端提前兜底 ---
    has_slip = args.slip_col in df.columns
    if has_slip:
        slip = pd.to_numeric(df[args.slip_col], errors="coerce")
        rule3 = slip < -args.max_early_days
    else:
        rule3 = pd.Series(False, index=df.index)

    # --- 规则4:批处理簇的邻居 ---
    # 同一批抓取会把结束时间戳散在相邻几十秒上,单秒不够高频而漏网(如 17:11:55、17:12:01)。
    # 凡带秒、且落在任一规则1命中时间戳 ±N 秒内的,一并标记。
    mass_with_sec_ts = [t for t in mass_set if _is_machine_ts(t, not args.no_exempt_5959)]
    if args.neighbor_seconds > 0 and len(mass_with_sec_ts) > 0:
        centers = pd.Series(sorted(mass_with_sec_ts))
        ae_sec = ae.astype("int64") // 10**9
        c_sec = centers.astype("int64") // 10**9
        # 对每条记录,找最近的簇中心距离(用 searchsorted)
        idx = c_sec.searchsorted(ae_sec.values)
        idx_lo = (idx - 1).clip(0, len(c_sec) - 1)
        idx_hi = idx.clip(0, len(c_sec) - 1)
        d = np.minimum(np.abs(ae_sec.values - c_sec.values[idx_lo]),
                       np.abs(c_sec.values[idx_hi] - ae_sec.values))
        rule4 = pd.Series(d <= args.neighbor_seconds, index=df.index) & sub_minute & ~is_mass
        # 记录每条的最近簇中心,用于明细
        nearest = np.where(np.abs(ae_sec.values - c_sec.values[idx_lo]) <= np.abs(c_sec.values[idx_hi] - ae_sec.values),
                           idx_lo, idx_hi)
        rule4_center = pd.Series(centers.values[nearest], index=df.index)
    else:
        rule4 = pd.Series(False, index=df.index)

    # --- 规则2(可选):所有带秒记录 ---
    if args.use_second_rule:
        rule2 = sub_minute
    else:
        rule2 = pd.Series(False, index=df.index)

    rule1 = pd.Series(rule1.values, index=df.index)
    rule3 = pd.Series(rule3.values, index=df.index) if has_slip else rule3
    is_censored = rule1 | rule2 | rule3 | rule4

    df = df.copy()
    df["is_censored"] = is_censored.values

    # --- 高频时间戳拆两类,便于核对 ---
    def _is_machine(ts):
        if ts.second == 0 and ts.microsecond == 0:
            return False
        if (not args.no_exempt_5959) and ts.minute == 59 and ts.second == 59 and ts.microsecond == 0:
            return False
        return True
    mass_with_sec = [(ts, c) for ts, c in mass_ts.items() if _is_machine(ts)]
    mass_hourly = [(ts, c) for ts, c in mass_ts.items() if not _is_machine(ts)]

    # 规则3 独有命中(不与规则1/2 重叠的部分,体现兜底新增了多少)
    rule3_only = rule3 & ~(rule1 | rule2 | rule4) if has_slip else pd.Series(False, index=df.index)

    # --- 报告 ---
    print("=" * 60)
    print(f"输入: {args.inp}")
    print(f"总样本: {n}")
    print(f"高频时间戳(频次 >= {args.min_count})共 {len(mass_ts)} 个,"
          f"覆盖 {int(is_mass.sum())} 条记录")
    print(f"  其中 带秒/亚秒(判为批处理污染): {len(mass_with_sec)} 个时间戳 -> "
          f"规则1 命中 {int(rule1.sum())} 条")
    print(f"  其中 整点/整分/HH:59:59(视为排班惯例,放行): {len(mass_hourly)} 个时间戳")
    if has_slip:
        print(f"规则3 极端提前 (slip_days < -{args.max_early_days:g}): "
              f"命中 {int(rule3.sum())} 条,其中兜底新增(规则1/2 未覆盖)"
              f" {int(rule3_only.sum())} 条")
    else:
        print(f"规则3 极端提前: 跳过 (找不到列 {args.slip_col})")
    print(f"规则4 邻居 (机器时刻且落在规则1簇中心 ±{args.neighbor_seconds}s): 新增 {int(rule4.sum())} 条")
    if int(rule4.sum()) > 0:
        print("  逐簇明细(簇中心: 邻居数):")
        for ts, c in rule4_center[rule4].value_counts().items():
            print(f"    {ts}   +{c}")
    if args.use_second_rule:
        print(f"规则2 (所有带秒记录): {int(rule2.sum())} 条  [已启用]")
    else:
        print("规则2 (所有带秒记录): 未启用 (加 --use-second-rule 开启)")
    print(f"合计标记为删失 is_censored=True: {int(is_censored.sum())} "
          f"({is_censored.mean()*100:.1f}%)")
    print(f"保留为有效  is_censored=False: {int((~is_censored).sum())}")

    print("\n[规则1 命中] 高频且带秒 -> 判为批处理时刻:")
    for ts, c in sorted(mass_with_sec, key=lambda x: -x[1]):
        print(f"  {ts}   x{c}")

    print("\n[放行] 高频但为整点/整分/HH:59:59 -> 视为排班惯例,不标记:")
    for ts, c in sorted(mass_hourly, key=lambda x: -x[1]):
        print(f"  {ts}   x{c}")

    if has_slip and int(rule3_only.sum()) > 0:
        print(f"\n[规则3 兜底新增] 极端提前但被规则1/2 漏掉的样本 "
              f"(共 {int(rule3_only.sum())} 条,按 slip 升序前20):")
        cols = [c for c in [args.end_col, args.slip_col] if c in df.columns]
        sub = df.loc[rule3_only, cols].copy()
        sub = sub.sort_values(args.slip_col).head(20)
        for _, r in sub.iterrows():
            print(f"  {r[args.end_col]}   slip={r[args.slip_col]:+.2f}")

    # --- slip_days 对比 ---
    if has_slip:
        clean = df.loc[~is_censored, args.slip_col]
        cens = df.loc[is_censored, args.slip_col]
        print(f"\n{args.slip_col} 统计对比:")
        print(f"  有效样本 (n={len(clean)}): "
              f"mean={clean.mean():+.2f}  "
              f"min={clean.min():+.2f}  max={clean.max():+.2f}  "
              f"P90={clean.abs().quantile(0.9):.2f}")
        if len(cens):
            print(f"  删失样本 (n={len(cens)}): "
                  f"mean={cens.mean():+.2f}  "
                  f"min={cens.min():+.2f}  max={cens.max():+.2f}")
        else:
            print("  删失样本: 0 条")

    df.to_parquet(args.out, index=False)
    print(f"\n已写出: {args.out}  (新增列 is_censored)")
    print("=" * 60)


if __name__ == "__main__":
    main()
