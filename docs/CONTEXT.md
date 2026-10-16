# WZDx 工区结束日预测评测 — 项目上下文文档

> 用途:把这份文档整体贴给任何一个新的对话/协作者,对方即可完整了解本项目做了什么、数据长什么样、方法与基线是什么、目前得到了哪些结论、还差什么。所有数字均来自实际运行输出(2026-09,claude-haiku-4-5)。
> 本文档**不涉及**任何外部论文,只描述本项目自身。

---

## 0. 一句话

用北卡罗来纳州 WZDx/TIMS 真实施工区 feed,构建了一个**"给定一条工区记录,预测它实际何时结束(相对 posted end date 的滑移)"**的评测基准;完成了标签清洗、分层设计、三条非 LLM 基线与三种 LLM prompt 的多 seed 对比。三步闭环:(1) 标签清洗与分层基准;(2) LLM vs 非 LLM 基线的配对评测——haiku 显著差于 TF-IDF kNN,sonnet 与 kNN 无显著差异(n=217,p=0.56),即 LLM 从描述里没有提取到超出文本最近邻的信号;(3) 决策层 + split conformal 闸门——在 1,892 条困难层记录上,误报率 ≤ α 的保证在 τ=7/14/30 天下均成立,最优非 LLM score 在 α=0.1 下召回约 30%、精确率较基率提升 1.8–2.6 倍,LLM score 无增量,LLM 自报置信度不可用作 nonconformity score。

---

## 1. 任务定义

**输入**(一条 WZDx 记录,首次出现在 feed 时的快照):
- `input_road`, `input_direction`:道路与方向
- `input_description`:自由文本描述(如 "Nightly lane closure 7pm–6am for bridge deck rehabilitation …")
- `input_first_seen`:首次观测时间
- `input_promised_end`:当时 feed 上 posted 的结束时间

**目标**:
- `label_actual_end`:记录最终从 feed 上观测到的实际结束时间
- `label_slip_days = actual_end − promised_end`(天,可正可负)

**输出**(LLM/基线):`predicted_end_date`(点估计)、`p90_end_date`(90% 分位上界)、`confidence_0_100`、`reasoning`。

**评测指标**(均在"有效样本"上,见 §3):
- 点估计绝对误差(天)的中位数与 P90
- "照抄 posted 比例":预测与 posted 相差 <1 天的占比
- 真延期(>7 天)题的方向命中率(预测比 posted 晚 ≥3 天)与仍低估中位天数
- P90 覆盖率(actual ≤ p90 的比例,理想 ≈ 90%)与区间宽度中位数(p90 − pred)
- 按来源(work zone / no-slip)与按层(困难/简单)拆分报告

---

## 2. 数据

### 2.1 文件与规模(均在 `C:/wzdx/out/`)
| 文件 | 内容 | 规模 |
|---|---|---|
| `task_c_slip_nc.parquet` | 滑移任务样本(原始) | 11,108 行;其中 `meta_is_incident=False` 的 work zone 5,726 行 |
| `task_c_slip_nc_marked.parquet` | 上表 + `is_censored` 列(清洗后) | 同上;删失 391 (3.5%),有效 10,717 |
| `north_carolina/entities_*.parquet` | 全体 feed 实体(用于构造 no-slip 样本、推断观测窗口) | 6,739,847 个唯一 `entity_id` |
| `pilot_<model>_<template>[_hard]_seed<N>.parquet` | 每次评测的逐题结果 | 每个 50 行 |

`task_c` 列:`qid, input_description, input_road, input_direction, input_first_seen, input_promised_end, label_actual_end, label_slip_days, meta_is_incident`。
`entities` 列(用到的):`entity_id, description, road, direction, first_seen, last_seen, first_end_date, last_end_date, is_incident`。

观测窗口结束 `obs_end = 2024-08-06`(entities 最后观测时间的最大值)。

### 2.2 关键事实
- `task_c` 是**滑移富集样本**:它只含结束日曾变动过的记录;而全 feed 中"结束日从未变动"的记录有 137 万条(过滤后)。两者不是同一人群。
- 6.7M 个 entity_id 全部唯一 → 同一物理工区在 feed 中会被反复以新 ID 重发("换 ID 续命"),这是系统性现象。

---

## 3. 标签清洗(核心工作之一)

首次 pilot 出现 1171 天级误差,追查后确认是**标签缺陷**而非模型问题。共识别并处理四类:

| # | 缺陷 | 表现 | 处理(`mark_censored.py`) |
|---|---|---|---|
| 1 | 批处理时间戳冒充实际结束 | 大量记录 `actual_end` 落在同一秒/相邻几秒(如 `2022-06-04 17:11:56/57/58`、`2022-10-01 04:06:02`),真实 slip 未知 | 规则1:同一精确时间戳出现 ≥5 次 **且** 为"机器时刻"(秒≠0 且非 `HH:59:59`)→ 标删失;规则4:机器时刻且落在规则1 簇中心 ±120s 内 → 标删失(+301 条) |
| 2 | 极端提前 = 记录中途从 feed 消失 | slip < −60 天 | 规则3:一律标删失(兜底 +10 条) |
| 3 | 承诺日超出观测窗口,结果从未被检验 | posted 在 2027 年、`actual == promised`、slip 恰好 0 | `runner.py` 组卷时剔除:promised 与 actual 都必须早于 `obs_end − 30 天`;no-slip 样本额外要求记录在 promised ±7 天内离开 feed |
| 4 | 僵尸记录 | 一夜施工的记录挂在 feed 上 150 天 | `is_stale`:promised − first_seen < 1 天 且 slip > 30 → 单独统计不计误差 |

重要修正:整点/整分(`HH:00:00`)与 `HH:59:59` 是 NCDOT 排班惯例,高频也**放行**,否则会误伤真实计划结束时间。

未处理但已识别:**换 ID 续命**(描述说 "Spring 2027",记录却在 2022-09 结束——旧 ID 消失、新 ID 接续),需要实体串联或改题目定义。

删失样本策略:**标记但保留**,评测时单独统计,不并入误差。

---

## 4. 分组统计(条件基率)

`group_stats.py` 在有效 work zone 样本(n=5,429)上按描述特征分组,发现滑移结构:

**全体**:准时(±1d) 60.0%,>7d 18.5%,>30d 6.6%;P50 +0.4,P75 +5,P90 +18,P95 +38。

**posted 窗口长度(promised − first_seen)是压倒性主特征**:
| 窗口 | n | 准时 | >7d | P90 |
|---|---|---|---|---|
| ≤1d | 3194 | 84.4% | 5.7% | +3 |
| 2–7d | 1517 | 26.6% | 36.1% | +37 |
| 8–30d | 620 | 24.0% | 38.1% | +36 |
| >30d | 98 | 9.2% | 40.8% | +160 |

**项目类型是第二主特征**(互斥、按优先级首个命中,`refclass.py`):
| 类型 | n | 准时 | P90 | 备注 |
|---|---|---|---|---|
| incident | 2187 | 87.7% | +1 | 几乎从不延期 |
| construction | 871 | 50.7% | +27 | |
| maintenance | 671 | 43.1% | +15 | |
| bridge | 538 | 24.7% | +34 | |
| ramp | 188 | 42.0% | +36 | |
| paving | 115 | 15.7% | +47 | P95 +123 |
| pipe/utility | 128 | 22.7% | +42 | |
| signal/sign | 76 | 15.8% | +56 | |
| guardrail | 49 | 34.7% | +96 | **双峰**,P95 +151 |
| culvert | 42 | 2.4% | +56 | |

**周期排期语言(nightly / Mon–Fri / 7pm–6am)在控制窗口后几乎无信息**:2–7d 内 recurring True/False 的 P90 为 +37 vs +35。这推翻了早期 prompt 里"周期排期 → 会拖很久"的手写规则。

**困难层定义**:窗口 >1 天 且 类型 ∉ {incident, mowing/trees};占 work zone 池 35%(1,989 / 5,684)。随机抽样时六成题是"照抄 posted 即正确"的送分题,因此正式对比一律在困难层上做。

---

## 5. 方法与基线(`runner.py --template …`)

| 模板 | 是否调模型 | 做法 |
|---|---|---|
| `cold` | 是 | 零信息 prompt,只给记录 |
| `warm` (v1) | 是 | 注入**总体**基率 + 手写规律(含错误的"周期排期"规则),要求输出 p90 |
| `warm2` | 是 | 代码先算好该记录的**参考类**(窗口桶 × 类型),把该类的准时率/P50/P75/P90/P95 注入 prompt,要求点估计留在 posted+类中位、尾部风险放 p90 |
| `lookup` | 否 | 纯查表:pred = posted + 类 P50,p90 = posted + 类 P90,confidence = 类准时率 |
| `knn` | 否 | 同窗口桶内,描述 TF-IDF(1–2gram)余弦最近邻 k=30,pred/p90 = posted + 近邻 slip 的 P50/P90 |
| `warm3` | 是 | 检索增强:把 kNN 最近的 8 条历史记录(含各自 posted 窗口与真实 slip)连同近邻汇总喂给模型,允许其在描述指向长期工程时远超近邻中位。用于"LLM + 检索 vs 纯检索"的公平对比 |

正式对比设置(第二步起):`temperature=0`(通过 `extra_body` 传入,SDK 兼容),`--no-noslip`(目标人群 = task_c 困难层),`--stratify`,seeds 1–10(每 seed 25 题,配对后 n=217)。

参考类表与 kNN 索引均**排除本次考卷的 qid**(防泄漏);类样本 <30 时退到窗口层,再退到全体。

组卷:每 seed 50 题 = 25 work zone + 25 no-slip(后者来自 entities 中结束日从未变动、且承诺日在窗口内、且在承诺日附近离开 feed 的记录)。`--stratify` 时 work zone 25 题全部从困难层抽。seeds 1–5。缓存 key 含模型名 + 模板版本 + qid + prompt 哈希;改 prompt 自动失效。

---

## 6. 结果

模型:`claude-haiku-4-5`,默认温度(未固定,见 §8)。以下均为 **work zone 那半**的数字;no-slip 那半单列。

### 6.1 随机抽样(含六成送分题):点估计误差 P90(天)
| seed | cold | warm v1 | lookup |
|---|---|---|---|
| 1 | 27.6 | 18.5 | 26.9 |
| 2 | 30.4 | 22.6 | 27.8 |
| 3 | 31.0 | 25.0 | 17.9 |
| 4 | 28.3 | 19.1 | 31.2 |
| 5 | 15.6 | 11.4 | 6.7 |

- cold 照抄 posted 比例 58–71%;warm v1 降到 46–71%,真延期题里照抄从 4–5 题降到 2–3 题,P90 稳定下降 25–35%。
- 但 warm v1 把本该准时的记录也推后(no-slip 半 P90 从 <1 天变成 ≈6.4 天),且是靠"周期排期"这条**错误**规则推的——产生了一批新的误判(slip≈0 却推 2–4 周)。
- LLM 自报 confidence:清洗前看似有分层(高置信误差 1 天 / 低置信 20 天),**清洗后消失**——原来的分层是模型正确地不信任 2027 占位日贡献的评测噪声;warm 之后 confidence 塌成常数(≈35 或 ≈62),无信息量。

### 6.2 困难层(`--stratify`,同题配对):中位 / P90 / P90覆盖 / 区间宽度 / 真延期仍低估
| seed | lookup | warm2 (haiku) | kNN |
|---|---|---|---|
| 1 | 6.4 / 29.1 / 88% / 31d / +13.5 | 6.3 / **29.1** / 92% / 28d / +13.5 | 4.8 / 29.9 / 83% / **17d** / **+10.8** |
| 2 | 2.6 / 36.1 / 88% / 31d / +17.4 | 2.6 / **36.1** / 83% / 26d / +17.4 | 2.2 / 30.9 / 83% / **16d** / **+12.4** |
| 3 | 3.8 / 33.9 / 91% / 34d / +23.2 | 4.1 / **33.9** / 91% / 30d / +23.2 | 5.1 / 33.9 / 83% / **12d** / **+19.4** |
| 4 | 4.1 / 9.9 / 100% / 32d / +7.8 | 3.4 / **10.0** / 100% / 29d / +8.8 | 3.4 / 13.0 / 100% / **14d** / **+5.5** |
| 5 | 4.8 / 23.8 / 96% / 33d / +9.6 | 3.9 / **23.8** / 91% / 30d / +10.0 | 3.6 / 19.4 / 91% / **18d** / **+7.7** |

**读法**:
- **warm2 ≡ lookup**:五个 seed 点估计 P90 精确到 0.1 天相同。看 reasoning 可知模型在做"posted + 类中位"的算术("applying class median slip of +5.3 days to posted … gives …")。即使 reasoning 里读出了描述中的具体日期("continuous closure through 5/25 … overrides class median"),预测仍未偏离查表值。
- **kNN 优于 lookup**:同等覆盖率下区间宽度砍半(12–18 vs 31–34 天);真延期仍低估天数五个 seed 全部缩小;若干"posted 5 天、实际 130–200 天"的极端案例被近邻部分捕捉(如 rest area closure 的 p90 给到距实际两周内)。代价:对准时记录误推更多(no-slip 半 +7 天,vs lookup +4,cold ≈0)。
- 因此结论从"文本无信号"修正为"**文本有查表之外的信号,haiku + 当前 prompt 没提取出来**"。

### 6.3 no-slip 那半(25 题/seed,posted == actual)
cold ≈ 0.3 天(照抄即对);warm v1 ≈ +6;lookup ≈ +4;kNN ≈ +7。所有基于 task_c 校准的方法都会把这一半系统性推后——因为 task_c 是滑移富集人群。

### 6.4 第二步:LLM + 检索 vs 纯检索(困难层,10 seeds 合并配对,n=217,`compare_runs.py`)
| 对比 | 点估计胜/负/平 | 误差中位 | 误差 P90 | Wilcoxon p | pinball q50 | pinball q90 | P90 覆盖 |
|---|---|---|---|---|---|---|---|
| sonnet warm3 vs kNN | 21% / 17% / 62% | 3.8 vs 3.9 | 29.8 vs 30.9 | **0.564** | 6.93 vs 6.68 | 6.83 vs 6.92 | 88% vs 89% |
| haiku warm3 vs kNN | 19% / 31% / 50% | 5.2 vs 3.9 | 31.9 vs 30.9 | **<0.001** | 7.07 vs 6.68 | 7.42 vs 6.92 | 76% vs 89% |
| sonnet vs haiku | 29% / 13% / 58% | 3.8 vs 5.2 | 29.8 vs 31.9 | **<0.001** | 6.93 vs 7.07 | 6.83 vs 7.42 | 88% vs 76% |

按真实 slip 分层(点估计误差中位,sonnet / kNN):准时 1.8/1.9;1–7d 2.4/2.3;7–30d 6.5/6.7;>30d 36.7/42.5(n=26,不显著)。

结论:模型变强把 LLM 从"显著差于检索"拉到"等于检索",但未超过。sonnet 偏离近邻时高方差、期望为零(如 bridge deck rehab +201d 它推到 +92 是赢,widening 项目实际 −23d 它推到 +140 是输)。sonnet 的 reasoning 多次识别出"同一工程的不同记录"(如 I-40 Winston-Salem rehab),即它看到了换 ID 续命现象,但标签按记录打,故不稳定获益。haiku 在拿到近邻后反而收窄区间(覆盖 76%),是错误方向。

### 6.5 第三步:决策层 + split conformal(`conformal.py`)
决策:是否发出"预计延期 > τ 天"的提醒。保证:在校准集真实未延期记录的 score 上取带修正的 (1−α) 分位作阈值 → P(提醒 | 未延期) ≤ α。20 次随机 50/50 划分平均。

**pool 模式(困难层全体 n=1,892,非 LLM score,α=0.1)**:
| τ | 正例比例 | 最优 score | 召回 | 精确率(基率) | 平均 FPR / 最大 |
|---|---|---|---|---|---|
| 7d | 36.8% | s_combo_knn | 32% | 66% (37%) | 9.9% / 12.5% |
| 14d | 23.5% | s_knn_med | 29% | 47% (23%) | 10.2% / 15.0% |
| 30d | 13.3% | s_knn_p90 | 36% | 34% (13%) | 10.2% / 13.3% |

召回 vs α(τ=7,s_combo_knn):α=0.05→16%,0.1→32%,0.15→44%,0.2→51%,0.3→64%。信息量排序在所有 τ 下一致:kNN 近邻特征 > 参考类 > 裸 posted 窗口。合成 score(交叉拟合 logistic,系数几乎全落在 `s_knn_frac`)在 α=0.1 仅比单特征多 1–2 点,α≥0.15 多 5–8 点。不校准的固定规则(kNN frac≥0.3)τ=7 时召回 77% 但 FPR 44%——闸门的作用是把 FPR 压到 10% 并把精确率从 37% 提到 66%,代价是召回降到 ~30%。

**pilot 模式(sonnet 的 217 条,α=0.1,τ=7)**:s_llm_pred 召回 39%,s_knn_med 34%,s_knn_frac 32%(82 个正例,噪声 ±10 点);合成 score 加入 LLM 特征后 33% vs 32%,无增量。s_llm_1mconf(100−自报置信度)召回卡在 17%——置信度只取几个离散值,不能作连续 score。合成 score 在 n=217 上 FPR 超 α(12%,最大 27%),系数与 pool 上完全不同 → 合成/校准需 ≥10³ 量级样本;非 LLM score 能在全量历史上校准,这是其作闸门的实际优势。

**校准样本量 → FPR 稳定性(附录,τ=7,α=0.1,50 次划分)**:
| 校准未延期样本数 | s_knn_frac 平均 FPR / 最大 | s_combo_knn 平均 FPR / 最大 | s_combo_knn 召回 |
|---|---|---|---|
| ≈120 (10%) | 8.4% / 17.8% | 8.4% / 15.3% | 28% |
| ≈240 (20%) | 8.4% / 11.4% | 9.1% / 15.0% | 31% |
| ≈360 (30%) | 8.5% / 12.0% | 9.4% / 13.9% | 32% |
| ≈600 (50%) | 8.3% / 12.6% | 9.5% / 14.1% | 32% |

平均 FPR 在任何规模下 ≤ α(期望意义的保证);单次划分最坏 FPR 从 ≈18% 收敛到 ≈12%,拐点约 240 条未延期记录(≈400 条困难层记录)。召回随校准集变大略升(修正变松)。这解释了 pilot 模式(≈70 条未延期)FPR 最大 27% 的不稳定。

### 6.6 成本
haiku:cold ≈ $0.03/seed(50 题),warm/warm2 ≈ $0.05/seed,warm3 ≈ $0.035/seed(25 题);sonnet warm3 ≈ $0.11/seed。lookup/kNN/conformal 零成本。第二、三步总花费约 $1.5。

---

## 7. 结论(截至目前)

1. 原始标签不能直接用于评测;四类缺陷各有可复现的检测规则,清洗后极端误差消失(有效样本 slip 范围收敛到 [−54, +365])。
2. 滑移由 posted 窗口长度主导,其次是项目类型;描述里的周期排期语言在控制窗口后无增量。
3. 滑移重尾(guardrail 双峰,P95 可达 +150 天),点估计天然吃亏,区间预测(p90)应作为主要输出与评测对象。
4. haiku 在 cold/warm/warm2 下均未超过 (窗口 × 类型) 查表;warm2 下与查表完全一致。LLM 自报 confidence 在清洗后的数据上无信息量。
5. TF-IDF kNN 在困难层上以一半的区间宽度达到相近覆盖率,并系统性减少对真延期的低估 → 描述文本有可提取信号,当前 LLM 配置未提取。
6. task_c(滑移富集)与全 feed(99%+ 从不变动)是两个人群;正式评测已将目标人群定为 task_c 困难层(去掉 no-slip 半)。
7. **LLM + 检索 ≈ 纯检索**:sonnet 与 TF-IDF kNN 在 217 道配对题上无显著差异(p=0.56),haiku 显著更差。描述文本中可利用的信号已被最近邻取尽;LLM 的领域推理高方差、期望为零。
8. **conformal 闸门可行且保证成立**:1,892 条上 FPR ≤ α 在 τ=7/14/30 下均验证。首次观测信息下召回天花板约 30%(与 τ、score 组合无关),精确率提升 1.8–2.6 倍。
9. **LLM 自报置信度不能作 nonconformity score**(离散、无信息);非 LLM score 因可在全量历史上校准而更适合做闸门。

---

## 8. 已知未决问题 / 限制
- **换 ID 续命**未处理,写为 limitation:同一物理工程在 feed 中被反复以新 ID 重发,标签按记录打;sonnet 的 reasoning 已能识别同一工程(定性证据)。可选处理:按路段+描述相似+时间衔接串联实体,或改题为"该记录何时被移除"。
- **召回天花板 ~30%** 来自"只看首次观测";序贯版本(闸门随记录存活时间更新)预期可大幅提升,属 future work。
- **规则4 邻居 ±120s** 略过严(301 条);对结论无影响,发布前抽查。
- 类型关键词为正则,`other` 仍有 165 条(obstruction / ferry / rest area / fire)。
- 已解决:目标人群(task_c 困难层)、温度(0)、样本量(10 seeds,n=217 配对;conformal 用 1,892)。

---

## 9. Repo 结构(`C:\wzdx\`)

```
C:\wzdx\
├── runner.py            # 主评测脚本:组卷、模板、调用/缓存、指标报告
│                        #   --template cold|warm|warm2|warm3|lookup|knn
│                        #   --seed N  --stratify  --no-noslip  --model  --temperature 0
│                        #   --n-wz 25 --n-noslip 25  --compare-to <parquet>(同题配对)
│                        #   --no-cache  --top-err N  --knn-k 30  --knn-show 8
│                        #   --window-margin-days 30  --noslip-exit-tol-days 7  --stale-slip-days 30
├── refclass.py          # 参考类定义:类型正则、窗口分桶、RefTable;build_knn / knn_neighbors(供 conformal 复用)
├── compare_runs.py      # 多 seed 合并配对检验:胜率、Wilcoxon/符号检验、pinball、覆盖、按真实 slip 分层
├── conformal.py         # 第三步:决策层 + split conformal;--source pool|pilot  --tau  --alpha  --alphas  --splits
│                        #   score:kNN 近邻中位/>τ 比例/P90、参考类 gt7/P90、窗口长度、交叉拟合合成 score、(pilot) LLM pred/p90/1−conf
├── mark_censored.py     # 标签清洗:规则1/3/4 (+可选规则2),写出 *_marked.parquet,打印逐簇明细
├── group_stats.py       # 分组统计:窗口 × 类型 × 周期语言的滑移分布;未分类描述高频词
├── first_exam.py        # 最早的单题 prompt 演示
├── test.py              # 临时诊断脚本(时间戳频次等)
├── cache\               # LLM 响应缓存,key = sha256(model|template_ver|qid|prompt_hash)
└── out\
    ├── task_c_slip_nc.parquet
    ├── task_c_slip_nc_marked.parquet
    ├── north_carolina\entities_*.parquet
    ├── pilot_<model|nomodel>_<template_ver>[_hard][_wzonly]_seed<N>.parquet
    └── conformal_<pool|pilot>_tau<τ>.parquet
```

典型流程:
```powershell
python mark_censored.py                                    # 1. 清洗 → marked
python group_stats.py                                      # 2. 看分组结构
1..5 | % { python runner.py --template lookup --stratify --seed $_ }   # 3. 基线(免费)
1..5 | % { python runner.py --template knn    --stratify --seed $_ }
1..5 | % { python runner.py --template warm2  --stratify --seed $_ }   # 4. LLM
# 第二步(正式):困难层、去 no-slip、T=0、10 seeds、与 kNN 同题配对
1..10 | % { python runner.py --template knn   --stratify --no-noslip --seed $_ }
1..10 | % { python runner.py --template warm3 --stratify --no-noslip --model claude-sonnet-4-6 --seed $_ `
   --compare-to "C:/wzdx/out/pilot_nomodel_knn30_v1_hard_wzonly_seed$_.parquet" }
python compare_runs.py --a "C:/wzdx/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" `
   --b "C:/wzdx/out/pilot_nomodel_knn30_v1_hard_wzonly_seed*.parquet" --label-a sonnet --label-b knn
# 第三步:conformal 闸门
python conformal.py --source pool  --tau 7  --alpha 0.1        # 也跑 --tau 14 / 30
python conformal.py --source pilot --files "C:/wzdx/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" --tau 7
```

---

## 10. 是否够写一篇 benchmark 短论文

**已经具备的部分**(通常是 benchmark 论文的主体):
- 真实 DOT 数据、明确任务定义、可复现的标签清洗规则及其数据证据(§3)
- 数据集结构分析与条件基率(§4),含一个反直觉发现(周期排期语言无增量)
- 分层设计(困难层)与其必要性的证据(随机抽样六成送分)
- 三条非 LLM 基线(查表、kNN)+ 三种 LLM prompt 的多 seed 配对对比(§6)
- 一个干净的负结果(小模型 = 查表)和一个正向线索(文本有信号)

- 第二步:两个模型 × 检索增强 vs 纯检索,217 题配对,带显著性检验(§6.4)
- 第三步:决策层 + conformal 闸门,1,892 条上保证验证、召回天花板、精确率提升、LLM 无增量(§6.5)

**发布前建议补的**:
- 校准样本量 → FPR 稳定性扫描(附录)
- 换 ID 续命写入 limitation,并引用 sonnet 识别同一工程的定性例子
- (可选)τ=7 下 s_combo_knn 的 α–召回曲线图、三条 baseline 的信息量排序图

**建议的论文结构**:
1. 引言:TMC 决策支持需要"带保证的提醒",而非点预测
2. 数据与标签校验(§2–3)
3. 基准设计:任务、分层、指标、基线(§1、§4、§5)
4. LLM 是否读出了文本信号(§6.1–6.4):答案是"没有超过最近邻"
5. Conformal 闸门(§6.5):保证成立、天花板、精确率提升、置信度不可用
6. 限制与 future work(§8)

**核心卖点**:(a) 真实 DOT 数据上的标签校验方法;(b) 一个负结果——LLM 在此任务上不超过 TF-IDF 最近邻,且自报置信度无信息;(c) 一个正结果——不依赖 LLM 的 conformal 闸门在真实数据上给出可验证的误报率保证,并量化了首次观测信息下的召回天花板。
