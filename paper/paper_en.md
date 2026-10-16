# A Label-Validated Benchmark for Work-Zone End-Date Slip in WZDx Feeds: Do LLMs Read More than a Nearest Neighbor, and Can Alerts Carry a False-Alarm Guarantee?

**Authors:** [TODO]
**Affiliation:** [TODO]
**Code and data:** [TODO GitHub URL] · [TODO Zenodo DOI]

---

## Abstract

Work-zone data feeds such as WZDx publish an end date for every lane closure, but the posted date is often not when the work actually ends. Traffic management centers (TMCs) would benefit from a tool that flags records likely to slip. Because each record includes a free-text description, large language models (LLMs) are a natural candidate for this task. We build a benchmark from 2.5 years of North Carolina WZDx/TIMS data (6.74M feed entities, 11,108 slip-labelled records) and report three results. First, labels harvested directly from the feed are not usable as ground truth. Batch-process timestamps are recorded as completion times, posted dates fall outside the observation window, some records linger long after the work is done, and long projects are re-issued under new IDs. We give detection rules for each of these problems. Without them, evaluation error exceeds 1,000 days on some records. Second, on a stratified hard subset we compare two LLMs (Claude Haiku 4.5 and Claude Sonnet 4.6) under four prompting regimes, including retrieval of the nearest historical records, against a (posted-window × work-type) lookup table and a TF-IDF nearest-neighbor baseline. On 217 paired questions the small model is significantly worse than the nearest-neighbor baseline (Wilcoxon p<0.001) and the large model is statistically indistinguishable from it (p=0.56). Neither model's self-reported confidence carries information. Third, we recast the prediction problem as the operator-facing decision of whether to raise a slip alert, and wrap it in a split-conformal gate that bounds the false-alarm rate by α. On 1,892 hard-stratum records the bound holds for α=0.1 at three slip thresholds (7, 14 and 30 days). The best non-LLM score reaches about 30% recall at 66% precision against a 37% base rate; scores derived from the LLM do not improve on this; and roughly 250 historical non-slipping records are enough to stabilize the gate. The benchmark, cleaning rules, baselines and gate are released for use on other state feeds.

---

## 1 Introduction

The Work Zone Data Exchange (WZDx) specification allows state DOTs to publish machine-readable work-zone records containing road, direction, a free-text description, and posted start and end times. Downstream consumers, including navigation applications, freight planners and TMC operators, treat the posted end time as a forecast. In practice it is a permit or shift window that is frequently extended. A closure posted for five nights can persist for six months through successive renewals.

An operator who could identify, when a record first appears, which closures are likely to slip could reroute traffic earlier and plan staffing more accurately. Since every record carries prose such as "Nightly lane closure 7pm to 6am for bridge deck rehabilitation", it is natural to ask an LLM to read the description and predict the true end date. A recent survey of LLMs in transportation systems management and operations (TSMO) identifies operator-facing decision support as the most promising role for these models, and notes the lack of benchmarks built from real DOT data [TODO: cite survey]. In multi-agent robotics, it has been shown that LLM outputs should pass through a calibrated statistical gate before they are acted on, with conformal prediction supplying a distribution-free bound on the error rate [TODO: cite CommCP].

This paper tests whether those two ideas hold up on a real state feed. Our contributions are as follows.

1. **A label-validated benchmark.** We identify four classes of label defect that arise when a dynamic feed is harvested by polling, give a detection rule for each, and quantify their effect on evaluation. We define the task, a hard stratum, metrics suited to the heavy-tailed slip distribution, and two inexpensive non-LLM baselines that any proposed method should be compared against.
2. **A scoped negative result on LLMs.** On the hard stratum, using paired questions and significance tests, neither a small nor a large LLM extracts more from the description than a TF-IDF nearest neighbor over historical records, even when the LLM is given those neighbors as context. Self-reported confidence is uninformative.
3. **A conformal alert gate evaluated on DOT data.** Recasting the task as an alert decision, we calibrate a threshold on historical non-slipping records and verify that the false-alarm bound holds out of sample. We measure the cost (recall is limited to about 30% at first observation), the benefit (precision nearly doubles relative to the base rate), and the calibration data required (a few hundred records). Because LLM confidence is uninformative, the gate's nonconformity score must come from outside the model; nearest-neighbor slip statistics serve this purpose.

The negative result applies to a feed whose descriptions are short and templated. Section 8 discusses which conclusions we expect to transfer to other feeds.

---

## 2 Data

**Source.** We use the North Carolina WZDx feed, which is generated from NCDOT's Traveler Information Management System (TIMS) and published through one.network. The archive was collected by USDOT ITS JPO by periodically polling state WZDx feeds between October 2019 and August 2024, with per-state coverage depending on feed activation dates; the earliest North Carolina record in the archive was first observed on 2022-01-18. Observed snapshot intervals in the North Carolina feed are approximately 15 minutes. We characterize the data as observed; polling configuration details were not distributed with the archive. Each snapshot contains the set of currently active records, and a record is identified by its feed entity ID.

**Entities.** Collapsing polls by entity ID yields 6,739,847 unique entities, each with first and last observation times, first and last posted end dates, road, direction, description and an incident flag. The count is large because NCDOT re-issues records frequently; the same physical closure typically appears under many IDs over its lifetime (Section 8).

**Slip-labelled subset (`task_c`).** From the entity table we select entities whose posted end date changed at least once between first and last observation (`first_end_date ≠ last_end_date`) and whose dates parse. Four filters are then applied: (R1) |slip| ≤ 365 days, which removes a small number of records with implausible multi-year revisions; (R2) the entity was observed in at least two polls; (R3) the description has at least 10 characters; (R4) entities sharing the same creation timestamp and description are collapsed to one, which removes bulk-import duplicates. The result is 11,108 records, of which 5,726 are work zones (non-incident). For each record the *input* is the snapshot at first observation: road, direction, description, first-seen time, and the posted end date at that time (`promised_end`). The *label* is the last posted end date recorded for the entity (`actual_end`), and the target is `slip_days = actual_end − promised_end`. We note that the label is the final value of the posted date, not the time at which the record left the feed. In TIMS the two usually coincide, because closing an event writes the current time into its end date; the batch-timestamp defect in Section 3 is a direct consequence of this behaviour. The entity table also records an ID era for each entity: NCDOT entity IDs are 7-digit through 2022 and 9-digit from 2023, indicating an upstream system change. The two eras differ in composition (the 7-digit era has a higher share of incidents; the 9-digit era contains many bulk "Road closure" records). [TODO: Table 2b, slip distribution and description length by ID era.] This within-feed drift is one reason to expect the findings of Sections 6 and 7 to be feed- and period-specific (Section 8). No entity linking across IDs is performed.

**Two populations.** `task_c` is enriched for slip by construction: it contains only records whose posted end changed at least once, and the complementary set (`first_end_date == last_end_date`) is the no-slip population. In the full feed, 1,371,321 entities (after the filters of Section 4) never changed their posted end. These are different populations, and a base rate estimated on one does not apply to the other. All formal results below are on `task_c`. Section 4 explains why a mixed design used in early pilots was dropped.

**Observation window.** The last observation in the entity table is 2024-08-06 (`obs_end`). A record whose posted or actual end falls after `obs_end − 30 d` has not had its outcome observed and is excluded from evaluation.

---

## 3 Label validation

Our first pilot (Section 6.1) produced errors above 1,000 days on some records. Tracing them showed that the defects were in the labels rather than the model. We found four classes of defect. Each is a general hazard of polling a dynamic feed, and each has a reproducible detection rule (Table 1).

**Table 1. Label defects and detection rules.**

| # | Defect | Symptom | Rule | Records flagged |
|---|---|---|---|---|
| 1 | Batch timestamp recorded as completion | Dozens of `actual_end` values within the same second or adjacent seconds (e.g. 2022-06-04 17:11:56/57/58; 2022-10-01 04:06:02) | (a) exact timestamp occurs ≥5 times and has non-zero seconds; (b) any non-zero-second timestamp within ±120 s of such a cluster center | 80 + 301 |
| 2 | Record dropped before its posted end | slip < −60 d | flag as censored | +10 |
| 3 | Posted end outside the observation window | posted end in 2027; `actual_end == promised_end`; slip exactly 0 | require posted and actual end < `obs_end − 30 d` | excluded at sampling |
| 4 | Stale record | a one-night task remains in the feed for 150 days | posted − first_seen < 1 d and slip > 30 d; reported separately | small |

Two refinements are needed. Timestamps on the hour (`HH:00:00`) and at `HH:59:59` are NCDOT scheduling conventions rather than machine times, and are exempted from rule 1 even when they occur frequently; without this exemption, 19 legitimate planned end times would be flagged. Rule 1(b) is required because a single batch job spreads its writes over tens of seconds, so any individual second may fall below the frequency threshold.

Flagged records are retained and marked as censored (391 records, 3.5%). They are excluded from error metrics but reported separately, since a record that disappears before its posted end says something about how the feed behaves. After validation, the slip range of valid work-zone records is [−54, +365] days.

A fifth defect is identified but not corrected. Long-running projects are re-issued under new entity IDs, so a record whose description reads "through Spring 2027" can legitimately "end" in September 2022 when its ID is retired. Labels are attached to records; a model reasons about projects. We treat this as a limitation (Section 9).

---

## 4 Benchmark design

**Task.** Given a record's first-observation snapshot, predict the actual end date. Methods output a point estimate, a P90 upper bound, a confidence in [0, 100], and a one-sentence rationale.

**Population and stratum.** Two features dominate slip (Section 5): the posted window (`promised_end − first_seen`) and the work type. Records with a window of one day or less, or of incident or mowing type, end on time 85 to 95% of the time, so copying the posted date is the correct answer for them. Random sampling therefore hands any method a majority of trivial items. We define the hard stratum as window >1 d and type not in {incident, mowing/trees}. It contains 1,989 of 5,684 eligible work-zone records (35%). All formal comparisons use it. Early pilots mixed 25 `task_c` items with 25 never-changed entities; every method calibrated on `task_c` systematically pushed the latter later, which exposed the two-population problem described in Section 2, and the mixed design was abandoned.

**Metrics.** Slip is heavy-tailed (Section 5). We report the median and 90th percentile of absolute point error; the fraction of predictions that copy the posted date (|pred − posted| < 1 d); for true slips over 7 days, the fraction pushed later by at least 3 days and the median remaining underestimate; P90 coverage (actual ≤ P90) with median interval width; and pinball loss at quantiles 0.5 and 0.9 measured in slip days. Paired comparisons use Wilcoxon and sign tests on per-question error differences.

**Baselines.**
- *Lookup.* The reference class is the pair (window bucket ≤1 / 2–7 / 8–30 / >30 d, work type from a priority-ordered keyword classifier). The point estimate is posted plus the class median slip; P90 is posted plus the class P90. If a class has fewer than 30 records the method falls back to the window bucket alone, then to the global distribution.
- *kNN.* Within the window bucket, the 30 historical descriptions nearest in TF-IDF (1–2-gram) cosine similarity are retrieved; point and P90 are taken from the neighbors' slip distribution.

Both baselines exclude the evaluated record from their statistics.

**Protocol.** Ten seeds of 25 hard-stratum items each; identical items across methods; temperature 0; responses cached by (model, template, item, prompt hash).

---

## 5 The structure of slip

Among the 5,429 valid work-zone records, 60.0% end within one day of the posted date, 18.5% end more than 7 days late and 6.6% more than 30 days late. The slip distribution has median +0.4 d, P90 +18 d and P95 +38 d.

**Table 2. Slip by posted window and by work type (valid work zones).**

| Group | n | On time ±1 d | >7 d late | P90 slip |
|---|---|---|---|---|
| window ≤1 d | 3,194 | 84.4% | 5.7% | +3 |
| window 2–7 d | 1,517 | 26.6% | 36.1% | +37 |
| window 8–30 d | 620 | 24.0% | 38.1% | +36 |
| window >30 d | 98 | 9.2% | 40.8% | +160 |
| incident | 2,187 | 87.7% | 4.9% | +1 |
| construction | 871 | 50.7% | 22.3% | +27 |
| maintenance | 671 | 43.1% | 24.0% | +15 |
| bridge | 538 | 24.7% | 32.5% | +34 |
| ramp | 188 | 42.0% | 33.5% | +36 |
| paving | 115 | 15.7% | 49.6% | +47 |
| pipe/utility | 128 | 22.7% | 45.3% | +42 |
| signal/sign | 76 | 15.8% | 42.1% | +56 |
| guardrail | 49 | 34.7% | 40.8% | +96 (P95 +151) |
| culvert | 42 | 2.4% | 59.5% | +56 |

Three observations follow. First, the posted window is the dominant feature: a one-day window ends on time 84% of the time, a two-day window 27%. This reflects NCDOT practice, in which the posted date is a permit window rather than a project end. Second, work type is the next most informative feature and separates the population cleanly. Third, recurring-schedule language ("nightly 7pm to 6am", "Mon to Fri") carries almost no information once the window is controlled for: within the 2–7 d bucket, P90 is +37 d with such language and +35 d without. An early hand-written prompt rule based on the opposite assumption (Section 6.1) turned out to be wrong. Several types are bimodal (guardrail ends on time 35% of the time yet has P95 +151 d), which makes a point estimate a poor deliverable and argues for interval prediction as the primary output.

---

## 6 Do LLMs extract signal beyond cheap baselines?

### 6.1 Pilot: prompt variants versus lookup (Claude Haiku 4.5)

We first ran five seeds of a mixed 50-item exam with three prompts: *cold* (record only), *warm* (record plus population base rates and hand-written rules, with a P90 requested), and *warm2* (the record's reference-class statistics injected, with the point estimate asked to stay near posted plus class median and tail risk expressed through the P90). On the work-zone half, the cold prompt copied the posted date 58 to 71% of the time, with P90 error of 16 to 31 days. The warm prompt reduced P90 error by 25 to 35%, but did so through the incorrect recurring-schedule rule, which also pushed on-time records later. Under warm2 on the hard stratum, the LLM's P90 error matched the lookup baseline to within 0.1 day on all five seeds. Its rationales read, for example, "applying class median slip of +5.3 days to posted 2022-03-23 gives...". In one case the rationale noted that the "description states continuous closure through 5/25, which overrides class median", yet the prediction did not move. The model carried out the lookup arithmetic and nothing more.

The LLM's self-reported confidence appeared informative on the uncleaned data (high-confidence median error 1 day versus 20 days for low confidence), but the split disappeared after label validation. It had been driven by the model correctly distrusting out-of-window placeholder dates in 2027. On clean data, confidence collapsed to a near-constant value.

### 6.2 Retrieval-augmented LLM versus nearest neighbor, paired

Unlike lookup, the kNN baseline uses the description. On the hard stratum it halves the interval width at similar coverage and reduces the underestimate on true slips on every seed, which shows that the description carries information beyond the type label. To test whether an LLM can add to this, we gave the model the same evidence. The *warm3* prompt lists the eight nearest historical records with their posted windows and realized slips, provides a summary of the neighbors, and explicitly permits the model to move well beyond the neighbor median when the description indicates a long-term project. We ran ten seeds on the hard stratum with Claude Haiku 4.5 and Claude Sonnet 4.6 at temperature 0 and compared each with kNN on identical items.

**Table 3. Paired comparison on 217 hard-stratum items.**

| Comparison | Win / lose / tie (±0.5 d) | Median err | P90 err | Wilcoxon p | Pinball q50 | Pinball q90 | P90 coverage | Width |
|---|---|---|---|---|---|---|---|---|
| Sonnet-warm3 vs kNN | 21 / 17 / 62% | 3.8 vs 3.9 | 29.8 vs 30.9 | **0.56** | 6.93 vs 6.68 | 6.83 vs 6.92 | 88 vs 89% | 16 vs 17 d |
| Haiku-warm3 vs kNN | 19 / 31 / 50% | 5.2 vs 3.9 | 31.9 vs 30.9 | **<0.001** | 7.07 vs 6.68 | 7.42 vs 6.92 | 76 vs 89% | 12 vs 17 d |
| Sonnet vs Haiku | 29 / 13 / 58% | 3.8 vs 5.2 | 29.8 vs 31.9 | **<0.001** | 6.93 vs 7.07 | 6.83 vs 7.42 | 88 vs 76% | 16 vs 12 d |

The small model is significantly worse than the nearest neighbor. Given the neighbors, it narrowed its intervals (coverage fell to 76%) without becoming more accurate, and its domain priors sometimes hurt; for instance it reasoned that "dual bridge replacement typically requires 60 to 90+ days" for a project that finished 23 days early. The large model closes the gap but does not open one. Stratified by true slip, Sonnet and kNN are within 0.2 day of each other at every level except slips over 30 days, where Sonnet's median error is 36.7 d against 42.5 d (n=26, not significant). Sonnet's departures from the neighbor median are high-variance with roughly zero mean. On a bridge-deck rehabilitation that ran 201 days it moved the P90 to +192 d (kNN: +116); on a road-widening record that ended 23 days early it predicted +140 d.

Two qualitative observations are worth recording. Sonnet's rationales repeatedly note that a record "is part of the ongoing I-40 Winston-Salem rehab project"; it recognizes from the neighbors the ID churn described in Section 3, but because labels are attached to records this recognition is not consistently rewarded. Sonnet's confidence shows weak signal on some seeds (high-confidence median 3.0 d versus 21 d) but not on others; Haiku's shows none.

In summary, increasing model size moves the LLM from worse than retrieval to equal to retrieval, but not beyond it. On this feed, the usable information in the description is captured by a bag-of-words nearest neighbor.

---

## 7 A conformal alert gate

### 7.1 Decision and guarantee

Operators act on alerts rather than on predicted dates. We therefore recast the task: for each record, raise an alert if it is predicted to slip by more than τ days, otherwise remain silent. Let s be a nonconformity score, with larger values indicating a record more likely to slip. On a calibration set, take the scores of records that did not slip (slip ≤ τ) and set the threshold q to their ⌈(n+1)(1−α)⌉-th smallest value. Alert when s > q. Under exchangeability this guarantees

P(alert | record does not slip) ≤ α,

that is, a false-alarm rate bounded by α, with no assumption on the distribution of s. This is the transportation counterpart of gating LLM messages behind a conformal confidence threshold [TODO: cite CommCP].

We evaluate on the hard stratum (n=1,892 after the window filter) using 20 to 50 random 50/50 calibration/test splits, and report the realized false-alarm rate (which should not exceed α), the alert rate, recall among true slips, and precision. Each record's scores are computed from all other valid records:

- `knn_frac`, the fraction of the 30 nearest descriptions in the same window bucket that slipped by more than τ; `knn_med` and `knn_p90`, the median and P90 of their slip;
- `rc_gt7` and `rc_p90`, the reference class's fraction slipping by more than τ and its P90;
- `window_days`, the raw posted window;
- `combo`, a cross-fitted logistic combination of the above plus work-type indicators, in which each record is scored by a model that did not see it;
- for the 217 items with LLM output, the LLM's predicted slip, its P90 slip, and 100 minus its confidence.

### 7.2 Results

**Table 4. Gate at α=0.1 on 1,892 hard-stratum records (mean over 20 splits).**

| τ | Positive rate | Best score | Recall | Precision (base) | Mean FPR / max |
|---|---|---|---|---|---|
| 7 d | 36.8% | combo | 32% | 66% (37%) | 9.9% / 12.5% |
| 14 d | 23.5% | knn_med | 29% | 47% (23%) | 10.2% / 15.0% |
| 30 d | 13.3% | knn_p90 | 36% | 34% (13%) | 10.2% / 13.3% |

**The bound holds.** The mean false-alarm rate is at or below α for every score at every τ. The maximum over random splits is 10 to 15%, which is the expected finite-sample dispersion.

**The information ranking is stable.** At α=0.1 and τ=7, kNN-derived scores reach 24 to 32% recall, reference-class scores 8 to 17%, and the raw window 11%. The ordering kNN > reference class > window holds at every α between 0.05 and 0.3 and at every τ. The combined score adds 1 to 2 points at α=0.1 and 5 to 8 points at α≥0.15; its coefficients load almost entirely on `knn_frac`, indicating that the features are largely redundant.

**Recall is limited to about 30%.** Regardless of the threshold definition or the score, at a 10% false-alarm rate the gate recovers roughly one third of true slips. The remaining two thirds are, at first observation, not distinguishable from on-time records. This reflects the information available at that moment rather than the method. The trade-off is quantified by the uncalibrated rule `knn_frac ≥ 0.3`, which recalls 77% of τ=7 slips at a 44% false-alarm rate; the gate reduces false alarms to 10% and raises precision from the 37% base rate to 66%, at the cost of recall. For rarer, larger slips the relative gain in precision is larger (2.6× at τ=30).

**Table 5. Recall versus α, τ=7 d, score = combo.**

| α | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| Recall | 16% | 32% | 44% | 51% | 64% |
| Realized FPR | 4% | 10% | 15% | 20% | 30% |

**LLM scores do not add information.** On the 217 items with Sonnet output (α=0.1, τ=7), the LLM's point prediction used as a score recalls 39%, compared with 34% for `knn_med` and 32% for `knn_frac`; the difference is within the roughly ±10-point noise implied by 82 positives. Adding LLM features to the combined score yields 33% against 32%. Self-reported confidence used as a score plateaus at 17% recall regardless of α, because it takes only a handful of discrete values and cannot rank records. This is the concrete reason the gate cannot be built on LLM confidence in this domain.

**Calibration size.** Sweeping the calibration fraction from 10% to 50% (roughly 120 to 600 non-slipping records), the mean false-alarm rate stays between 8 and 10% throughout, while the worst case over 50 splits falls from 18% at about 120 records to about 12% at 240 or more and then levels off. Recall of the best score rises modestly (28% to 32%) as the finite-sample correction loosens. About 250 historical non-slipping records (roughly 400 hard-stratum records) are sufficient for a stable gate. This also explains why the combined score calibrated on the 217 LLM items (about 70 non-slipping) overshoots α (12%, maximum 27%): non-LLM scores can be calibrated on the full history, whereas LLM scores are limited by inference cost to a few hundred records.

---

## 8 Discussion: what is likely to transfer

We separate the method from the findings.

**Likely to transfer.** The four defect classes of Section 3 arise from polling any dynamic feed; the detection rules transfer, the specific timestamps do not. The conformal bound is distribution-free and will hold on any feed satisfying exchangeability; what changes is recall. The uninformativeness of LLM self-reported confidence is consistent with the broader calibration literature, although we observed it on one feed.

**Likely feed-specific.** The dominance of the posted window reflects a permit-window posting practice. A DOT that posts project-level end dates may show a weaker window effect and richer descriptions. The finding that the LLM does not improve on kNN is the most exposed. NC descriptions are short, system-generated strings ("Night Time Maintenance; Right lane closed") that a bag-of-words model captures adequately; a feed with longer human-written descriptions may favor a language model. The 30% recall limit and all base rates are NC-specific numbers.

**Sequential decisions.** The recall limit arises because we evaluate at first observation. A record that persists past its posted end is itself evidence of slip. A gate re-evaluated daily, with the bound maintained across evaluations, is the natural next step and is closer to per-message gating in multi-agent settings.

**Within-feed drift.** The ID-era split in Section 2 shows that the same feed changes composition over time; a time-ordered calibration/test split for the gate, and a by-era comparison of the LLM results, are the corresponding checks and are left for the revised version.

**Cross-state replication.** WZDx is a federal specification and other state feeds share its schema. Running the full pipeline on a second state would turn each finding above into either a pattern or a stated condition, and is the most valuable extension we can identify.

---

## 9 Limitations

- **ID churn.** Labels are attached to feed records, but slip is a property of projects. Records re-issued under new IDs cause some true slips to be labelled as on-time ends and vice versa. Entity linking by road, description similarity and temporal adjacency is left for future work.
- **Single state and single feed generator.** All numbers are from NC/TIMS. Section 8 lists which conclusions we expect to hold elsewhere.
- **Two LLMs and one prompt family.** We tested Claude Haiku 4.5 and Sonnet 4.6 with four prompts. Other models or prompting strategies (for example chain-of-thought over the neighbors) may behave differently. The paired protocol makes such tests inexpensive to add.
- **Exchangeability.** The gate was evaluated with random splits. A time-ordered split (calibrate on 2022, test on 2023 to 2024) would test robustness to drift.
- **Tolerance in rule 1(b).** The ±120 s tolerance flagged 301 records and may be conservative. Group statistics moved by less than 0.5 percentage points across tolerances, so the conclusions are not affected.

---

## 10 Conclusion

We built a label-validated benchmark for work-zone end-date slip from a real state WZDx feed. Two inexpensive baselines, a reference-class lookup and a TF-IDF nearest neighbor, set a bar that a small LLM does not reach and a large LLM only matches. An alert gate with a distribution-free false-alarm bound is feasible on DOT data: at a 10% false-alarm rate it recovers about one third of true slips with precision nearly double the base rate. The gate does not depend on the LLM and cannot, since the model's confidence is uninformative. The pipeline is released so that the same questions can be asked of other feeds, where the answers may differ.

---

## References

[1] [TODO] Li, J. et al. CommCP: Conformal-calibrated communication for multi-robot embodied question answering. ICRA 2026.
[2] [TODO] Li, J. et al. Large language models for transportation systems management and operations: a survey. 2026.
[3] Vovk, V., Gammerman, A., Shafer, G. *Algorithmic Learning in a Random World*. Springer, 2005.
[4] Angelopoulos, A. N., Bates, S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in Machine Learning*, 2023.
[5] U.S. DOT. Work Zone Data Exchange (WZDx) Specification. [TODO version/URL]
[6] NCDOT. Traveler Information Management System (TIMS). [TODO URL]

---

## Appendix A. Reproduction

```
python mark_censored.py                                    # label validation, writes *_marked.parquet
python group_stats.py                                      # tables in Section 5
1..10 | % { python runner.py --template knn   --stratify --no-noslip --seed $_ }
1..10 | % { python runner.py --template warm3 --stratify --no-noslip --model claude-sonnet-4-6 --seed $_ }
python compare_runs.py --a "out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" \
                       --b "out/pilot_nomodel_knn30_v1_hard_wzonly_seed*.parquet"
python conformal.py --source pool --tau 7 --alpha 0.1      # also --tau 14, 30; --cal-frac 0.1 to 0.5
```

Total LLM cost for all reported experiments was approximately US$1.5.

## Appendix B. Prompt templates

[TODO: paste cold / warm / warm2 / warm3 templates from runner.py]
