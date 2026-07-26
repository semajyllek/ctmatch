# Error Analysis — Full-Corpus Ensemble Pipeline (TREC 2022)

Source: `eval_predictions_ensemble.jsonl` (dumped by `train_ensemble_full.ipynb`), 50 TREC 2022 topics,
7,632 records: the top-20 ranked trials per topic plus every judged-relevant trial tagged by where it
landed (`retrieval_miss` = never entered the candidate pool; `ranking_miss` = in the pool but ranked
below 10). Each record carries the 9 feature values, so errors are attributable to a stage/feature.

**Headline: the binding limitation is the RERANKER, not retrieval — which reverses the going assumption.**

---

## 1. Where the relevant trials end up

Across 50 topics there are 6,975 judged-relevant (rel≥1) trials (3,939 *Eligible* rel=2, 3,036
*Excluded* rel=1):

| Outcome | count | % of relevant |
|---|---|---|
| in top-10 | 343 | 4.9% |
| ranking miss (in pool, rank > 10) | 3,686 | 52.8% |
| retrieval miss (never in pool) | 2,946 | 42.2% |

At first glance retrieval loses a huge share (42%). **But this does not bind NDCG@10**, because of
surplus: after retrieval, a mean of **48 *Eligible* (rel=2) trials remain in the pool per topic**
(median 29) against only **10 top-10 slots**. The reranker has far more eligible trials available than
it can place — so losing 42% to retrieval still leaves ~5× more eligible trials than slots. The oracle
confirms it: the retrieved pool supports NDCG@10 = 0.957, but the pipeline extracts 0.61. **The gap is
the reranker's, not retrieval's.** (The large `ranking_miss` count is partly structural — with ~140
relevant trials/topic and 10 slots, most relevant trials *must* fall below rank 10 even for a perfect
system. The meaningful signal is top-10 precision and the oracle gap, below.)

## 2. Top-10 precision — ~31% of slots are wasted

Per-topic the top-10 averages **4.9 Eligible + 1.9 Excluded + 3.1 Not-Relevant**. So **157 of 500 top-10
slots (31%) are false positives** (rel=0), and another 96 are Excluded (rel=1, half credit) — while ~48
Eligible trials sit unused in the pool. Closing this precision gap is the single largest available lever.

## 3. The LLM feature drives the top-10 — and its false negatives are the main ranking-miss mechanism

`llm_yesno` (Qwen eligibility logit) is the sharpest discriminator in the ensemble. Mean values:

| group | llm_yesno |
|---|---|
| Eligible (rel=2) **in top-10** | **+5.78** |
| Eligible (rel=2) **ranking-missed** | **−11.51** (median −2.33) |
| Not-relevant (rel=0) false positive | −2.87 |

The trials the pipeline surfaces are the ones the LLM says "yes" to (+5.78); the eligible trials it
**buries are the ones the LLM says "no" to** (−11.51). So the dominant ranking-miss cause is **LLM false
negatives** — Qwen judging a genuinely eligible trial as ineligible. Improving the LLM judge (stronger
model, chain-of-thought, calibration) directly attacks the largest error class.

## 4. The Eligible-vs-Excluded misalignment (a metric/objective mismatch)

The LLM answers *"is the patient likely **eligible**?"*, but NDCG rewards *topical* relevance — *Excluded*
trials (rel=1: right condition, but an exclusion criterion applies) should still rank above Not-Relevant.
The data shows the tension directly, for Excluded (rel=1) trials:

- rel=1 **in top-10**: llm_yesno **+3.81** — the LLM *missed* the exclusion, called them eligible, so they surfaced.
- rel=1 **ranking-missed**: llm_yesno **−14.53** — the LLM *correctly* caught the exclusion and buried them.

So the LLM is penalized both ways by the graded metric: catching an exclusion (correct clinically) pushes
a rel=1 trial down when the metric wants it up. This is a fundamental objective mismatch between an
*eligibility* judge and a *topical-relevance* metric, and it caps how much the LLM feature alone can help.

## 5. False positives = cross-encoders overriding the LLM (and LLM coverage gaps)

FPs (rel=0 in top-10) vs true positives (rel=2 in top-10), mean features:

| feature | FP (rel=0) | TP (rel=2) |
|---|---|---|
| llm_yesno | −2.87 | +5.78 |
| clf_rel | **0.248** | 0.132 |
| v2_rel | **0.149** | 0.048 |
| dense_rank | 297 | 200 |

False positives get into the top-10 **despite a negative LLM score**, carried by *higher* cross-encoder
scores (clf_rel, v2_rel) than the true positives have. Two implications:
- The **cross-encoder features are miscalibrated/noisy** — they score higher on false positives and on
  buried-relevant trials than on the trials actually surfaced. They add noise the ensemble partly trusts.
- **28 of 157 FPs (18%) were beyond the LLM's top-500** (floor'd `llm_yesno`), so the LLM never got to
  veto them. Widening LLM coverage (or a cheap LLM pass on the reranked head) would suppress those.

## 6. Retrieval misses concentrate on implicit-diagnosis / terse-symptom topics

Worst pool-recall topics (recall ≈ 0.21, vs 0.60 mean) are all **symptom-presentation topics where the
diagnosis is implicit**:

| topic | recall | presentation |
|---|---|---|
| 28 | 0.21 | 23F, nosebleeds 1h (→ bleeding disorder) |
| 41 | 0.21 | 61M, acute vision disturbance |
| 43 | 0.21 | 27F, skin rash + oral ulcers (→ Behçet's/lupus) |
| 32 | 0.21 | 30M, azoospermia lab result |
| 10 | 0.23 | 19F, wrist mass |
| 11 | 0.29 | 63M, weight loss + epigastric pain |

The patient presents with **symptoms**, but trial text names **diseases** — so both BM25 and the dense
retriever (which encode surface text) miss the trials for the unnamed condition. This is the same failure
mode as the KZ one-liners (§7i): terse, implicit-diagnosis inputs. Query expansion / LLM diagnosis
synthesis before retrieval would raise recall on exactly these topics — though per §1 this is a
secondary lever for NDCG@10 given the current surplus.

---

## Where the limitations lie (prioritized)

1. **Reranker precision at the top (largest lever).** 31% of top-10 slots are non-relevant while ~48
   eligible trials sit in the pool; oracle 0.957 vs extracted 0.61. This is the binding constraint.
2. **LLM judge quality.** LLM false negatives are the main ranking-miss mechanism; the LLM is also the
   feature that actually surfaces true positives. A stronger/CoT/calibrated LLM judge is the most direct
   attack on #1. (18% of FPs also come from *un-scored* deep candidates → widen LLM coverage.)
3. **The Eligible-vs-Excluded objective mismatch.** An eligibility judge is structurally misaligned with
   the graded topical metric on the Excluded class; a two-headed signal (topicality *and* eligibility)
   or metric-aware aggregation could recover the rel=1 tier.
4. **Cross-encoder calibration.** clf_rel/v2_rel score higher on FPs and buried-relevant than on true
   positives — they inject noise; recalibration or a ranking-loss retrain (§9b) may help, or down-weighting.
5. **Retrieval on implicit-diagnosis topics (secondary for NDCG@10, primary for recall/KZ).** Query
   expansion / diagnosis synthesis for terse symptom presentations.

**Net:** the earlier assumption that "retrieval recall is the ceiling" is *not* supported by this data.
Retrieval loses many relevant trials but leaves a large eligible surplus; the pipeline's NDCG@10 is
gated by the **reranker's top-10 precision**, dominated by the **LLM judge's false negatives** and an
**Eligible-vs-Excluded objective mismatch**, with a secondary contribution from **miscalibrated
cross-encoder features** and **LLM coverage gaps** on the deep pool.
