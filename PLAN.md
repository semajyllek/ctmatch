# ctmatch Modernization Plan

Regression gate: SciBERT baseline NDCG@10=0.6525 (TREC21+KZ, sim+svm+clf).
All changes must match or beat this on the same eval split before merging.

---

## Status legend
- `done` — complete and on main
- `in progress` — active work
- `blocked` — waiting on something
- `todo` — not started

---

## 1. Better classifier — BioLinkBERT-large

**Goal:** replace SciBERT (biased, 80% predictions = relevant) with a calibrated cross-encoder.

| Task | Status | Notes |
|---|---|---|
| Build `retrain_classifier.ipynb` with focal loss + class weights | `done` | FocalLossTrainer, DataConfig/TrainConfig dataclasses |
| Train BioLinkBERT-large; standalone eval NDCG@10=0.9307 (TREC21) | `done` | clf_results.json on Drive |
| Fix `classifier_filter` to rank by P(relevant) not P(not_relevant) | `done` | `relevant_col_index()` reads id2label; commit 0777e6f |
| Fix hub push (local model ≠ hub model) | `in progress` | hub model has no discrimination; re-push from local Drive checkpoint |
| Confirm pipeline eval NDCG matches standalone (~0.87–0.93 TREC21) | `blocked` | waiting on hub fix + Colab rerun |
| Update eval_baseline regression gate to BioLinkBERT-large number | `todo` | |

---

## 2. Lab value extraction integration

**Goal:** wire the existing ctproc `lab/` extractor into the pipeline as a hard pre-filter.
A patient with Hgb=8 should be automatically excluded from trials requiring Hgb≥10
before any embedding or classifier work happens.

The extractor (`ctproc/lab/extractor.py`, `patterns.py`, `reference_ranges.py`) already
parses constraints like "Hemoglobin ≥ 10 g/dL" from eligibility criteria text.
It is not yet called anywhere in ctmatch.

| Task | Status | Notes |
|---|---|---|
| Audit lab extractor: what does it parse, what does it miss? | `todo` | Read extractor.py + patterns.py; build a test set of criteria strings |
| Add patient-side lab extraction (topic text → lab values dict) | `todo` | Mirror the criteria parser; topics say "Hgb 8.2 g/dL", "creatinine 1.4" |
| Write `lab_filter(pipe_topic, doc_set)` in pipeline.py | `todo` | Hard exclude docs where patient demonstrably fails a threshold criterion |
| Eval impact: NDCG delta vs. baseline on TREC21 | `todo` | Expect FPR ↓, possible small NDCG ↑ |
| Handle uncertain/missing lab values correctly (skip, not exclude) | `todo` | Missing lab ≠ failing the criterion |

---

## 3. Replace gen_model with open-weight model + DSPy

**Goal:** remove the paid Claude/OpenAI dependency from the gen filter.
Current: `gen_model_checkpoint = 'claude-opus-4-8'` in PipeConfig.

| Task | Status | Notes |
|---|---|---|
| Choose open model for gen filter (Mistral-7B-Instruct, Llama-3-8B-Instruct, etc.) | `todo` | Must run on Colab T4; test latency vs. quality tradeoff |
| Replace `GenModel` prompt with DSPy `dspy.Predict` or `dspy.ChainOfThought` | `todo` | DSPy already in `train` extras in pyproject.toml |
| Optimise DSPy prompt on a small TREC21 dev set | `todo` | Use a held-out 10-topic split; don't touch the eval set |
| Eval: gen filter NDCG delta vs. classifier-only | `todo` | Establish whether gen filter adds value at all |

---

## 4. RLHF / DPO on clinical judgments

**Goal:** fine-tune a smaller cross-encoder on pairwise preference data derived from TREC qrels
and Claude-generated synthetic pairs. Use TRL (already in `train` extras).

| Task | Status | Notes |
|---|---|---|
| Build preference dataset from TREC qrels (rel=2 > rel=1 > rel=0 pairs) | `todo` | ~40K pairs from existing clf dataset |
| Generate hard synthetic pairs with Claude (cases where model is wrong) | `todo` | Use `evaluate_detailed()` output to find errors; prompt Claude to write contrasting patient descriptions |
| DPO fine-tune BioLinkBERT-large (or smaller model) with TRL | `todo` | `trl.DPOTrainer`; add cell to retrain_classifier.ipynb |
| Eval: DPO model NDCG vs. SFT-only baseline | `todo` | |

---

## 5. Criterion-level entailment (longer term)

**Goal:** score each inclusion/exclusion criterion individually rather than treating
the full eligibility text as a single document. TrialGPT-style but with open models.

See `docs/deep_dive_outline.md §9a` for detailed notes on why naive NLI per criterion
fails (missing information, vague thresholds, temporal reasoning, partial eligibility).
The aggregation function — how uncertain criterion-level signals combine into a
trial-level score — is the core modeling challenge, not the per-criterion NLI itself.

| Task | Status | Notes |
|---|---|---|
| Audit ctproc criteria parser output quality on TREC21 trials | `todo` | What fraction parse correctly into inc/exc lists? |
| Choose NLI model for per-criterion scoring | `todo` | MedNLI-trained model; or few-shot with open LLM |
| Build aggregation model (criterion scores → trial relevance) | `todo` | Start with simple heuristics; then learned aggregator |
| Use TrialGPT criterion annotations as training signal | `todo` | `ncbi/TrialGPT-Criterion-Annotations`, 1020 rows |
| Eval vs. BioLinkBERT-large pipeline | `todo` | |

---

## Ablations / eval debt

These are eval runs needed to fill gaps in `docs/deep_dive_outline.md §7`.

| Task | Status |
|---|---|
| svm+clf ablation (drop sim filter) | `todo` |
| TREC21 + TREC22 + KZ combined baseline table | `todo` |
| Quantify KZ corpus overlap: % of KZ qrels NCT IDs in 2021 corpus | `todo` |
| MedCPT embedding swap (re-embed corpus, re-run eval) | `todo` |
| Measure inference latency per config (GPU T4, seconds/query) | `todo` | Fills efficiency table in deep_dive_outline §7a |
