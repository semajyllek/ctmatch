# ctmatch Modernization Plan

Regression gate: BioLinkBERT-large NDCG@10=0.7528 (TREC21+KZ combined, 134 topics, sim+svm+clf, local Drive checkpoint).
Previous SciBERT baseline: 0.6525. All changes must match or beat 0.7528 before merging.

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
| Fix hub push (local model ≠ hub model) | `done` | re-pushed from local Drive checkpoint; verified col0=0.096 vs 0.635 on hub |
| Pipeline eval with local checkpoint | `done` | NDCG@10=0.7528, MRR=0.8253 (TREC21+KZ, 134 topics) |
| Error analysis on eval_predictions.jsonl | `done` | See `docs/error_analysis_biolinkbert_pipeline.md`; dominant error: diagnosis specificity FPs |
| Update eval_baseline regression gate to BioLinkBERT-large number | `done` | Updated in this file: gate = 0.7528 |

---

## 2. Lab value extraction integration

**Goal:** wire hard demographic and lab pre-filters into the pipeline before the classifier.
Error analysis identified two concrete, well-scoped sub-problems:
- **Age filter** (quick win): ~28 FP errors where patient age is outside the trial's stated range. Regex-parseable from both topic and trial text.
- **Lab filter**: patient with Hgb=8 excluded from trials requiring Hgb≥10. Requires ctproc lab extractor.

The ctproc extractor (`ctproc/lab/extractor.py`, `patterns.py`, `reference_ranges.py`) already
parses constraints like "Hemoglobin ≥ 10 g/dL" from eligibility criteria text.
It is not yet called anywhere in ctmatch.

| Task | Status | Notes |
|---|---|---|
| **Age filter**: extract patient age from topic text; extract trial age range from criteria; hard-exclude out-of-range | `done` | `src/ctmatch/matching/demographic.py`; 45 tests pass; production pipeline only — see note below |
| Write `demographic_filter(pipe_topic, doc_set)` in pipeline.py | `done` | Wired after svm, before classifier; no top_n (hard exclusion); logs excluded count |

> **TREC eval incompatibility:** TREC qrels define relevance as topical match (right condition/treatment), not strict eligibility. An adult trial (age ≥ 18) can be labeled `relevant` for a 16yo patient if the condition matches. The demographic filter is medically correct but drops these from ranking, collapsing NDCG@10 from 0.75 to 0.39. **Do not include `demographic` in `FILTER_CONFIGS` in eval_baseline.ipynb.** Evaluate the filter via clinical review of removed FP cases, not NDCG.
| Audit lab extractor: what does it parse, what does it miss? | `todo` | Read extractor.py + patterns.py; build a test set of criteria strings |
| Add patient-side lab extraction (topic text → lab values dict) | `todo` | Mirror the criteria parser; topics say "Hgb 8.2 g/dL", "creatinine 1.4" |
| Write `lab_filter(pipe_topic, doc_set)` in pipeline.py | `todo` | Hard exclude docs where patient demonstrably fails a threshold criterion |
| Eval impact: NDCG delta vs. 0.7528 gate | `todo` | Expect FPR ↓, NDCG neutral-to-small gain |
| Handle uncertain/missing values correctly (skip, not exclude) | `todo` | Missing value ≠ failing the criterion |

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
Error analysis (`docs/error_analysis_biolinkbert_pipeline.md`) identified the dominant failure mode:
**diagnosis specificity** — model ranks same-organ-system trials high for patients with a different diagnosis
(e.g., Kawasaki trials for atopic dermatitis, pulmonary fibrosis for COPD). Affects ~120/199 FP errors.
DPO hard negatives should target this cluster first.

| Task | Status | Notes |
|---|---|---|
| Mine hard-negative triples from eval_predictions.jsonl | `todo` | For each FP error (actual=0, pred=2): emit (topic, correct_trial, wrong_trial) triple; priority clusters: KD vs eczema, IPF vs COPD, generic dermatology vs specific diagnosis |
| Build preference dataset from TREC qrels (rel=2 > rel=1 > rel=0 pairs) | `todo` | ~40K pairs from existing clf dataset; combine with hard negatives above |
| Generate synthetic hard negatives with Claude | `todo` | For diagnosis-specificity failures: prompt Claude to write a patient who matches the wrong trial's domain but not its specific inclusion criteria; use eval errors as seeds |
| DPO fine-tune BioLinkBERT-large with TRL | `todo` | `trl.DPOTrainer`; add cell to retrain_classifier.ipynb |
| Eval: DPO model NDCG vs. 0.7528 gate | `todo` | |

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

**Important:** `eval_baseline.ipynb` runs in eval mode — it passes the full judged doc set to
`match_pipeline`, which calls `reset_filter_params` and disables all top-n cutoffs. In this mode
sim/svm/classifier are pure **rankers** (nothing is dropped); final ranking is determined by
whichever stage runs last. Multi-config ablations in eval mode are only meaningful for stages that
hard-filter regardless of top_n (currently: `demographic`; future: `lab`). For production-mode
cascade behavior, use the filter recall task below.

| Task | Status |
|---|---|
| **Filter recall (production mode):** run sim → svm → classifier on all corpus docs (not the judged set); measure fraction of true-relevant docs surviving each stage | `todo` | This is the actual cascade eval; eval_baseline NDCG does NOT measure this |
| svm+clf ablation (drop sim stage entirely) | `todo` | Only meaningful in production mode (see above) |
| TREC21 + TREC22 + KZ combined baseline table | `todo` |
| Quantify KZ corpus overlap: % of KZ qrels NCT IDs in 2021 corpus | `todo` |
| MedCPT embedding swap (re-embed corpus, re-run eval) | `todo` |
| Measure inference latency per config (GPU T4, seconds/query) | `todo` | Fills efficiency table in deep_dive_outline §7a |
| Qrel completeness audit: topics 18 (atopic dermatitis) and 201528 (Lyme arthritis) have all docs labeled not_relevant | `todo` | These contribute NDCG=0; verify whether correct trials are missing from judged pool vs. genuinely no match |
