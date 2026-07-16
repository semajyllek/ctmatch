# Notebook cleanup & re-run plan (2026-07-14)

Goal: one clean, config-controlled experiment set that produces **representation-tagged** numbers,
after the §2g representations audit invalidated the prior scores. Backbone is `src/ctmatch/experiments.py` (installed with the package)
(one `ExperimentConfig`, small portable functions, one frozen representation `R`).

**Ground rule:** nothing is hard-deleted yet. Proposed removals move to `notebooks/archive/` (reversible)
only after you approve this list. Rewrites keep the notebook name but replace the body with clean cells
that call `ctmatch.experiments` functions and take an `ExperimentConfig`.

## The clean target set (the "good first state")

Ordered by dependency — each step's output feeds the next, all on the frozen `R`:

| # | Notebook | Role | Status |
|---|---|---|---|
| 0 | `src/ctmatch/experiments.py` | config + portable functions (backbone, in-package) | **built** |
| 1 | `exp_truncation.ipynb` | **freeze the representation constant** (strategy × max_length, evaluated in isolation) | **built (scaffold)** |
| 2 | `build_fulltext_corpus.ipynb` | fetch trial fields → `doc_fulltext.jsonl` (the field source every repr is built from) | keep, light rewrite |
| 3 | `build_dataset.ipynb` | topic splits + training pairs, joined to fields (NOT eligibility-only) on `R` | **rewrite** (removes the R1 join, §2g) |
| 4 | `finetune_retriever.ipynb` | retriever fine-tune **on `R`** → retriever-v3, **contingent**: run an off-the-shelf-vs-fine-tuned ablation on `R` and keep whichever wins (off-the-shelf is the default if it ties/beats — simpler). The prior +18% was on the confounded repr, so it's re-open. | **rewrite** |
| 5 | `retrain_classifier.ipynb` | clf fine-tuned **on `R`** → clf-v5 | **rewrite** |
| 6 | `train_reranker_hardneg.ipynb` | hard-neg reranker **on `R`** → reranker-v3 | **rewrite** |
| 7 | `rerank_llm_feature.ipynb` | Qwen judge feature **on `R`** (sees eligibility); absorbs the old standalone `rerank_llm` | **rewrite + merge** |
| 8 | `eval_fullcorpus.ipynb` | retrieval + oracle + rerank harness on `R` | **rewrite** |
| 9 | `train_ensemble_full.ipynb` | features + LambdaMART + eval on `R` | **rewrite** |
| 10 | `eval_baseline.ipynb` | 4-stage cascade on `R` with **fine-tuned embeddings** (sim/SVM re-pointed) | **rewrite** (keep-and-fix, per decision) |

External-baseline & external-test set (keep, rewrite to `ctmatch.experiments` when touched):
`reproduce_h2oloo.ipynb`, `finetune_monot5_ct.ipynb`, `build_corpus_2023.ipynb`, `eval_external_2023.ipynb`.

Re-run-under-`R` (were confounded by the eligibility-blind representation — clean re-run may change the verdict):
`finetune_judge_lora.ipynb` (§11c), `rerank_listwise.ipynb` (§11d), `rerank_cot_final.ipynb` (§11a).

## Proposed removals → `notebooks/archive/` (need your OK)

| Notebook | Why archive |
|---|---|
| `train_ensemble_ltr.ipynb` | superseded by `train_ensemble_full` (its own header says so) |
| `rerank_llm.ipynb` | standalone §7f judge — merged into `rerank_llm_feature` (one judge notebook) |
| `reembed_corpus.ipynb` | ad-hoc re-embed — now `encode_corpus()` in `ctmatch.experiments` |
| `hf_download_test.ipynb` | download debugging scratch (see [[hf_xet_colab_download_stall]]) |
| `annotate_criteria_r1.ipynb` | criterion path (Tier 3b) — shelved (§9d) |
| `dataprep_criteria.ipynb` | criterion parsing — shelved |
| `rerank_criteria.ipynb` | criterion rerank (Qwen) — shelved |
| `train_criterion_clf.ipynb` | distilled criterion CE (Tier 3b) — shelved |

## Archive-but-keep (documented findings, not on the critical path)

`rerank_criteria_claude.ipynb` (§7g Claude ceiling test — a documented result), `gen_synthetic_criteria.ipynb`
(built, parked; may be repurposed for judge-data augmentation, §9e), `lab_filter_audit.ipynb` (§9c future work).

## What "rewrite clean" means (applies to every rewrite above)

- **Standard 3-cell Colab setup**, matching the existing notebooks (see `exp_truncation.ipynb`):
  (1) `!pip install -q git+https://github.com/semajyllek/ctmatch.git` + deps;
  (2) `from google.colab import drive; drive.mount('/content/drive')`;
  (3) env (`HF_HUB_DISABLE_XET=1`, local `HF_HOME`) + `from ctmatch.experiments import ...`.
  The backbone ships **inside the ctmatch package** (`src/ctmatch/experiments.py`), so the `pip install
  git+...` in step (1) provides it — no Drive copy, no `sys.path` hacks. (During dev, install a branch:
  `pip install git+https://github.com/semajyllek/ctmatch.git@<branch>`.)
- Config cell: `cfg = ExperimentConfig(data_root=DATA_ROOT, ...)`. No other magic paths/constants anywhere.
- Each cell does one thing and calls a small named function from `ctmatch.experiments` (so the logic is
  portable and unit-testable outside Colab). No 80-line mega-cells.
- Every result written via `log_result(cfg, ...)` so it lands in the ledger with `repr_tag` + fingerprint.
- No number is reported without its `repr_tag`.

## Principle: representation is mandatory, fine-tuning is not

The binding requirement is that every model sees the **same frozen `R`**. Whether a component is
**fine-tuned or off-the-shelf is a separate, empirical choice decided on clean `R`** — default to the
simpler off-the-shelf when it ties or wins (no training, fewer moving parts). This applies to the
retriever (step 4) and the LLM judge (`finetune_judge_lora`): both fine-tune "wins" in the old doc were
measured on the confounded representation and must be re-run before they count. Report both rows; keep
what actually helps on `R`.

## Sequencing

1. **Run `exp_truncation.ipynb` first** and freeze `ExperimentConfig` defaults (`repr_strategy`, `max_length`,
   `head_frac`). Everything downstream inherits them, so this must be decided before any model is retrained.
2. Then steps 2→10 in order (retriever/clf/reranker retrains depend on the frozen `R`).
3. Re-run the confounded negatives (§11a/c/d) under `R` and update §11 verdicts.
4. The backbone already lives in the package (`src/ctmatch/experiments.py`); keep growing it there as
   notebooks are rewritten, and commit/push so Colab's `pip install git+...` picks it up.
