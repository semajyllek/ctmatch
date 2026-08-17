# notebooks/

Colab notebooks. Each starts with the same 3-cell setup (`pip install git+…ctmatch`, mount Drive,
`from ctmatch.experiments import …`) and drives an experiment through `ExperimentConfig`. Data lives on
Google Drive (`ct_data23/`), not in the repo.

**The 17 notebooks here are canonical** — they build the reproducible pipeline and the headline results.
Everything else (diagnostics, bake-offs, documented negatives) is under **`experiments/`** — real
provenance, off the critical path. Genuinely-dead notebooks are in `archive/`.

## The pipeline (in dependency order)

The SOTA-competitive open ensemble (deep-dive §7j / §12). Each step's output feeds the next, all on the
frozen representation `R = elig_first-L512`.

| # | Notebook | Role |
|---|---|---|
| 1 | `build_fulltext_corpus` | Fetch trial fields → `doc_fulltext.jsonl` (the field source every representation is built from) |
| 2 | `build_dataset` | Topic splits (train TREC21+KZ / test TREC22) + labeled (topic, doc) pairs |
| 3 | `exp_truncation` | **Freezes the representation constant** `R` — every downstream model inherits it |
| 4 | `finetune_retriever` | Dense retriever on `R` |
| 5 | `retrain_classifier` | Eligibility-view cross-encoder `clf-R` |
| 6 | `train_classifier_topic` | Topicality-view cross-encoder (the diverse second reranker) |
| 7 | `train_reranker_hardneg` | Hard-negative reranker `reranker-v3` |
| 8 | `rerank_llm_feature` | Open Qwen-2.5-7B yes/no judge feature |
| 9 | `eval_fullcorpus` | Hybrid BM25+dense retrieval → RRF pool → recall + oracle (writes `pool_R.json`) |
| 10 | `train_ensemble_full` | LambdaMART over all features → the full-corpus NDCG@10 number |
| 11 | `eval_baseline` | Original 4-stage cascade baseline on `R` |

## Funnel + frontier labeler (the wrap-up, deep-dive §8b–§8f)

The cheap-funnel → Opus-labeler investigation. See `docs/project_wrapup.md`.

| Notebook | Role |
|---|---|
| `eval_funnel_recall` | Per-stage recall waterfall; funnel preserves the 0.90 top-10 ceiling (§8b) |
| `label_pointwise_opus` | Three-arm gate: clf-v4 vs Opus no-CoT vs CoT (§8c) — labeler beats clf +0.15 held-out |
| `label_scored_trec21` | Continuous-score labeler; localizes the false-positive ceiling (§8e) |
| `reflect_distill_trec21` | Error-reflective prompt distillation — the confirmed negative (§8f) |

## External held-out test (deep-dive §13)

| Notebook | Role |
|---|---|
| `build_corpus_2023` | TREC 2023 corpus + topics (blind external set) |
| `eval_external_2023` | Frozen pipeline, one-shot external evaluation |

## `experiments/` (26) and `archive/`

`experiments/` holds documented but off-critical-path work, grouped by line: diagnostics
(`diagnose_*`, `exp_*_diagnose`, `exp_validate_clf`, `exp_ablation`), retriever/reranker bake-offs
(`exp_retriever_*`, `exp_reranker_bakeoff`, `exp_retrieval_repr`), the adaptive-gate line
(`exp_adaptive_gate*`, `exp_softgate`), topicality (`rerank_condition_match`, `rerank_topicality_feature`),
NQS query expansion (`nqs_retrieval`, `rerank_llm_nqs`), monoT5 adaptation (`finetune_monot5_ct*`),
the fine-tuned judge (`finetune_judge_lora`, `regen_judge_feature`), synthetic data (`gen_synthetic_pairs`),
and the documented negatives/ceilings (`rerank_criteria_claude` §7g, `rerank_listwise` §11d,
`rerank_cot_final` §11a). `archive/` is genuinely superseded scratch.
