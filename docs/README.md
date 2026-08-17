# docs/

Where to read, in order — from "what is this" to the full reference.

| Read | Doc | What it is |
|---|---|---|
| **1st** | [`project_wrapup.md`](project_wrapup.md) | **Start here.** The honest close: what was built, the four findings (funnel, labeler-beats-CE, CoT, the reflection negative), positioning against the field ceiling, method spec, results table. ~8 pages. |
| 2nd | [`reflective_labeler_design.md`](reflective_labeler_design.md) | The funnel + frontier-labeler investigation, turn by turn (§8b–§8f): the design, the gate, the oracle diagnosis, the reflection negative and its mechanism. The evidence behind the wrap-up. |
| 3rd | [`error_analysis_ensemble.md`](error_analysis_ensemble.md) | Error analysis of the full-corpus ensemble (where its top-10 mistakes come from). |
| Reference | [`deep_dive_outline.md`](deep_dive_outline.md) | **The appendix / reference-of-record (~280K).** Everything: the data, the representation audit (§2g/§2h), the full pipeline (§5), all experiments and results (§7), error analysis (§8), negative results (§11), the SOTA-ceiling and external-generalization story (§13), and paper assembly (§12). Not a start-here doc — a searchable master reference. It has its own document map near the top. |

`archive/` holds superseded planning docs (an earlier notebook-cleanup plan, an untried-levers review).

## The one-paragraph version

An open, full-corpus clinical-trial matching pipeline for TREC Precision Medicine. It did **not** beat the
TREC 2022 winner (NDCG@10 0.6125), but the investigation is the contribution: a cheap open funnel that cuts
~375k trials → 100 while preserving a 0.90 top-10 ceiling; held-out evidence that a frontier labeler
out-ranks a fine-tuned cross-encoder (+0.15); and a clean mechanistic negative result (error-reflective
prompt distillation over-corrects, because it steers toward criterion-verification the terse topics can't
support — the §7g information wall). The honest positioning: the field's best system gets ~0.51 P@10, so a
90%-clean top-10 from a three-line vignette is above the current frontier — this is a fast, cheap,
~SOTA-precision **assistive** first pass, not an autonomous oracle.
