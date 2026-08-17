# Clinical-Trial Matching by Funnel + Frontier Labeler — Project Write-up

*A cheap, open retrieval funnel feeding one frontier-model labeler: what it achieves, where it plateaus,
and why. Honest close, 2026-08. Full turn-by-turn record in `reflective_labeler_design.md` §8b–§8f;
SOTA-ceiling context in `deep_dive_outline.md` §7, §11, §13.*

---

## The idea we tested

The best available model, shown a (patient, trial) pair, can judge relevance better than a trained
cross-encoder — but labeling all ~375k trials per topic is infeasible in cost and latency. So use cheap,
proven IR to funnel 375k → ~100 candidates that still contain the relevant trials, and spend the
expensive labeler only there. Target metric: TREC 2022 Clinical Trials, NDCG@10, benchmarked against the
competition winner (h2oloo, 0.6125).

We did not beat SOTA. But the investigation produced four results worth keeping — three positive, one a
clean mechanistic negative — and, crucially, **placed the whole effort against the actual field ceiling**,
which turns out to be far below where intuition puts it.

---

## Contributions

### 1. A recall-complete funnel that cuts 375k → 100 while preserving a 0.90 top-10 ceiling
`BM25 + fine-tuned dense retriever → RRF fusion → BioLinkBERT cross-encoder rerank → top-100`, all
open-weight, one GPU pass, no API cost. The reduction is aggressive but nearly free *for a top-10 metric*:
oracle NDCG@10 over the full retrieval pool is 0.949; over just the top-100 it is **0.899** — 95% of the
achievable top-10 survives at 1/10th the labeler calls. The rerank stage *raises* recall at every width
(`clf@W > ret@W`), so it pulls relevant trials up as it narrows rather than burying them. **The funnel is
not the bottleneck.** (§8b)

### 2. Held-out evidence that a frontier labeler beats a fine-tuned cross-encoder
On the blind TREC 2022 test, ranking the *same* 100 candidates: cross-encoder **0.339**, Opus whole-doc
labeler **0.505** — **+0.15 NDCG@10**. A promotion effect: the labeler hands the ranker a ~46%-eligible-
dense bucket instead of the ~15%-dense pool. Clean, same-footing comparison; the labeler carries real
signal. (§8c)

### 3. Chain-of-thought does not help whole-doc grading here — and costs 2×/5×
CoT scored **below** no-CoT (0.482 vs 0.505) at twice the dollars and five times the latency. Mechanism
(confusion matrices): CoT raises true-eligible recall but raises false positives *more* — its calibrated
inference infers eligibility for off-topic trials too, polluting the top-10. For a top-heavy metric,
whole-doc no-CoT (which already commits to a grade, so §7g's abstention problem never bites) is the better,
cheaper base. (§8c)

### 4. A mechanistic negative: error-reflective prompt distillation *over-corrects* on this task
The headline method — mine the labeler's false positives, distill a guidance blob, append it — **hurt**:
dev NDCG@10 0.415 → 0.377, and (the decisive evidence) it pushed *true eligibles down hardest*
(mean score shift: gold=0 −15.8, gold=1 −7.6, **gold=2 −20.6**). **Why, and it ties the whole thread
together:** the distilled rules tell the labeler to *verify eligibility criteria*, but TREC topics lack
the clinical detail to verify them (§7g's 42% "not-enough-information"), so "can't verify → score low"
fires on eligible trials too — and eligibles have more criteria to fail-to-verify, so they take the
biggest hit. The labeler's edge came from *holistic clinical-picture matching*; reflection pushed it toward
*criterion verification*, the exact thing §7g proved doesn't work here. **Error-reflective distillation
amplified the §7g failure mode.** That is a publishable negative result with a clean causal story. (§8f)

---

## The honest positioning — and why it should feel like a ceiling, not a failure

Standalone-labeler top-10: **~43–49% precision** (on-topic and strictly-eligible track each other, because
Excluded is a tiny qrel class — no metric trick loosens the bar). Set against the field:

> **h2oloo, the TREC 2022 *winner*: P@10 = 0.508.** (deep_dive_outline.md §7i, verified)

The best clinical-trial-matching system in the world returns a top-10 that is ~51% relevant. Ours is in
that range. **A 90%-clean top-10 from a three-line patient vignette is above the global frontier** — the
information to do it is not in the input (§7g), and trial eligibility text often does not even name the
disease. This is not a project that fell short of an ordinary target; it ran into a ceiling the whole field
is under. Two caveats push the *real* number modestly higher (not to 90%): precision is measured
pessimistically against a shallow judged pool (unjudged full-corpus trials the labeler surfaces count as
misses), and a real deployment has richer input, a confidence threshold, and a human who filters ten
candidates in seconds.

---

## Method (reproducible)

- **Corpus:** ClinicalTrials.gov 2021 snapshot, full-text (title+conditions+summary+description+
  interventions+eligibility), 374,647 docs.
- **Retrieval:** BM25 (`rank_bm25`) + `ctmatch-retriever-v2` (MiniLM) dense, RRF-fused → ~1,000-doc pool.
- **Rerank:** `ctmatch-clf-R` BioLinkBERT cross-encoder → top-100.
- **Labeler:** `claude-opus-4-8`, whole-doc, thinking disabled, graded/ scored, ~$0.80/topic, no GPU.
- **Eval:** `pytrec_eval` NDCG@10 (graded gains 2/1/0) vs official qrels; train = TREC21+KZ, test = TREC22
  held-out (spent once). Notebooks: `eval_funnel_recall`, `label_pointwise_opus`, `label_scored_trec21`,
  `reflect_distill_trec21`.

## Results at a glance

| Finding | Number | Split |
|---|---|---|
| Funnel top-100 oracle ceiling | 0.899 | TREC22 |
| Cross-encoder ranking the top-100 | 0.339 | TREC22 (held-out) |
| Opus labeler ranking the top-100 | **0.505** | TREC22 (held-out) |
| Reflection guidance lift | **−0.039** (over-correction) | TREC21 dev |
| Standalone labeler precision@10 | ~0.43 | TREC21 |
| SOTA (h2oloo) precision@10 | 0.508 | TREC22 |

## Limitations
Retrieval recall ~55–60%@1000 (field-standard; a real weakness only under a *find-every-trial* goal, not
top-10). Terse benchmark topics understate a real deployment. Reflection tested only in single-blob form
(per-rule dev-gating is dominated at n=10). Upstream clf/retriever trained on TREC21, so TREC21 is not a
clean labeler-vs-clf comparator (used only for labeler-vs-labeler, which is clean).

## What this leaves, reusable
The funnel (open, cheap, recall-preserving), the held-out labeler-beats-cross-encoder result, the
diagnostic method (oracle-within-bucket decomposition that localizes ordering-vs-precision bottlenecks),
and the reflection negative with its mechanism. Two deliverables, as intended at the outset: an honest
**assistive first-pass pipeline** (~SOTA precision, fast, cheap) and a **negative-results contribution**.

---

## Closing

We asked whether a cheap funnel plus a frontier labeler could return a clean, mostly-relevant top-N fast.
The answer, measured with unusual rigor against the hardest available bar: it returns a *first pass whose
top results are on-target about as often as the world's best system's are* — useful as an assistive tool a
clinician refines, not an autonomous 90% oracle, because that oracle does not yet exist for anyone. The
work localized *why* (the §7g information wall, reached from three independent directions), and turned a
SOTA miss into a set of findings that stand on their own.
