# Reflective-Prompt Labeler over a Recall-Complete Funnel — Design Spec

*Design doc for artifact #2: a fast, cheap, ~SOTA clinical-trial matching application.*
*Status: DESIGN (pre-build). Locks methodology before any API spend. Created 2026-08-12.*

---

## 1. Thesis

> The best available model, shown a (topic, trial) pair, can label relevance better than
> SOTA ranking methods — but labeling all ~374k trials per topic is infeasible in **cost**
> and **latency**. So use cheap, proven IR to funnel 374k → a few dozen survivors that still
> contain the relevant trials, and spend the expensive labeler only there.

Success is a **tuple on the accuracy × cost × latency frontier**, not a single NDCG number:

| Operating point | NDCG@10 | Labeler calls / topic | Latency / topic |
|---|---|---|---|
| Naive: label full corpus | (ceiling being approximated) | ~374,000 | hours |
| **This app: funnel → label W survivors** | **to measure** | **W (~25–100)** | seconds |
| Open pipeline (already built, §7i) | 0.6105 | 0 *frontier* calls — but see below | ~1 GPU pass |
| SOTA — h2oloo, blind (TREC22) | 0.6125 | — | — |

**Honest framing of the baseline.** The 0.6105 open pipeline is *not* LLM-free: it already contains
an open **Qwen-2.5-7B yes/no** labeler as an ensemble feature (§7i). So the real research question is
narrower and more defensible than "labels better than SOTA methods": **does a whole-doc closed-frontier
labeler (Opus, in-path) beat an open-local-Qwen-yes/no-feature ensemble** — closed-frontier-in-path vs
open-local-as-a-feature. State it this way; a reviewer will otherwise note the baseline already spends
LLM inference.

The app wins if it lands near 0.61+ at ~50 calls/topic — a ~7,500× reduction over naive
full labeling — **and** the added labeler cost buys accuracy the open (Qwen-feature) pipeline can't.

## 2. Two deliverables

1. **Negative-results paper** — the SOTA-ceiling findings; mostly already in `deep_dive_outline.md`
   (§7, §11, §13). Not covered further here.
2. **The application** (this doc): funnel + reflective-prompt-optimized graded labeler, with a
   clean held-out evaluation.

## 3. The load-bearing method: reflective prompt optimization

Family: Reflexion / ProTeGi / APO / DSPy-MIPRO — mine a labeler's errors, have a strong model
diagnose them in natural language, distill those diagnoses into compact guidance, prepend it,
re-evaluate. Our instance: **reflect on labeled errors → distill one guidance blob → freeze → test blind.**

Why it fits the app: the optimization is a **one-time offline cost on training data**; at inference
it is only a few hundred extra prompt tokens per call, so the labeler stays fast and cheap.

## 4. Data & splits — the validity landmine

The reflection step reads **gold labels** (it must, to know an error occurred). Therefore it is
**supervised and may only ever touch training data.** Three-way split, no exceptions:

| Split | Topics | Role |
|---|---|---|
| **Train** | TREC21 (75) + KZ (59) = 134 | Label blind → collect errors → reflect → propose guidance |
| **Dev** | held-out slice of train (e.g. a TREC21 fold, or KZ held apart) | Gate: does `base+guidance` beat `base`? Tune/stop distillation here |
| **Test** | TREC22 (50) | Run **once**, guidance frozen. No reflection, no re-selection, ever |

**Hard rule:** if TREC22 errors feed the guidance and TREC22 is then reported, the number is
leaked and worthless. TREC22 gold never enters the reflection or selection loop.

Note (KZ caveat, from §7i): KZ has a 2015-qrels-vs-2021-corpus vintage mismatch and terse
one-liner topics; it is a weak dev signal. Prefer TREC21 folds for the dev gate; report KZ
separately if at all.

## 5. Architecture (four stages + funnel front-end)

### 5.0 Funnel front-end (candidate generation — the cheap part)
- **Hybrid retrieval**: BM25 (`retrieval/bm25.py`) + fine-tuned dense retriever, fused by RRF
  (`retrieval/hybrid.py`). This is what actually achieves recall@1000 ≈ 0.52; wire it in as the
  front-end **replacing** `pipeline.py::sim_filter` (a pure-Python per-doc cosine loop over 374k —
  the current latency killer).
- **Speed**: dense stage must run on an ANN index (FAISS / hnswlib), precomputed once. BM25 already fast.
- Optional mid-funnel narrowing: `svm_filter` / `classifier_filter` to trim the pool before the labeler.
- **Instrument recall@k after every stage** (see §6) — the funnel's real job is "don't drop relevant."

### 5.1 Labeler (blind, per-call)
- Pointwise, whole-doc, **graded 0–2**: `(topic, trial_text) → {0,1,2}` matching qrel gains
  (Eligible=2, Excluded=1, Not-relevant=0). One call per (topic, doc); N docs → N calls.
- Model: `claude-opus-4-8` (in-path Claude is an intentional scoped exception for this artifact;
  the open-models rule still governs everything else).
- **Reasoning: CoT is not optional — it is the likely source of gain.** §7g Finding 5 is explicit:
  Sonnet's *loss* to clf-v4 was caused by a bare label prompt **abstaining under incomplete
  information** (42% not-enough-info), whereas TrialGPT's win came from **CoT-enabled calibrated
  inference** ("ambulates independently → ECOG 0-1"). A single-token, no-CoT labeler is close to the
  *losing* configuration from our own evidence, and Sonnet→Opus won't fix a mechanism problem. So the
  labeler reasons step-by-step, then emits the 0–2 grade (optionally a within-grade confidence).
- **Cost note:** CoT means more output tokens → real $ and latency. The "seconds/topic" and "$/topic"
  lines in §6 must be measured with CoT on.
- **This is whole-doc, not §7g's criterion-level decomposition** — it sidesteps §7g's per-criterion
  "42% not-enough-info" dilution by weighing the whole clinical picture at once *and* by letting the
  model make calibrated inferences instead of abstaining criterion-by-criterion.

### 5.2 Reflection (supervised, errors only) — TRAIN ONLY
- For each `pred ≠ gold`: give Opus `(topic, trial, predicted_label, gold_label)` and ask
  *why gold is correct and why the labeler chose predicted* → one short diagnosis.
- Include a few **correct** cases for contrast so the model can articulate what "right" looks like.
- **Reflect on TREC21, not KZ.** KZ is a weak reflection signal for the same reason it's a weak dev
  signal (§4): its terse-one-liner failures don't transfer to TREC22's vignette style, so distilling
  KZ errors pollutes the guidance with off-distribution rules. Use KZ errors only if explicitly
  targeting the one-liner failure mode as a *separate* conditional rule.
- **Weight reflection toward clean confusions.** The 2↔1 (Eligible vs Excluded) boundary carries both
  the largest NDCG gain difference (3 vs 1) *and* most of the qrel inter-annotator noise — reflecting
  there risks distilling annotator idiosyncrasy into "guidance." First check the train error mix: if
  errors are dominated by 2↔1, treat with suspicion; prioritize the cleaner, learnable 2↔0 and 1↔0
  confusions.

### 5.3 Distillation — the crux and the main risk
- Compress hundreds of diagnoses → one short, prompt-appendable guidance blob.
- **Failure mode: collapse to vague mush** ("be careful with lab values") that changes no decision.
- Method: cluster diagnoses by failure type → one *operational* rule per cluster → keep top-K by
  frequency/impact. Then iterate ProTeGi-style: propose blob → measure on **dev** → keep only
  rules that move the dev metric → repeat until no dev gain.
- **The guidance's real job is calibrated-inference proxies, not caution.** Because the labeler now
  reasons (§5.1), the useful rules are of the form *unstated-X → infer-from-observable-Y*, e.g.:
  *"If the topic does not state performance status (ECOG/Karnofsky) but describes independent
  ambulation / normal activity, infer ECOG 0–1 rather than treating the criterion as unmet or
  abstaining."* Vague caution ("be careful with labs") does nothing; a no-CoT base couldn't act on
  inference guidance at all — which is a second reason CoT (§5.1) is a prerequisite for this method.

### 5.4 Deploy + evaluate
- `base_prompt + frozen_guidance` re-labels the **test** pool → score vs qrels.
- **Mandatory ablation**: identical model + prompt **minus** the guidance blob. The entire claim
  is "distilled guidance improves held-out accuracy," so this is the only baseline that proves it.
- **Score-composition rule — specify BEFORE building the harness (known crater).** The labeler scores
  only the top-W of a ~1000-doc pool; the rest must still be ranked. §7i Finding 5 already burned this:
  letting the reranker's sort replace the full retrieval order (empty/zeroed tail) **cratered MRR**.
  Do **not** zero the tail. Use one of the two fixes that worked in §7i:
  1. **Rerank-top-W, keep hybrid order below** — labeler grades only reorder the top-W; docs below W
     retain their hybrid-RRF rank. (Best in §7i at W≈100.)
  2. **Score fusion** `α·labeler + (1−α)·hybrid` over the pool (α tuned on **dev**, not test).
  Reproducing the crater and misreading it as "labeler failure" is the specific risk this guards against.

## 6. Metrics & instrumentation

- **Per-stage recall@k** (`experiments.py::recall_at_k`, `rel_level` = 2 for Eligible-only,
  1 for Eligible+Excluded): fraction of judged-relevant surviving each funnel stage. The funnel's
  own success criterion; also picks the widths.
- **Oracle NDCG@10 on the pool** (`sorted(pool, key=gold, reverse=True)` → `ndcg_at_k`): the
  ceiling a perfect labeler could extract from the retrieved pool. On the hybrid pool ≈ **0.957**
  (recall-limited, not a labeler claim) — confirms the relevant docs are present.
- **Final NDCG@10** (graded gains `2^rel−1`, `experiments.py::ndcg_at_k`) vs qrels — headline accuracy.
- **Cost**: labeler calls/topic × price; report $/topic.
- **Latency**: retrieval ms + labeler wall-clock (parallelized with a semaphore) — report sec/topic.
- Report the whole **accuracy × cost × latency** curve as W ∈ {25, 50, 100} sweeps, not one point.

## 7. Risks & guardrails

1. **Test contamination** — reflection/selection must never see TREC22 gold. (§4)
2. **Distillation collapse** — keep only dev-validated operational rules; reject vague ones. (§5.3)
3. **Guidance overfit to train quirks** — the dev gate is the defense; don't skip it.
4. **Prompt bloat** — long guidance erodes the fast/cheap pitch; keep the blob short.
5. **Selection multiplicity** — pick the guidance variant on dev; TREC22 spent exactly once.
6. **Premise may fail** — whole-doc Opus labeling might not beat the free 0.6105 open pipeline
   (§7g: criterion-level Sonnet *lost* to clf-v4; §8a: open LLM judge buried eligibles). If so,
   the honest finding — "a cheap open funnel already sits on the frontier; the frontier LLM
   doesn't earn its cost" — is itself a result and folds into deliverable #1.

## 8. Baseline experiment to run FIRST (de-risk before building the loop)

Before the reflection machinery, settle the premise cheaply on a **fixed** train pool:

> Score the **clf top-100** three ways — (a) clf-v4 (free, cached), (b) Opus whole-doc 0–2 **no-CoT**,
> (c) Opus whole-doc 0–2 **with CoT** — read off W ∈ {25, 50, 100} by truncation. Report
> `(NDCG@10, $/topic, sec/topic, fail%)` each, using the §5.4 composition rule (no zeroed tail).

**Cost is real — this is a cost-efficiency artifact, so measure it.** The full 50-topic run is
`50×100×2 = 10,000` Opus calls with whole-doc prompts (no prompt caching — the shared prefix is under
Opus-4.8's 4096-tok cache minimum): **≈ $50–200 total** (CoT dominates, thinking billed as output),
not "a few dollars." Protocol: (1) a `count_tokens` estimate cell prints the real number before any
spend; (2) **probe 10 topics first** (`PROBE_N=10`, ~$15–40) to read true $/topic + the no-CoT
**parse-failure rate** before lifting to all 50 (resumable). The no-CoT arm uses `max_tokens=64` (not
16) and every defaulted-to-0 grade is counted — if no-CoT `fail% > 2%` the control is truncation-
compromised (biased toward 0, which would fake the §7g gap) and gets rerun with headroom before its
number is trusted.

- **The CoT arm is non-optional.** Running the gate as clf-v4 vs single-token-Opus only would risk a
  *false negative* that kills the premise for the wrong reason — §7g Finding 5 says the no-CoT config
  is the losing one. (b) vs (c) also directly measures what CoT costs on the frontier (tokens → $/latency).
- If CoT-Opus clears clf-v4 / 0.6105 by a real margin → premise holds; build the reflection loop (§5.2–5.4).
- If even CoT-Opus ties/loses → premise fails cleanly; pivot to deliverable #1. Either way we learn decisively.

Then, only if the premise holds, the reflection loop is evaluated for its **marginal** lift:
`Opus base` vs `Opus base+guidance` on frozen TREC22.

## 8b. Empirical funnel-recall results (2026-08-13, `eval_funnel_recall.ipynb`)

Run on the existing `pool_R.json` + cached `ce_clf_R.npz` (no rerun). Two findings lock the funnel design.

**Ceiling by width (oracle NDCG@10 if the labeler perfectly scored clf's top-W):**

| split | pool | top-25 | top-50 | top-100 | top-500 |
|---|---|---|---|---|---|
| TREC21 | 0.988 | 0.819 | 0.935 | 0.983 | 0.992 |
| TREC22 | 0.949 | 0.725 | 0.837 | **0.899** | 0.956 |
| KZ | 0.492 | 0.259 | 0.325 | 0.400 | 0.556 |

Every width — even top-25 — leaves a TREC22 ceiling **above** the 0.6125 SOTA target. Funnel width is
therefore *not* the binding constraint; labeler extraction is (consistent with §8a). Knee at **W=100**
(0.899; +0.057 only for 5× the calls at 500). KZ caps at 0.400 even with perfect labeling — a retrieval/
corpus-vintage wall no funnel width fixes; stays out of the headline (§4).

**Recall waterfall — clf *improves* recall, never buries it** (TREC22, Eligible-only `rel=2`):

| W | ret@W (RRF) | clf@W | lift |
|---|---|---|---|
| 25 | 0.102 | 0.160 | +57% |
| 50 | 0.156 | 0.230 | +47% |
| 100 | 0.228 | 0.320 | +40% |
| 500 | 0.452 | 0.553 | +22% |

`clf@W > ret@W` at every width → the §5.4 "clf narrowing throws away recall, must fuse" worry does **not**
occur; feed the labeler clf's top-W straight. The clf lift is concentrated on Eligible (rel=2); on
Eligible+Excluded the lift shrinks (clf@100 0.243 vs ret@100 0.210) — clf-R is an eligibility view, and
NDCG@10's graded gains reward rel=2 most, so clf's strength is aimed where the metric pays. 32% eligible
recall at W=100 ≈ 15 eligible/topic in the pool → the 0.899 ceiling (surplus story, confirmed).

**Locked:** funnel = hybrid → clf-R rerank → **top-100** (no fusion); labeler width **W=100** (read off
50/25 for free by truncation); gate holds candidate set fixed = clf top-100, varies only the scorer.
Labeler must extract ~68% of the 0.899 ceiling to clear SOTA; clf-v4 does ~50% — that gap is the bet.

## 8c. Gate probe result (2026-08-13, `label_pointwise_opus.ipynb`, 10 TREC22 topics)

NDCG@10 over clf top-100, labeler-alone (no fusion/ensemble). Control clean: no-CoT fail% = 1.0.

| arm | W=25 | W=50 | W=100 | $/topic | sec/topic |
|---|---|---|---|---|---|
| clf-v4 (free) | 0.339 | 0.339 | 0.339 | 0 | — |
| **opus-no-CoT** | 0.475 | 0.495 | **0.505** | 0.82 | 9 |
| opus-CoT | 0.458 | 0.478 | 0.482 | 1.72 | 43 |

**Finding 1 — premise HOLDS, decisively.** Both Opus arms beat clf-v4 by **+0.14–0.17** NDCG@10
(0.50 vs 0.34) on the *same* candidate set. A whole-doc frontier labeler extracts far more than the
clf cross-encoder. The funnel + strong-labeler thesis is validated.

**Finding 2 — CoT does NOT help; it slightly hurts, at 2× cost and ~5× latency.** This *reverses* the
§7g-based design assumption. The confusion matrices explain why: CoT raises true-Eligible recall
(gold=2→pred=2: 107 vs 90) **but** raises false positives more (gold=0→pred=2: **117 vs 98**). Its
calibrated inference infers eligibility for topically-unrelated trials too, and those false-2s pollute
the top-10. Whole-doc no-CoT already commits to a grade (doesn't abstain), so §7g's abstention
mechanism never bites — CoT's extra "willingness to call Eligible" is net-negative for a top-heavy
metric. **Base labeler = no-CoT** (better, cheaper, faster).

**Finding 3 — the binding error is now visible and clean: false positives.** ~100/arm truly-not-
relevant trials graded Eligible (gold=0→pred=2). This is a **0↔2 confusion** (learnable, not the noisy
2↔1 tier) — the ideal target for the reflection/distillation stage: suppress "Eligible" for trials that
don't target the patient's actual condition. Note CoT's *higher* true-recall (107 vs 90) means a
false-positive-suppressing guidance might recover CoT's ceiling — CoT isn't dead, it's miscalibrated.

**Caveats:** n=10 (high variance — the no-CoT>CoT gap of 0.023 is within noise; the Opus≫clf gap is
not). Labeler-alone — no fusion with hybrid/clf retrieval scores or the LambdaMART ensemble, so the
0.50 is *not* comparable to the 0.6105 full pipeline; the apples-to-apples comparison is clf-alone
0.34 vs Opus-alone 0.50 (same footing). Reaching 0.61+ needs the ensemble/fusion on top, or the
reflection lift, or both.

**Finding 3, sharpened — the pred-2 bucket saturates, so clf (not the grade) orders the top-10.**
Docs graded 2: no-CoT 19.5/topic, CoT 23.3/topic — both ≫ the 10 NDCG slots. Composition sorts by
`(grade, clf_score)`, so the entire top-10 comes from the pred-2 bucket **ordered by the clf tiebreak**;
the labeler's grade only *selects the bucket*. Both buckets are ~46% eligible-dense (90/195 no-CoT,
107/233 CoT). So the Opus≫clf win is a **promotion effect** (Opus hands clf a 46%-dense bucket vs the
~15%-dense pool), and two reflection levers exist, not one:
- **(a) bucket precision/recall** — the 2-vs-not decision (raise the 46%, recover the ~42% eligible
  no-CoT misses).
- **(b) intra-bucket ordering** — a within-grade **confidence/score** (deferred in §10) could reorder
  the top-10 better than clf's tiebreak. *Back on the table because the grade saturates.*
A **$0 oracle-within-pred-2** check (sort each bucket by gold, NDCG@10) discriminates: ≫0.505 → lever
(b) has headroom; ≈0.505 → clf already orders the bucket near-optimally, lever (a) only.

**Test-hygiene correction (load-bearing).** The gate ran on **TREC22** (§4's held-out test). A one-shot
"does Opus beat clf" observation is defensible and no config was selected on it — but **all further
development stops using TREC22 now**. The no-CoT-over-CoT choice stands because it's justified on
cost/latency (2× cheaper, 5× faster, NDCG within noise), *not* a TREC22 win. The $0 oracle diagnosis is
clean (selects no config). But the **reflection loop and any fusion-weight tuning move to TREC21**
(§4 train), frozen, then TREC22 is spent **once** for the headline — else 0.50→final is another
test-adapted number (the §7i failure). Do **not** run the full 50-topic bare-labeler spend now; it buys
an intermediate number on a config about to change and eats test budget.

## 8d. Oracle decomposition — the ordering signal is the lever (2026-08-13, $0, on the probe)

```
nocot: actual@100=0.505  oracle-within-pred2=0.791  elig-recall-into-bucket=0.58
cot:   actual@100=0.482  oracle-within-pred2=0.827  elig-recall-into-bucket=0.69
```

**oracle-within-pred2 ≫ actual (0.791 vs 0.505).** Perfectly *ordering the same pred-2 bucket* lifts
NDCG@10 by **+0.29**, above SOTA — with no change to bucket membership. So the binding constraint is
**intra-bucket ordering, not bucket precision/recall.** The dominant lever is **lever (b): a
fine-grained labeler ordering signal.**

**Root cause — the 3-way grade is too coarse for a top-heavy metric.** ~20 docs/topic all grade "2";
the bucket saturates and clf's (weak, within-bucket) tiebreak decides the top-10. Fix: the labeler
emits a **continuous relevance score** (e.g. 0–100 or P(eligible)); rank by it. NDCG uses gold gains
regardless, so the labeler output need only be a good *ordering* — finer is strictly better.

**Re-sequence (does not drop reflection):** (1) scored labeler → capture the 0.505→0.79 ordering gap —
biggest, cheapest win, a prompt change; (2) CoT or reflection to raise recall-into-bucket (0.58→0.69→)
— the 0.79→0.83 secondary lever; (3) reflection to calibrate the score / cut residual errors. Reflection
was aimed at lever (a); the oracle says lever (b) is ~6× larger and simpler, so it goes first.

**All of the above develops on TREC21** (§4). TREC22 is spent once at the end. This oracle check
selected no config, so it's clean on TREC22.

## 8e. Scored labeler on TREC21 — the lever is false positives, and reflection is clean here (2026-08-13)

Scored no-CoT labeler (0–100), TREC21 probe (10 topics): `clf-alone=0.518, scored=0.514, oracle=1.000`.

**Two corrections:**
1. **TREC21 clf is contaminated** — clf-R trained on TREC21, so clf-alone (0.518) is inflated vs held-out
   TREC22 (0.339). "scored ≈ clf here" is clf on home turf, NOT a weak labeler. The labeler's real edge
   over clf is the held-out §8c result (0.50 vs 0.34), which stands.
2. **The oracle lever flipped from (b) to (a).** Score diagnostics: gold=2 mean **62.6** vs gold=0 **37.6**
   (signal present), 24 distinct scores/topic (resolution fine — *not* clustered). So ordering-resolution
   is not the problem. The high-score zone is ~40% precision (278 docs ≥70, ~189 eligible total) — the
   ceiling is **false positives** (lever a), the *same* ~46% top-precision as the gate. Plus a mechanical
   leak: 59/1000 parse-fails, **13 of them eligible** → forced to 0 (fix: `max_tokens` 64→128, score-first).

**Why this is good news for the plan:** false-positive suppression is exactly what **error-reflection**
targets (the original idea, §3/§5.2–5.4) — and TREC21 is a **clean** dev for it, because the reflection
comparison is **base-labeler vs base+guidance (Claude vs Claude, neither trained on TREC21)**. The clf
contamination only broke labeler-vs-clf, which reflection doesn't use. So: (i) quick parse-fix + re-probe
for a clean base number; (ii) build the reflection loop on TREC21 targeting gold=0-scored-high; (iii)
freeze; (iv) spend once on TREC22.

## 8f. Reflection loop — the distilled guidance OVER-CORRECTS (2026-08-13, TREC21, $41.51)

30-topic probe (20 train / 10 dev). `base_dev=0.4154, base+guidance_dev=0.3769, lift=-0.039`
(clf-contaminated 0.505, oracle 0.996). **The guidance hurt.**

**Not distillation collapse** — the blob was *operational*, not vague: classify trial type (cap
observational/registry low), verify the gating enrollment action (not topic overlap), check every hard
gate/exclusion, anchor to the active presenting problem, plus one "don't over-exclude" counter-rule.

**Over-correction CONFIRMED at the mechanism level** (paired check on cached scores). Per-topic
guided−base: 7/10 dev topics down (one +0.393 outlier drags the mean toward 0; sign test p≈0.17 — the
*NDCG* delta isn't individually significant at n=10). But the score-shift is unambiguous and directional:
mean shift **gold=0 −15.8, gold=1 −7.6, gold=2 −20.6**. The guidance pushed everything down AND pushed
**true eligibles down the most** — the opposite of what a useful reranker does.

**Why — and it ties the whole thread back to §7g.** The rules tell the labeler to *verify gating actions
and every exclusion criterion*. But TREC topics lack the clinical detail to verify criteria (§7g's 42%
"not-enough-info"), so "can't verify → score low" fires on **eligible** trials too — and eligibles have
*more* criteria to fail-to-verify, so they take the biggest hit. The labeler's edge (§8c) came from
**holistic clinical-picture matching**, not criterion checking; the reflection guidance pushed it toward
criterion verification, the exact thing §7g proved doesn't work on this data. Error-reflective distillation
mined false positives → strictness → amplified the §7g failure mode.

**Verdict:** the reflection method (this single-blob form) is a **confirmed negative** with a clean
mechanistic explanation. n=10 keeps the NDCG number from being individually significant, but the
gold-level score shift is decisive. Not worth per-rule gating (dominated, §5.3) or more spend. Both
deliverables are already in hand (app: §8b+§8c; negative paper: this + §11/§13). Recommend: bank + write up.

**What it points to (the strategic read).** The single-labeler-as-standalone-reranker path plateaus
~0.50 held-out, capped by false-positive precision that prompt-level strictness can't fix without
over-correcting. The labeler carries real signal (beats clf +0.15 held-out, §8c) — but its value is
likely as a **feature in the learned ensemble** (§7i LambdaMART already combines clf + LLM-yes/no +
retrieval → 0.6105), where a listwise combiner calibrates the false-positive tradeoff far better than a
blunt global prompt rule. Fork pending (§9): (a) declare the reflection negative + write up; (b) proper
per-rule dev-gate (fragile at n=10); (c) pivot the graded labeler to an ensemble feature and test the
0.6105 lift.

## 9. Build order & file plan

1. `notebooks/eval_funnel_recall.ipynb` — wire hybrid front-end into the funnel; per-stage recall@k
   instrumentation (§6). Needed regardless of the labeler question.
2. `notebooks/label_pointwise_opus.ipynb` — §8 gate: clf-v4 vs Opus-no-CoT vs Opus-CoT over a fixed
   pool, §5.4 composition rule applied, scored vs qrels; the accuracy × $/topic × sec/topic table.
3. *(gated on §8 result)* `notebooks/reflect_distill.ipynb` — stages §5.2–5.3 on **train**; emits a
   frozen guidance blob + the dev-gating loop.
4. *(gated)* extend `label_pointwise_opus.ipynb` with the `base` vs `base+guidance` ablation on TREC22.
5. ANN index for the dense retriever (FAISS/hnswlib) — the latency deliverable, parallelizable with 1–2.

## 10. Open questions

**Resolved 2026-08-13 (§8b):**
- ~~Funnel width W~~ → **W=100** (knee of the ceiling curve).
- ~~Fuse clf with hybrid for recall?~~ → **No** — clf@W > ret@W at every width; feed clf top-W straight.

**Still open (resolve before/while building):**
- **Dev split**: which TREC21 fold(s)? Keep KZ out of the gate given its caveats?
- ~~Labeler output: bare 0/1/2 or finer?~~ → **RESOLVED (§8d): fine-grained continuous score** (0–100 /
  P(eligible)). The 3-way grade saturates the top bucket and wastes ~0.29 NDCG@10 to clf's tiebreak.
- **Distillation granularity**: single global blob vs a few conditional rules keyed on topic features
  (e.g. terse vs detailed topics — the KZ one-liner failure mode).
- **Reflection scope**: all errors, or stratified by error type — §5.2 notes 2↔1 is noisiest; the clf
  recall lift being rel=2-concentrated (§8b) suggests the surviving pool under-represents rel=1, so
  most labeler errors on rel=1 may be recall-driven, not judgment-driven. Check the train error mix first.
