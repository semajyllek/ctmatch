# One Last Look — Untried Levers Review (2026-07-14)

Independent code-level review of the ctmatch pipeline against the deep-dive (`deep_dive_outline.md`),
asked after the pipeline reached a defensible ceiling (TREC22 full-corpus NDCG@10 = 0.6105 vs
h2oloo 0.6125; TREC21 CV = 0.552). Question: is there any materially-sized lever not yet tried?

**Answer: yes — one, and it is the best kind for this project (concrete, cheap, derived from your own
data, not model-swapping). The cross-encoder features that dominate the reranker are scored on a
document representation that (a) truncates the eligibility criteria off most trials, and (b) for
clf-v4, is a distribution the model was never trained on.** A short literature pass is at the end;
it is secondary to this finding.

---

## The finding: the reranker's dominant features never reliably see eligibility text

### What the code does (verified)

- **Corpus blob order** (`build_fulltext_corpus.ipynb`, §7j.2): each trial is assembled as
  `briefTitle + officialTitle + conditions + briefSummary + detailedDescription + interventions +
  eligibilityCriteria`. **Eligibility is the *last* field.**
- **Cross-encoder feature extraction** (`train_ensemble_full.ipynb`, cell 5): both cross-encoder
  features — `clf_rel`/`clf_partial` (clf-v4) and `v2_rel` (reranker-v2) — are computed with
  `clf_tok([topic]*n, chunk, truncation=True, max_length=512)` over that full-text `chunk`.
  Default `longest_first` truncation trims the long side (the trial) first.
- **Doc length** (§7j.2): median 2,853 chars ≈ 700–900 tokens; BioLinkBERT's 512-token budget holds
  roughly the first ~1,200–1,400 chars of the trial after the topic. Since eligibility is last, **the
  majority of pool docs lose their eligibility section entirely at scoring time.** The cross-encoders
  are largely scoring *topicality* (title + conditions + summary), not *eligibility* — exactly the
  rel=2-vs-rel=0 signal they exist to provide.

### The second, verified part: clf-v4 has a train/inference representation mismatch

- **clf-v4 training text** (`build_dataset.ipynb` cell 24 → `load_doc_texts()` →
  `dataprep.DOC_TEXTS_PATH = "doc_texts.txt"` from `semaj83/ctmatch_ir`). The doc itself confirms
  `ctmatch_ir` is **eligibility-criteria text only** (§7i Finding 1: "built from eligibility-criteria
  text only … the indexed docs are enrollment-rule text").
- **The standalone clf-v4 harness** (`eval_baseline.ipynb`, the 0.6388 / gate-0.7460 numbers) runs
  through the `CTMatch` pipeline with `ir_setup=True`, which also scores on that **eligibility-only**
  `doc_texts.txt`.
- **The ensemble** (`train_ensemble_full.ipynb`) scores clf-v4 on `doc_texts_fulltext.txt` — the
  full-text blob above.

**So clf-v4 scores 0.6388 on its native representation (eligibility-only), and the ensemble then
feeds it a different distribution — full text, with the one field it *was* trained on frequently
truncated away.** (reranker-v2 was both trained and scored on full text, so it has no *mismatch* — but
it still eats the eligibility-last truncation on long trials.)

**Correction / nuance (2026-07-14, after full audit — see deep-dive §2g).** The full-text choice is
**deliberate and empirically validated**, not an unnoticed bug: `eval_fullcorpus.ipynb` sets
`RERANK_TEXT='fulltext'` because full-text reranking beat eligibility-only for clf-v4 on every pool
(BM25→clf **0.458 vs 0.369**). So the real point is not "you're feeding clf the wrong text" — it is that
**both tested options are flawed**: full-text-with-eligibility-truncated vs eligibility-only-with-no-topicality.
The genuinely untried third option is a representation that keeps **both** topicality and eligibility in
one budget (field-reorder or head+tail), trained *and* scored consistently. The audit (§2g) also found the
same eligibility-truncation hits the **`llm_yesno` judge feature** (`trial[:1800]` chars of full-text →
eligibility off) and the **dense retriever** (MiniLM 256-token cap on full-text) — so it is a systemic
representation issue, not a clf-v4 quirk.

### Why this is the right lever, mechanistically

- **You already proved the truncation costs real signal, on this exact data.** §11d found RankZephyr
  was crippled because `PASSAGE_CHARS=400` on the eligibility-last blob meant it "saw only
  title+conditions — never any eligibility criteria." Preserving eligibility (head 220 + tail 1200)
  moved it **−0.039 → −0.009, a +0.030 swing.** That fix was applied to the *listwise reranker only*
  and **never propagated to the cross-encoder features** — which are the ensemble's dominant
  reranking signal, not a discarded experiment.
- **It fits the §11c gap decomposition, with the correct framing.** §11c found the cross-encoders rate
  buried eligibles *higher* than surfaced ones (clf 0.265 vs 0.132) but are "too flat/uncalibrated to
  override" the LLM-dominated top-10, where buried eligibles sit at a lukewarm LLM +1.90. So the
  cross-encoder is *already pointing the right way and being overridden*. The fix is not "make a blind
  feature see eligibility" — it is "**sharpen a feature the ensemble under-uses so the trees can
  promote these +0.265 eligibles over the LLM's +1.90**." That is precisely the orthogonal, sharper
  signal §11c said was needed to win the tight race — and it is a feature you already have, mis-fed.

### Why "untried" is accurate here (pre-empting the §11b objection)

§11b reports "eligibility-only rerank text lowered all pools → disproved the 512-truncation
hypothesis." That tested the *crude* version: eligibility **only**, which drops topicality and
condition text (needed to separate rel=0) and conflates two changes at once. The targeted fix —
**preserve *both* topicality and eligibility** via field-reorder or head+tail truncation — is what is
genuinely untried on the cross-encoders. §11b does not rule it out; it rules out the opposite extreme.

---

## Proposed test (cheap first, then the clean version)

This is Colab/GPU-bound, so this is a recommended experiment, not a result. Anti-gaming: **develop on
TREC21 CV; touch TREC22 exactly once.**

**Step 0 — diagnostic (minutes, no training).** Measure the size of the problem: over the TREC21
candidate pools, what fraction of docs have their `eligibilityCriteria` section fully or partly
truncated when tokenized to 512 with the topic? Also print the frozen LambdaMART's feature
importances for `clf_rel`/`v2_rel` — if they are already near-zero, the ensemble has learned to
distrust the mis-fed feature (corroboration *and* a caveat on upside).

**Step 1 — cheapest test (feature re-extraction + CPU LightGBM).** Re-extract `clf_rel`/`clf_partial`/
`v2_rel` with an **eligibility-preserving representation** — either (a) head+tail truncation landing
the tail on the eligibility section (reuse the §11d recipe that recovered +0.030), or (b) reorder the
blob to `conditions + eligibility + title + summary + …` so eligibility survives `longest_first`.
Re-run the **frozen** LambdaMART on TREC21 CV. If NDCG@10 moves, the lever is real.
Note the caveat: clf-v4 was trained eligibility-only, so re-scoring it on head+tail full-text is still
off-distribution — a positive result here is a floor, a null does not kill the idea (Step 2 does).

**Step 2 — the clean version (one A100 fine-tune).** Retrain clf-v4 (and continue-train v2) on the
**same eligibility-preserving representation used at inference**, removing the mismatch entirely.
Since you are retraining the neural cross-encoder anyway, this is the natural place to also apply the
**still-untried listwise/ranking loss on the cross-encoder itself** (§9b / Tier 2b — the LambdaMART
combiner is listwise, but clf-v4/v2 are still pointwise cross-entropy and only produce features).
That folds two untried levers — representation fix + neural-listwise loss — into one run, both aimed
at the §11c diagnosis (sharpen the cross-encoder so it can override a lukewarm LLM on buried
eligibles). Re-extract features, re-run the ensemble on TREC21 CV, then the single TREC22 shot.

**Honest ceiling.** Even with eligibility restored, RankZephyr stayed flat (−0.009). So this is a
materially-sized, mechanism-clean, *untried* lever — **not** a guaranteed win. The strongest claim
supported by the evidence is "the ensemble is provably mis-feeding its dominant reranking features,
and the one on-data fix for it has never been tried." That is worth the ~1 GPU-hour before banking.

---

## Secondary: literature pass (2024–2026), ranked below the code fix

The user invited a literature scan. None of these is stronger than the representation fix above; they
are model-swaps on an axis (§11c) shown to be near-saturated. In rough EV order:

1. **Fine-tuned listwise reranker (your own "only remaining reranking card").** Zero-shot RankZephyr
   failed fairly (§11d), but a listwise reranker *fine-tuned* on TREC21+KZ is untried. Current open
   options past RankZephyr: **RankLLaMA** (Ma et al., pointwise/listwise, Llama-2 based) and
   **FIRST** (Reddy et al., 2024 — listwise reranking via the first output logit, faster + trained
   with a learning-to-rank loss). A listwise model *fine-tuned on your qrels* is the version that
   matches the §11c mechanism (reads candidates together) without RankZephyr's zero-shot transfer gap.
2. **Setwise / reasoning rerankers.** **Setwise** (Zhuang et al., 2024) is a cheaper comparison
   primitive than sliding-window listwise. **Rank-R1 / reasoning rerankers** (2025, GRPO-trained
   rerankers that reason before ordering) are the current frontier — but your own Tier-3c
   deprioritization argument (GRPO elicits, does not exceed, a ceiling a frontier model doesn't reach;
   Claude criterion-level was 0.6359 < clf-v4) applies here too. Low EV given that evidence.
3. **Listwise distillation into an open reranker.** Distill a strong listwise teacher's orderings on
   TREC21 pools into a small open reranker (a la RankVicuna/RankZephyr's own training). This is the
   supervised path §11a pointed to ("the path is supervision, not more zero-shot cleverness") applied
   to *listwise* rather than the pointwise judge — and it complements Step 2 above.
4. **monoT5-3B (`monoT5_CT`) as an ensemble feature** — already on your §10 plan. It is a far stronger
   reranker than clf-v4-340M *and* h2oloo score it with sliding-window MaxP over multi-field, so it
   does not truncate eligibility. Same pointwise axis that went null twice, but a categorically
   stronger model on the fixed representation. Worth the near-free feature test once
   `finetune_monot5_ct.ipynb` validates ≈0.71 on TREC21.

## Bottom line

There is one genuinely untried, code-verified, mechanism-matched lever left: **the ensemble's
dominant cross-encoder features are scored on a representation that truncates eligibility off most
trials, and clf-v4 is additionally scored off its training distribution.** Fixing the representation
(head+tail or field-reorder), ideally with a consistent clf-v4 retrain plus a neural-listwise loss,
is ~1 GPU-hour, develops cleanly on TREC21, and is the fix §11d's +0.030 swing and §11c's "sharpen
the under-used feature" both point to. If it lands, it is a real gain and a clean paper story ("we
found and fixed a representation mismatch in the reranker"). If it does not, **banking 0.6105
(TREC22) / 0.552 (TREC21 CV), fully open, with §11 as the negative-results appendix, is a legitimate
and defensible stop** — but this one lever is worth spending before accepting the ceiling.
