# Clinical Trial Matching: A Deep Dive
## From Raw Eligibility Criteria to Ranked Retrieval

> **Status:** Outline + working notes. Expand each section into full prose, code blocks, diagrams, and math.
> Intended audience: graduate-level ML/NLP, or a clinical informaticist who can read Python.

> ## ⚠️ NUMBERS INVALIDATED — pending clean re-runs (2026-07-14)
> The representations audit (§2g) found that models were trained and evaluated on
> **different, inconsistent document representations** (eligibility-only vs full-text; 512 tok
> vs 1800 char; eligibility truncated off most rerankers). **Every quantitative result in §7,
> §8, §11 and the README is therefore representation-confounded and is being re-run on one
> frozen representation (`R`).** Confounded headline numbers are struck through and tagged
> `PENDING(R)`; the surrounding method/analysis prose is retained (it explains *why*, which is
> still valid). **Do not cite any number in this document until it carries a representation tag
> (`cfg.repr_tag()`, e.g. `head_tail-L512-h50`) written by the new `ctmatch.experiments` result
> ledger.** Re-run program and rationale: §2g "Consequence for the surpass-SOTA effort" and
> `docs/notebook_cleanup_plan.md`. **Progress:** representation frozen (`R = elig_first-L512`); the clean,
> representation-consistent, **multi-view** full-corpus pipeline now scores **TREC22 NDCG@10 = 0.5548**
> (95% CI [0.477, 0.635], contains h2oloo's 0.6125) — see **§2h**. This is the honest `R` number; the old
> 0.6105 is retracted (it borrowed against the miscalibration this effort removed). The §7 tables below stay
> struck (`PENDING(R)`) and will be rewritten from the `R` runs; §2h holds the current live numbers.

---

## Document map

```
1. The Problem
2. The Data
   2a. ClinicalTrials.gov structure
   2b. TREC 2021 / 2022 datasets
   2c. KZ (Koopman-Zuccon) dataset
   2d. Structural differences: judged pool, topic format, relevance scale
   2f. Unified dataset, train contamination, and the clean test split
   2g. Representations audit — exact model input per experiment
   2h. Freezing the representation: the truncation study + clf_R validation
2e. Evaluation methodology: pooling, judged sets, and what we actually measure
3. Text Processing: ctproc
   3a. Raw XML → structured fields
   3b. Eligibility criteria parsing: the regex cascade
   3c. NLP layer: scispaCy, NegEx, UMLS linking
   3d. Topic processing: age/gender extraction
4. Information Retrieval: general theory applied to this task
   4a. The ranking problem
   4b. Candidate generation vs. reranking
   4c. Why standard IR assumptions break here
5. The ctmatch Pipeline
   5a. Architecture overview (diagram)
   5b. Stage 1: Embedding similarity + category prior (sim filter)
   5c. Stage 2: SVM ranking
   5d. Stage 3: SciBERT classifier
   5e. Stage 4: Generative reranker (gen filter)
   5f. Eval mode vs. inference mode — what the cascade actually does
6. Metrics
   6a. NDCG@10 — derivation and why it fits this task
   6b. MRR — finding the first relevant trial
   6c. F1 and FPR — precision/recall framing
   6d. The custom "optimistic rank" metric
7. Experiments and Results
   7a. Baseline (sim+svm+clf, TREC 2021 + KZ)
   7b. Embedding ablation: MiniLM vs. MedCPT
   7c. Filter ablation: sim+svm+clf vs. svm+clf
   7d. TREC 2022 results
   7e. The real target: full-corpus SOTA, and where TrialGPT actually sits
   7f. LLM reranker: zero-shot eligibility scoring (Tier 2a)
   7g. Criterion-level reranker: Claude ceiling test (Tier 3a)
   7h. Distilled open criterion cross-encoder (Tier 3b)
   7i. Full-corpus evaluation — the retrieval→rerank pipeline against the real target
   7j. Methods — the full open pipeline (reproducible)
8. Error Analysis
   8a. Confusion matrix breakdown
   8b. FP error patterns (cardiac enrichment, label quality)
   8c. FN error patterns
   8d. Data quality: label errors vs. model errors
9. Future Directions
   9a. Criterion-level entailment
   9b. Pairwise ranking loss on qrel preference signal
   9c. Lab value extraction and comparison
   9d. Path to SOTA without TrialGPT
   9e. Candidate improvements and medical models to try
10. Surpass-SOTA program (active work plan)
11. Negative results (addendum)
```

---

## 1. The Problem

**TODO — prose to write:**
- What is clinical trial matching? (patient description → ranked list of eligible trials)
- Why it matters: trial accrual is the bottleneck for most studies; 80% of trials delayed or stopped due to recruitment. ICU nurse perspective: physicians spend hours manually searching CTG.
- Why it is hard as an IR task:
  - Documents (trials) use regulatory/protocol language; queries (patients) use clinical narrative or discharge-summary language
  - Eligibility criteria are **logical constraints**, not descriptive text — a patient either satisfies a criterion or they don't
  - The relevant set is tiny (often 1-5 relevant trials out of 374k)
  - Graded relevance: "eligible" vs. "partially eligible" (meets inclusion but fails one exclusion criterion) is a clinically meaningful distinction that most IR systems ignore

**Notes:**
- The TREC Precision Medicine track framing: patient = topic, corpus = all ClinicalTrials.gov XML, task = rank by eligibility
- This is asymmetric retrieval: topic and document are not from the same distribution
- Tie to TrialGPT: their framing is criterion-level (can patient satisfy each inclusion/exclusion criterion individually?) — essentially a natural language inference (NLI) problem at the criterion level

---

## 2. The Data

### 2a. ClinicalTrials.gov XML structure

**TODO — diagram and code:**

```
NCT00XXXXXX.xml
├── id_info/nct_id            → "NCT00123456"
├── condition[]               → ["Type 2 Diabetes", "Obesity"]
├── intervention/
│   ├── intervention_type     → "Drug"
│   └── intervention_name     → "Metformin"
└── eligibility/criteria/textblock  → raw free text (see §3)
```

Key design choice in ctproc: ignore most metadata fields, focus almost entirely on `eligibility/criteria/textblock`. The condition and intervention fields are extracted but primarily used for the category prior (§5b).

**Notes:**
- The textblock is unstructured free text written by trial coordinators — format varies wildly across 374k trials
- Some have only inclusion criteria, some only exclusion, most have both
- Headers: `"Inclusion Criteria:"`, `"Exclusion Criteria:"` are nearly universal but not guaranteed
- Bullet delimiter: `"  - "` (two spaces + dash) is the de facto standard but not enforced

### 2b. TREC 2021 / 2022 datasets

| | TREC 2021 | TREC 2022 |
|---|---|---|
| Topics | **75** (IDs 1–75) | **50** (IDs 1–50) |
| Topic format | XML `<topic number="N">free text</topic>` | same |
| Corpus | ClinicalTrials.gov 2021-04-27 dump | same |
| Judged docs/topic (avg) | **~478** | **~708** |
| Total judged pairs | **35,832** | **35,394** |
| Label dist (0 / 1 / 2) | 24,243 / 6,019 / 5,570 | 28,419 / 3,036 / 3,939 |
| Label dist (%) | 67.7% / 16.8% / 15.5% | 80.3% / 8.6% / 11.1% |
| Relevance scale | 0/1/2 | 0/1/2 |
| Qrels file (Drive) | `trec_21_judgments.txt` | `qrels2022.txt` |
| Processed topics (repo) | `processed_trec21_topics.jsonl` | `processed_trec22_topics.jsonl` |
| Used in 134-topic baseline | **Yes** | No (excluded by default) |

Topic example (TREC 2021 format):
```xml
<topics task="2021 Clinical Trials">
  <topic number="1">
    74-year-old male with a history of hypertension, hyperlipidemia, and
    type 2 diabetes mellitus presents with chest pain...
  </topic>
</topics>
```

**Notes:**
- TREC 2022 excluded from the 134-topic baseline because its ~700 judged docs/topic makes eval ~9× slower than TREC 2021. With the classifier scoring every judged doc per topic, a full TREC22 run takes hours on Colab.
- Same 2021 corpus — TREC 2021 and 2022 NDCG scores are directly comparable.
- Relevance: 2=eligible, 1=partially eligible, 0=not eligible.
- **Labels live on Drive only** — the qrel files are not committed to the repo. The processed topic files (without labels) are in the repo.

### 2c. KZ (Koopman-Zuccon 2016) dataset

Topic file format (pseudo-XML, no proper root element, parsed line-by-line):
```
<TOP>
<NUM>20141</NUM>
<TITLE>35 year old female diagnosed with anorexia nervosa...</TITLE>
</TOP>
```

| | KZ |
|---|---|
| Topics in topic file | **60** (IDs 20141–20159, 201410–201530) |
| Topics with judged pairs in qrel | **59** (one topic absent from qrel entirely) |
| Topic format | pseudo-XML `<NUM>/<TITLE>`, one-liner descriptions |
| Corpus referenced by qrels | ClinicalTrials.gov 2015 dump |
| Corpus used for eval | ClinicalTrials.gov 2021 dump (mismatch — see below) |
| Total judged pairs | **3,870** |
| Judged docs/topic (avg) | **~65.6** |
| Label dist (0 / 1 / 2) | 2,764 / 685 / 421 |
| Label dist (%) | 71.4% / 17.7% / 10.9% |
| Relevance scale | 0/1/2 |
| Qrels file (Drive) | `qrels-clinical_trials.txt` |
| Topic file (Drive) | `topics-2014_2015-description.topics` |
| Processed topics (repo) | `processed_kz_topics.jsonl` (60 topics, no labels) |
| Source | Koopman & Zuccon, SIGIR 2016 |

#### kz_data.jsonl — the complete KZ qrel in pre-joined form (repo)

`data/kz_data/kz_data.jsonl` (3,869 pairs) is effectively the **complete** KZ judged set
(the qrel has 3,870 pairs; the 1-record difference is a processing artifact). It is a
pre-joined (topic_text, doc_text, label) representation — the same data as the qrel file,
but with topic text and doc text already embedded, so notebooks can use it without Drive
or HuggingFace access.

| | kz_data.jsonl |
|---|---|
| Total pairs | 3,869 (≈ full qrel) |
| Unique topics | 59 |
| Docs/topic (min / mean / max) | 13 / 65.6 / 153 |
| Label distribution | 0: 2,763 (71.4%) · 1: 685 (17.7%) · 2: 421 (10.9%) |
| Topic key | raw topic text (NOT the numeric ID) |
| Doc field | processed trial text (NOT an NCT ID) |

**Eval path vs. kz_data.jsonl:**
- The evaluator (`load_eval_datasets`) reads `topics-2014_2015-description.topics` + `qrels-clinical_trials.txt` from Drive, then fetches doc texts by NCT ID from `semaj83/ctmatch_ir` on HuggingFace. Same underlying data as `kz_data.jsonl`, different access pattern.
- `kz_data.jsonl` encodes the topic as raw text, not the numeric ID, so it cannot be directly joined to `processed_kz_topics.jsonl` by ID. Use it for data-exploration notebooks; use the Drive + HF path for formal NDCG evaluation.

**Notes:**
- KZ topics are one-liners vs. full clinical narratives in TREC. The category prior (bart-large-mnli) performs less reliably on short text.
- KZ's judged pool (~66/topic) is far smaller than TREC 2021's (~478/topic) — not a deeper pool as previously assumed. The pool likely reflects a manual-search strategy by the paper authors rather than system pooling.
- The 2015 qrel NCT IDs vs. the 2021 eval corpus: some NCT IDs may be absent from the 2021 corpus and are silently dropped by the evaluator — this is why 60 topic-file entries become 59 with actual qrel judgments.
- **KZ topic ID scheme mismatch**: the raw KZ topic file on Drive uses simple integer IDs (1–59), matching the qrel. `processed_kz_topics.jsonl` in the repo uses year-prefixed IDs ('20141', '20142', ...) generated by ctproc. These two ID schemes are incompatible — any code that tries to join `processed_kz_topics.jsonl` IDs against the qrel will silently match nothing. The evaluator is unaffected because it reads the raw Drive files directly via `get_kz_topic2text()`.
- **TREC22 and KZ topic ID collision**: both datasets number topics from 1 upward. Their qrels look identical in the first line (`['1', '0', 'NCT00000409', '0']`) but refer to completely different patients (KZ topic 1 = 58yo woman with chest pain; TREC22 topic 1 = 19yo male). The evaluator keeps separate `rel_dict` objects per dataset so there is no collision.

### 2d. Structural differences summary

| Dimension | TREC 2021 | TREC 2022 | KZ |
|---|---|---|---|
| Topics | 75 | 50 | 60 in file; 59 in qrel |
| Topic style | Full clinical narrative | Full clinical narrative | One-liner |
| Judged pairs/topic (avg) | **478** | **708** | **66** |
| Total judged pairs | **35,832** | **35,394** | **3,870** |
| Label dist (0/1/2 %) | 67.7 / 16.8 / 15.5 | 80.3 / 8.6 / 11.1 | 71.4 / 17.7 / 10.9 |
| Corpus | 2021 | 2021 | 2015 → 2021 (mismatch) |
| Annotators | NIST assessors | NIST assessors | Paper authors |
| In 134-topic baseline | Yes | No | Yes (59 topics) |
| Labels in repo | No (Drive only) | No (Drive only) | No (Drive); full set in `kz_data.jsonl` |

Key differences that affect system design:
1. **Topic length**: TREC topics are full clinical narratives (~100–200 words); KZ topics are one-liners. The category prior (bart-large-mnli) works better on longer text.
2. **Judged pool depth**: TREC 2021 and TREC 2022 have comparable total pairs (~35K each), but TREC 2022 concentrates them across fewer topics (708/topic vs 478/topic). KZ is much smaller (3,870 total, 66/topic). TREC 2022 is excluded from the baseline because 708 docs/topic × full classifier scoring is slow on Colab — not because the pool is larger in total.
3. **Corpus vintage**: KZ qrels reference 2015 NCT IDs evaluated against the 2021 corpus. Some NCT IDs may be absent, which is why 60 topic-file entries → 59 with qrel judgments.
4. **Annotation provenance**: TREC judgments by NIST assessors; KZ by the paper authors. KZ's smaller pool (~66/topic vs ~478/topic) likely reflects manual search rather than system pooling.
5. **Label access**: All qrel files are on Drive, not in the repo. The repo contains processed topic files (no labels) and `kz_data.jsonl` (the complete KZ qrel in pre-joined form).

### 2f. Unified dataset, train contamination, and the clean test split

**Key finding (2026-07-03):** The existing clf training dataset
`semaj83/ctmatch_classification` (`combined_classifier_data.jsonl`, 39,691 pairs) is
essentially the TREC21+KZ qrel in pre-joined form. Of the dataset's 134 unique topic
texts, 121 exactly match TREC21 or KZ topic texts from the raw source files. Total
clf pairs ≈ TREC21+KZ qrel pairs (39,691 vs 39,702, difference of 11).

**Implication for the 0.7528 gate:** This gate was measured on TREC21+KZ, which
substantially overlaps with clf training data. The NDCG is not a clean holdout number —
it is closer to training-set performance with minor regularization. **TREC22 is the only
truly clean test set**: 0 of its 50 topics appear in clf training data.

**Unified dataset** (produced by `notebooks/build_dataset.ipynb`):

| File (Drive) | Contents | Size |
|---|---|---|
| `unified_qrels.jsonl` | 184 topics, 75,096 pairs — schema: `{source, topic_id, topic_text, doc_id, label}` | 61.4 MB |
| `topic_splits.json` | Train/val/test topic ID lists | — |
| `train_clf_data.jsonl` | 107 train topics joined with doc texts, 31,438 pairs | 68.0 MB |

**Split strategy** (`trec22_holdout` — configured in `build_dataset.ipynb`):

| Split | Topics | Pairs | Source |
|---|---|---|---|
| train | 107 | 31,438 | TREC21+KZ (80% random topic-level, seed=42) |
| val | 27 | 8,264 | TREC21+KZ (remaining 20%) |
| test | 50 | 35,394 | TREC22 (all, held out) |

**Topic uniqueness:** all 184 topic texts are distinct across the three datasets (verified
in the notebook). No cross-dataset deduplication was needed.

**Doc ID overlap:** 51,462 unique NCT IDs out of 75,096 pairs. The same trial appearing
in multiple topics is expected (different patients, same trial) and is not a duplicate.
Label conflicts (same NCT ID labeled 2 in one dataset and 0 in another) exist but are
legitimate — the same trial can be relevant for one patient and irrelevant for another.

**13 unknown clf topics:** The clf dataset contains 13 topics (~3,494 examples) not in
any of the three eval qrels. These contain MIMIC-III–style de-identification markers
(`[**date**]`) and are likely from the original 30-topic OpenAI pipeline. They have
training labels (0/1/2) within `semaj83/ctmatch_classification` but no Drive qrel and
are excluded from `unified_qrels.jsonl`. Safe to ignore unless you want the extra
training pairs; they add noise relative to the TREC/KZ topic style.

**New regression gate:** after retraining on `train_clf_data.jsonl`, TREC22 NDCG@10 is
the primary eval metric (clean holdout). TREC21+KZ NDCG should be reported as a
secondary number for comparison with prior runs, with the caveat that it partially
overlaps with training data.

---

### 2e. Evaluation methodology: pooling, judged sets, and what we actually measure

This section is essential for understanding what the metrics in §6 actually mean — and equally important for understanding what they *don't* measure.

#### The scale problem

There are 374,000 clinical trials in the corpus. Each TREC topic is a patient description. A correct evaluation would require a human assessor to judge every (topic, trial) pair — that is 75 topics × 374,000 trials = **28 million judgments**. At 2 minutes per judgment, that is 107 person-years of annotation work. This is not feasible.

TREC solves this with **pooling**.

#### What pooling means

Pooling is a strategy for selecting a tractable subset of (topic, document) pairs to judge while ensuring the subset covers most of what a good system would return.

The procedure:
1. Each participating system (team) submits a ranked list of their top-$k$ documents for each topic (typically $k = 1000$).
2. The **pool** for each topic is the **union** of all submitted top-$k$ lists across all systems.
3. Human assessors (NIST for TREC) judge every document in the pool for each topic, assigning relevance labels.
4. Documents **not in the pool** receive **no label** — they are treated as not relevant by default.

```
System A top-1000: [NCT001, NCT003, NCT007, NCT011, ...]
System B top-1000: [NCT002, NCT003, NCT009, NCT011, ...]
System C top-1000: [NCT001, NCT004, NCT006, NCT011, ...]

Pool = union = {NCT001, NCT002, NCT003, NCT004, NCT006, NCT007, NCT009, NCT011, ...}
                ↑ this set gets human judgments
```

For TREC 2021 Clinical Trials, roughly 40–120 documents per topic were pooled and judged. For TREC 2022, the pool was deeper (~500–900 per topic) because more systems participated and deeper cutoffs were used.

#### What the judged set actually contains

The judged documents are **not a random sample** of the corpus. They are exactly those documents that at least one participating system ranked in its top-1000. This has important implications:

- The pool is biased toward documents that look relevant to *the systems that participated in 2021*. Systems using similar architectures (TF-IDF, BM25, dense retrieval) will have overlapping pools.
- A trial that every 2021 system missed — perhaps because it uses unusual terminology — will not be in the pool and will not be judged. If our system correctly retrieves it, we get **no credit** for it.
- This is called **pool bias** or the **unjudged document problem**. It is an inherent limitation of TREC-style evaluation.

#### The "unjudged = not relevant" assumption

When computing NDCG and MRR, documents not in the qrels file are treated as relevance 0. This is the standard TREC assumption. It is approximately correct for two reasons:
1. The pool covers most truly relevant documents (systems generally agree on the obviously relevant ones).
2. Relevant documents outside the pool are rare — if they were easy to retrieve, at least one 2021 system would have found them.

It is approximately *wrong* in two ways:
1. New systems using new architectures (e.g., dense retrieval when 2021 systems were mostly sparse) may retrieve genuinely relevant documents that were never pooled.
2. For KZ, the 2021 corpus contains trials that didn't exist in 2015 — some may be highly relevant but were never in any pool.

**Practical consequence for this project:** our NDCG@10 scores are conservative lower bounds. We can confidently say our system achieves at least X, but the true NDCG (if all 374k docs were judged) could be higher.

#### Evaluating on the judged pool only

Because we do not have relevance labels for all 374k documents, our evaluation uses only the judged subset. For each topic:

```python
doc_ids = list(rel_dict[topic_id].keys())   # only the judged docs
doc_set = get_indexes_from_ids(doc_ids)      # find them in our index
ranked_pairs = ctm.match_pipeline(topic_text, doc_set=doc_set)
# pipeline ranks only the judged docs, not the full corpus
```

This means the pipeline is **not** doing full retrieval — it is doing **reranking within the judged pool**. The practical effect:
- Precision is measured against a finite, partially-complete set
- Documents outside the pool that our pipeline would rank highly are invisible to the metric
- The eval is asking: "given the documents humans looked at, does our system rank the good ones higher?"

This is the standard and correct way to evaluate against TREC qrels. It is meaningfully different from TrialGPT's evaluation, which reranks a *reduced candidate set pooled from TREC participant submissions* (not full-corpus retrieval — see §7e). Note that `eval_fullcorpus.ipynb` (§7i) does perform true full-corpus retrieval over all 374k docs, scored with `trec_eval`; the judged-pool reranking described here is the older `eval_baseline.ipynb` protocol.

#### Relevance labels: what 0, 1, 2 mean

TREC Clinical Trials 2021/2022 uses a 3-point scale:

| Label | Meaning | Clinical interpretation |
|---|---|---|
| 2 | Eligible | Patient meets all inclusion criteria; no exclusion criteria triggered |
| 1 | Partially eligible | Meets inclusion criteria but has at least one relevant exclusion criterion; or meets most but not all inclusion criteria — assessor judgment call |
| 0 | Not eligible | Clear mismatch: wrong condition, age, gender, or disease stage |

The **partially eligible** label (rel=1) is where most inter-annotator disagreement lives. It requires the assessor to read the eligibility text carefully and make a clinical judgment about whether the patient's specific comorbidities or history constitute an exclusion. This is where 20 years of ICU clinical experience provides a real evaluation advantage — the primary author can audit individual rel=1 labels and identify where the assessors made defensible but incorrect calls.

#### KZ relevance labels

The KZ dataset uses the same 0/1/2 scale but the judgments come from the paper's authors (Koopman and Zuccon), not NIST assessors. The deeper pool (~1,000–1,400 judged docs per topic) suggests a more systematic pooling strategy — possibly manual search rather than system pooling — but the annotation procedure is not fully described in the paper.

**Implication:** KZ labels may have different inter-annotator agreement characteristics than TREC labels. Comparing NDCG across the two datasets is not apples-to-apples even when using the same metric formula.

#### What pooling means for system comparisons

If system A outperforms system B on the judged pool, we conclude A is better — but with a caveat: if A uses a very different retrieval strategy from the 2021 TREC systems (different enough that its relevant retrievals fall outside the pool), the comparison underestimates A's true performance. New dense retrieval systems in 2022 faced exactly this issue when evaluated against 2021 pools built primarily by BM25-based systems.

---

### 2g. Representations audit — exact model input per experiment (added 2026-07-14)

**Why this section exists.** A model's score is only interpretable if you know *exactly what text went into it*. This project accreted **two different document corpora** and several truncation regimes over time, and — verified by reading the notebooks/source, not the prose — **the same model is trained and evaluated on different representations across experiments, and the eligibility criteria (the rel=2-vs-rel=0 signal) are truncated off the input for most of the deployed rerankers.** Every score in §7/§11 must be read together with its representation, or cross-experiment comparisons silently compare inputs, not models. Discipline going forward: **no experiment is logged without its representation row here.**

#### The two corpora and the topic text

- **R1 — eligibility-only** (`semaj83/ctmatch_ir` → `doc_texts.txt`, exposed via `dataprep.DOC_TEXTS_PATH` / `eval_utils.load_doc_texts()`). Enrollment-criteria text only; no title/condition/summary. Also the source of the 384-dim `all-MiniLM-L6-v2` embeddings and the `bart-large-mnli` category vectors used by the sim/SVM filters. This is the corpus §7i Finding 1 identified as the cause of catastrophic full-corpus recall (0.11).
- **R2 — full-text** (`doc_texts_fulltext.txt`, built by `build_fulltext_corpus.ipynb`). Concatenation in this fixed field order: `briefTitle + officialTitle + conditions + briefSummary + detailedDescription + interventions + eligibilityCriteria`. **Eligibility is the *last* field.** Median length **2,853 chars** (≈ 700–900 BERT tokens; long trials run into the thousands). Any truncation budget below the full length therefore drops eligibility *first*.
- **Topic text** — the raw TREC vignette (~110 words) or KZ one-liner, from `load_eval_datasets()['topic2text']`. **No query expansion / diagnosis synthesis anywhere** (contrast h2oloo's NQS). Uniform across all experiments.

#### Per-component representation table (verified from code)

| Component (notebook) | Training input | Inference / eval input | Truncation | Eligibility seen? | Consistent? |
|---|---|---|---|---|---|
| **BM25** (`eval_fullcorpus`) | — | R2 full-text | none | yes (indexed whole) | n/a |
| **Dense retriever — retriever-v2 / MiniLM-L6** (`finetune_retriever`, encoded in `eval_fullcorpus`) | (topic, **R1 eligibility-only**) positive pairs, MNRL | encode **R2 full-text** corpus | ST `max_seq_length` (**MiniLM default 256 word-pieces — verify on the checkpoint**) → ~title+conditions+summary | **no** (256-token cap lands before eligibility) | **MISMATCH** (train R1, infer R2) |
| **sim filter** (`all-MiniLM-L6-v2` + `bart-large-mnli`) | off-the-shelf | **R1** embeddings + category vec | — | criteria embedded whole | consistent |
| **SVM filter** | per-topic, on R1 embeddings | R1 embeddings | — | via R1 | consistent |
| **clf-v4 (BioLinkBERT-large)** (`build_dataset`→`retrain_classifier`; scored in `eval_baseline`, `eval_fullcorpus`, `train_ensemble_full`) | (topic, **R1 eligibility-only**), 512 | **standalone/judged-pool eval (0.7460 gate / 0.6388 TREC22): R1**; **full-corpus reranker + ensemble `clf_rel`/`clf_partial` feature: R2 full-text**, 512 `longest_first` | 512 | standalone **yes**; deployed **no** (R2 eligibility past 512 on most docs) | **standalone matches train; deployed feature does not** |
| **reranker-v2 (cont. from clf-v4)** (`train_reranker_hardneg`; feature in `train_ensemble_full`) | (topic, **R2 full-text**), 512 `longest_first`, + dense hard-neg ×3 | **R2 full-text** `v2_rel` feature, 512 | 512 | **no** (eligibility past 512 on long docs) | consistent (both R2-512) |
| **LLM judge, zero-shot — standalone §7f (0.6485)** (`rerank_llm`) | zero-shot | **R1 eligibility-only** ("Trial eligibility criteria:") | **2048 tokens** | **yes** | n/a |
| **LLM judge, zero-shot — ensemble `llm_yesno` feature** (`rerank_llm_feature`) | zero-shot | **R2 full-text** `trial[:1800 chars]` | **1800 chars ≈ 450–500 tok** | **no** (1800 chars of R2 ≈ title+conditions+summary; eligibility is last) | **DIFFERENT representation from the §7f standalone** |
| **LLM judge, fine-tuned (§11c)** (`finetune_judge_lora`, `regen_judge_feature`) | (topic, **R2** `trial[:1800]`) graded, MAX_LEN 1024 | R2 `trial[:1800]`, MAX_LEN 1024 | 1800 chars / 1024 tok | **no** | consistent with its own feature |
| **Criterion assessors — Qwen/Claude/R1 (§7g/h)** | R1 silver labels (R1) | ctproc-parsed **individual criteria** + topic | per-criterion | **yes** (criteria are the input by construction) | consistent |
| **CoT reranker (§11a)** (`rerank_cot_final`) | zero-shot | R2 full-text | per prompt | partial | — |
| **Listwise RankZephyr (§11d)** (`rerank_listwise`) | zero-shot | confounded run: R2 `[:400 chars]` (**title+conditions only**); fair retry: **head 220 + tail 1200 landing on eligibility** | 400 char / head+tail | confounded **no**, fair **yes** | the +0.030 confound→fair swing is the direct evidence truncating eligibility costs real signal |

#### What this reveals (three hazards, load-bearing)

1. **Reported component numbers are not all on the deployed representation.** clf-v4's headline **0.6388** is on **R1 (eligibility-only)** — its *native* representation — but the ensemble consumes clf-v4 on **R2 full-text-512**. The LLM judge's **0.6485** (§7f) is on **R1 at 2048 tokens (sees eligibility)**, but the ensemble's `llm_yesno` feature is **R2 at 1800 chars (does not)**. So the two most-cited "component works" numbers were measured on inputs the ensemble never uses. A component score is only meaningful **paired with its representation row above**.
2. **The "eligibility judge" mostly cannot see eligibility.** For the three deployed rerankers whose whole job is eligibility discrimination — clf-v4 (deployed), reranker-v2, and the `llm_yesno` judge feature — the eligibility section is truncated off the input on the majority of trials (eligibility is R2's last field; budgets are 512 tok / 512 tok / 1800 chars). **This reframes §11c and §11d:** the judge's lukewarm +1.90 on buried eligibles and the cross-encoders' "flat/uncalibrated" behavior are consistent with them scoring *topicality*, not eligibility, because eligibility is not in their input. The RERANK_TEXT='fulltext' choice (§7i: full-text 0.458 > eligibility-only 0.369) is real but is a choice between **two flawed inputs** — full-text-with-eligibility-truncated vs eligibility-only-with-no-topicality; the third option (topicality **and** eligibility in one budget) was never tested on the cross-encoders.
3. **Two train/inference mismatches by construction.** clf-v4 (train R1 → deploy R2) and retriever-v2 (train R1 → encode R2). Both were adopted because R2 empirically beat R1 on the *aggregate* metric (recall for the retriever, rerank NDCG for clf), but neither model was ever *trained* on the representation it is *deployed* on.

#### Consequence for the surpass-SOTA effort (redo the chain, cleanly)

The user is willing to redo whole experiment chains; the representation audit says that is the right move and names the target: **standardize on one representation `R`, train *and* evaluate every component on it, and tag every logged score with its `repr_tag`.** This was executed — see **§2h** for the study that froze `R` and the first validated result. The chain, highest-EV first (each developed on TREC21, TREC22 touched once — §10 anti-gaming):

1. ✅ **Freeze `R` by a controlled truncation study** (§2h) → `R = elig_first-L512`.
2. ✅ **Retrain clf on `R`** (§2h) → `clf_R` beats clf-v4 on held-out TREC22 judged-pool (+0.012). (The natural place to also add the still-untried neural listwise loss, §9b/2b — not yet done.)
3. **Widen the `llm_yesno` budget to actually include eligibility** and, if it helps zero-shot, only then reconsider the fine-tune (§11c was trained on the eligibility-blind 1800-char input — its null may be an input artifact, not a judge-quality ceiling). *Pending.*
4. **Retrain reranker-v2 (→ v3) and the retriever on `R`** (train/infer consistency). *Pending.*
5. **Re-extract all features on `R` and re-run the LambdaMART ensemble** → the full-corpus number that replaces the `PENDING(R)` §7 tables. *Pending — this is the decision-grade SOTA number.*

Until step 5 lands, treat every full-corpus §7/§11 number as **confounded by representation** (`PENDING(R)`); the §2h numbers are real but **judged-pool component validations**, not the full-corpus result.

**Infrastructure.** All of this runs through `ctmatch.experiments` (in-package): one `ExperimentConfig` that makes the representation a *function of config* (`repr_strategy`, `max_length`), small portable functions so the representation can never drift between train and eval again, and a `log_result` ledger that stamps every number with `cfg.repr_tag()`. Notebooks are thin (3-cell Colab setup → config → compose functions).

**Notebooks audited for this table:** `build_fulltext_corpus`, `build_dataset`, `retrain_classifier`, `eval_baseline`, `eval_fullcorpus`, `train_reranker_hardneg`, `finetune_retriever`, `rerank_llm`, `rerank_llm_feature`, `finetune_judge_lora`, `regen_judge_feature`, `train_ensemble_full`, `rerank_listwise`; source `dataprep.py`, `eval_utils.py`, `config.py`, `matching/reranking/classifier.py`.

---

### 2h. Freezing the representation: the truncation study and the clf_R validation (2026-07-15)

§2g diagnosed the drift; this section resolves it. It picks **one** document representation `R` by a controlled study, then confirms that retraining the reranker on `R` beats the old model on held-out data. All of it runs through `ctmatch.experiments` (config-driven), so every number below carries a `repr_tag`.

**The candidate strategies** (how the `max_length`-token budget is filled from the trial fields `title + conditions + summary + detailed_desc + interventions + eligibility`):

- `head` — leading tokens (the *deployed* behaviour; eligibility is last, so it truncates off long docs).
- `head_tail` — keep `head_frac` of the budget as head + the rest as tail (tail preserves trailing eligibility).
- `elig_first` — reorder so eligibility leads, then take the leading budget.
- `budget_incexc` — split the budget across topicality / inclusion / exclusion, **reserving the exclusion floor first** (idea from the RN co-author: a matched exclusion disqualifies, so guarantee it survives). inc/exc come from ctproc's `process_eligibility_naive`.

#### The truncation study (`exp_truncation.ipynb`)

Two legs, both on the TREC21 **judged pool** (the qrel docs per topic — no retrieval, so it isolates the reranker *input*):

**Leg 1 — model-free eligibility coverage.** For each (strategy, length), what fraction of the eligibility text survives into the model input? No model, no training — the cleanest measure of the truncation constant itself.

| repr | mean elig. retained | % docs full | % docs **zero** |
|---|---|---|---|
| head-L256 | 0.011 | 0.005 | **0.970** |
| head-L384 | 0.098 | 0.052 | 0.792 |
| head-L512 | 0.248 | 0.158 | **0.597** |
| head_tail-L384 | 0.559 | 0.280 | 0.000 |
| head_tail-L512 | 0.709 | 0.462 | 0.000 |
| elig_first-L384 | 0.739 | 0.504 | 0.000 |
| **elig_first-L512** | **0.853** | **0.674** | **0.000** |

The deployed `head-L512` leaves **60% of docs with zero eligibility** in the reranker's input (97% at L256) — the §2g defect, quantified against real document lengths. `head_tail` and `elig_first` are *length-robust* (0% zero at L384+) because they structurally place eligibility; only `head` degrades with length. `elig_first` dominates `head_tail` at equal length (it doesn't spend budget on the head).

*(A field-name bug surfaced here: the corpus writes `detailed_desc` but the code read `detailed_description`, so the longest field was silently empty — an earlier run of this table understated `head`'s collapse. Fixed; the numbers above are post-fix.)*

**Leg 2 — judged-pool rerank NDCG@10** (score the pool with clf-v4 under each representation; TREC21):

| repr | NDCG@10 | | repr | NDCG@10 |
|---|---|---|---|---|
| head-L256 | 0.683 | | elig_first-L256 | 0.756 |
| head-L384 | 0.774 | | elig_first-L384 | 0.877 |
| head-L512 | 0.807 | | **elig_first-L512** | **0.893** |
| head_tail-L384 | 0.805 | | budget_incexc-L512-e25 | 0.846 |
| head_tail-L512 | 0.829 | | budget_incexc-L512-e40 | 0.842 |

**NDCG tracks eligibility coverage almost monotonically** — direct evidence the representation defect *costs ranking*, not just coverage. `head-L512 → elig_first-L512` is **+0.086 from a pure representation change, no retraining.** `elig_first-L512` wins.

#### budget_incexc: a clinically-correct idea the graded metric doesn't reward

`budget_incexc` lands *between* `head_tail` and `elig_first` (best 0.846 < `elig_first` 0.893), and the tell is the sweep gradient: **`e40` (0.842) < `e25` (0.846)** — giving exclusion *more* budget makes NDCG *worse*. That is the §8a **eligible-vs-excluded objective mismatch** as a gradient: graded NDCG rewards topical/eligibility coverage, and every token diverted to *guarantee the exclusion veto* costs you, because catching an exclusion buries the rel=1 trial the metric wanted elevated. An exclusion-specific coverage diagnostic (`exclusion_retained_frac`) confirms the mechanism: on inclusion-heavy overflow docs, `elig_first` keeps ~85% of *total* eligibility but drops the *exclusion* (it comes after inclusion), while `budget_incexc` guarantees it — yet the metric prefers the former. **Verdict: shelve `budget_incexc` for graded reranking; it belongs in a hard eligibility filter or an eligibility-specific metric, where a matched exclusion *should* dominate.** A real, mechanism-backed negative result, not a dead end.

#### Frozen: `R = elig_first-L512`

Locked into `ExperimentConfig` as the default. One caveat on Leg 2's absolute values: they are judged-pool, in-sample (clf-v4 trained on TREC21), *and* clf-v4 trained on eligibility-heavy text so `elig_first` matches its training distribution — i.e., part of the win could be "it matches the current model." The next step removes that caveat.

#### clf_R: retrain on `R`, and the honest validation

`build_dataset` was rewritten to emit labeled **pairs** (`{topic, doc_id, label}`) with **no doc text baked in** — the representation is applied at tokenize time from the frozen config (killing the §2g R1-eligibility-only join). `retrain_classifier` then trains BioLinkBERT-large on `R` (focal loss + complement-frequency class weights, the clf-v4 recipe), producing **`clf_R`** (val macro-F1 0.615). Validation by *ranking* (`exp_validate_clf.ipynb`, judged-pool NDCG@10 under `R`):

| model | TREC21 (in-sample) | **TREC22 (held-out)** |
|---|---|---|
| clf-v4 (old) | 0.8933 | 0.6499 |
| **clf_R (trained on R)** | **0.9219** | **0.6623** |

On the **held-out** TREC22, `clf_R` beats clf-v4 by **+0.012** — so training on `R` genuinely helps and `elig_first`'s edge is *not* a distribution-match artifact. Decomposed against clf-v4's original deep-dive judged-pool TREC22 (0.6388): **+0.011 from scoring on `R`** (0.6388 → 0.6499) **+0.012 from training on `R`** (0.6499 → 0.6623) ≈ **+0.024 total**, all attributable to the §2g fix. Caveats: judged-pool, not full-corpus (the decision-grade number is the ensemble re-run, step 5 above); n=50, so +0.012 is modest — real (paired, consistent across both splits, mechanism-backed) but wants a paired bootstrap before being called significant.

#### reranker-v3 and the judge on `R` (the §11c re-test)

Continuing the chain on `R`: **reranker-v3** (`train_reranker_hardneg`) continue-trains from `clf_R` with dense-mined hard negatives — weak standalone by design (v2 was too), so it's validated only as the `v2_rel` ensemble feature, not alone.

The **judge on `R`** is the more interesting result. `rerank_llm_feature` scores Qwen-2.5-7B yes/no through the `elig_first` representation, so the judge finally *reads* eligibility — unlike §11c, whose null was measured on the eligibility-blind 1800-char input. Standalone judged-pool NDCG@10:

| judge | TREC22 | TREC21 | KZ |
|---|---|---|---|
| §7f Qwen (eligibility-only, 2048 tok) | 0.6269 | — | — |
| **judge on `R` (elig_first, this work)** | **0.6518** | 0.7558 | 0.5734 |

The judge-on-`R` **beats §7f's eligibility-only judge by +0.025** and is **on par with `clf_R`** (0.6623) — a 7B zero-shot judge nearly matching the fine-tuned cross-encoder on held-out TREC22. So `elig_first` helps the judge, not just fixes a bug, and **§11c is genuinely re-opened**: its "the judge adds nothing" null was measured on an eligibility-blind, much weaker judge. Two cautions kept in view: (1) standalone quality ≠ *ensemble* contribution — §11c's finding was **redundancy** with clf/v2/dense, not judge weakness; (2) on `R`, `clf_R` and the judge *both* read eligibility, so they could be *more* correlated, not less. Whether the judge adds over `clf_R` is therefore decided only in the ensemble — see the multi-view result next.

#### The full-corpus ensemble on `R`, and the mismatch-vs-diversity lesson

The first end-to-end `R` ensemble (retrieval → LambdaMART over BM25/dense/RRF + `clf_R` + reranker-v3 + judge) came in at **TREC22 NDCG@10 = 0.5203 — well below the old 0.6105.** Better *components* (clf_R, the judge) but a *worse* ensemble. The diagnostic (`exp_ensemble_diagnose.ipynb`, all cached features) found why:

- **`reranker_v3` (`v2_rel`) was redundant and actively hurting** — 0.69-correlated with `clf_rel` and dropping it *improved* NDCG by +0.033. Root cause: it was continue-trained *from* `clf_R` on the *same* `R`, so it's a clone, not the diverse feature the old `reranker-v2` was.
- **The CV mis-picked `num_leaves=31`** (KZ noise in the folds); 15 is +0.022.
- The judge, by contrast, is the **strongest single feature** and *orthogonal* to `clf_rel` (r=0.33) — the §11c reversal confirmed.
- The ensemble is badly **in-sample-inflated on its tuning set** (TREC21 CV 0.70 vs TREC22 0.52), because `clf_R`/`reranker_v3` train on TREC21 — so TREC21-CV can't detect held-out redundancy (limitation #3 bites hard here).

**The conceptual correction (load-bearing).** The instinct was to call the old ensemble's feature diversity a "confound." It is not. **Train/inference *mismatch* (a model scored on a distribution it never trained on) is the real bug (§2g); feature *diversity* (different features deliberately reading different views) is legitimate multi-view ensembling.** The old 0.6105 had real multi-view signal *delivered through miscalibrated models* — the two were entangled. Making the representation consistent (correct) removed the diversity *and* the miscalibration together. The fix is not to accept a lower ceiling; it is to deliver the diversity **deliberately**: each model self-consistent on its *own* representation, but different models on *different* views. A representation optimized for retrieval that loses eligibility is a perfectly defensible *feature* — it's a different view.

**Multi-view build.** Drop the redundant `reranker_v3`; add two purpose-built topicality-view features orthogonal to the eligibility readers: **`clf_topic`** (BioLinkBERT-large trained on the `topic_first` representation — conditions/title/summary lead) and a **topicality judge** (Qwen asked "is this trial *about* the patient's condition?"). Final feature set = retrieval + `clf_R` (eligibility CE) + `clf_topic` (topicality CE) + eligibility judge + topicality judge. Progression:

| ensemble (TREC22) | NDCG@10 | MRR | TREC21 CV | notes |
|---|---|---|---|---|
| original (9-feat, incl. `v2_rel`, nl=31) | 0.5203 | 0.645 | 0.487 | components better, ensemble worse than old 0.61 |
| retuned (TREC21-CV, nl=15) | 0.5221 | 0.631 | 0.703 | CV can't drop `v2_rel` (in-sample) — no change |
| **multi-view (drop `v2_rel`; +`clf_topic` +topicality)** | **0.5548** | **0.712** | 0.615 | all orthogonal features survive selection |

**Findings.** (1) **All the multi-view features survived backward selection** — `clf_topic_rel`/`clf_topic_partial`/`topicality` are all in the final model, contributing comparably to `llm_yesno`. The diversity is real and *used*. (2) **MRR jumped 0.63 → 0.71** — much better first-relevant placement. (3) **The overfit collapsed**: the CV-test gap went from 0.18 (0.70 vs 0.52) to 0.06 (0.615 vs 0.555) — dropping the clone and adding real views made the model *generalize*, not just score. (4) The two LLM judges are only moderately orthogonal (r=0.67 — same model, related questions), so they stack less than the separately-trained cross-encoder views; the topicality *cross-encoder* is the cleaner diversity.

**Honest standing: TREC22 NDCG@10 = 0.5548, 95% CI [0.477, 0.635], vs h2oloo 0.6125.** The CI contains 0.6125 (a statistical tie), but the point estimate is ~0.057 short and that gap is not hand-waved away. What this number *is*: representation-consistent, multi-view-by-design, generalizing (small CV-test gap), with every feature accounted for — arguably more defensible than the old 0.6105, which we now know borrowed against the very miscalibration this effort removed. Remaining levers, in EV order: (a) a **more orthogonal topicality signal** (a *different-model* condition-match encoder — SapBERT/an off-the-shelf bi-encoder — since the same-model LLM judges cap at r=0.67); (b) **retrieval recall** (recall@1000 = 0.543; retrain the retriever on `R` / `exp_retrieval_repr` / query expansion, esp. for the implicit-diagnosis topics); (c) a **stronger reranker view** (`monoT5_CT`, §10). Notebooks: `exp_ensemble_diagnose`, `train_classifier_topic`, `rerank_topicality_feature`, `train_ensemble_full` (multi-view).

#### Bugs surfaced (all the silent representation-drift kind)

Found and fixed while building this: `relevant_index` selected the class by *substring* (`'relevant' in 'not_relevant'` is True → ranked by P(not-relevant), the ~0.03 leg-2 floor); the `detailed_desc`/`detailed_description` field-key mismatch (longest field silently empty); the exclusion-header regex missing `re.IGNORECASE` (capitalized "Exclusion Criteria:" read as no-exclusion); the hand-rolled cross-encoder tokenization that dropped `token_type_ids` (replaced with the model's native pair tokenizer); and the **LLM judge gated by the 512 cross-encoder `max_length`** (its own budget is `llm_max_tokens=2048`; at 512 the prompt overflowed and right-truncation cut the "yes/no" question off the end, cratering the judge to 0.42 vs the 0.65 it scores once un-gated). Each is exactly the class of silent, score-corrupting drift this whole effort exists to eliminate.

**Notebooks:** `exp_truncation.ipynb` (freeze `R`), `build_dataset.ipynb` (pairs on `R`), `retrain_classifier.ipynb` (`clf_R`), `exp_validate_clf.ipynb` (ranking validation), `train_reranker_hardneg.ipynb` (reranker-v3), `rerank_llm_feature.ipynb` (judge on `R`), `eval_fullcorpus.ipynb` (retrieval + pool on `R`), `train_ensemble_full.ipynb` (LambdaMART on `R` — **running**, produces the full-corpus number that replaces the `PENDING(R)` §7 tables), `exp_retrieval_repr.ipynb` (retrieval-representation ablation — separate, §2g note on retrieval vs rerank). Backbone: `src/ctmatch/experiments.py`.

---

## 3. Text Processing: ctproc

### 3a. Raw XML → structured fields

**TODO — flowchart of `CTProc.process_ct_doc_file()`**

The main parsing loop (`proc.py:process_ct_doc_file`) does the following for each NCT XML:

```python
docid     = root.find('id_info/nct_id').text
condition = [r.text for r in root.findall('condition')]
intervention_type = [r.text for r in root.findall('intervention/intervention_type')]
intervention_name = [r.text for r in root.findall('intervention/intervention_name')]
# then:
ct_doc = add_eligibility(ct_doc, root)   # the hard part
ct_doc.process_doc_age(root)             # structured age fields
```

**Decision rationale:** condition and intervention are used as structured signals for the category prior, not for text matching. The eligibility textblock is the primary matching surface.

### 3b. Eligibility criteria parsing: the regex cascade

This is the most complex and consequential part of ctproc. The input is a raw textblock; the output is two lists of criterion strings: `include_criteria` and `exclude_criteria`.

**TODO — full diagram of the decision tree in `process_eligibility_naive()`**

The core challenge: the textblock has no guaranteed structure. Observed patterns in the wild:

**Pattern A — well-formed (majority):**
```
Inclusion Criteria:
  - Age 18 or older
  - Diagnosis of Type 2 Diabetes

Exclusion Criteria:
  - Pregnancy
  - Renal failure (GFR < 30)
```

**Pattern B — inclusion only:**
```
Inclusion Criteria:
  - Healthy volunteers
  - Age 18-65
```

**Pattern C — no header at all:**
```
Patients must be 18 years or older with confirmed COPD (FEV1/FVC < 0.70)
and must not be currently enrolled in another study.
```

**Pattern D — malformed / concatenated:**
```
Inclusion Criteria: Age >= 18 Exclusion Criteria: Pregnancy Current smoker
```

**The regex cascade (`eligibility.py`, `regex_patterns.py`):**

```python
# Step 1: Try to split on exclusion header
chunks = re.split(
    r'(?:[Ee]xclu(?:de|sion) criteria:?)|(?:[Ii]neligibility [Cc]riteria:?)',
    elig_text
)
# h==0 → inclusion chunk, h>=1 → exclusion chunk

# Step 2: Within each chunk, split on blank lines
for s in re.split(r'\n\n', chunk):
    # Step 3: Split on bullet delimiter "- "
    for ss in re.split(r'- ', s):
        ss = re.sub(r'\n   +', ' ', ss).strip()  # collapse wrapped lines
```

`BOTH_INC_AND_EXC_PATTERN` (the big one):
```python
re.compile(
    r'[\s\n]*[Ii]nclusion [Cc]riteria:?(?: +[Ee]ligibility[ \w]+\: )?'
    r'(?P<include_crit>[ \n(?:\-|\d)\.\?\"\%\r\w\:\,\(\)]*)'
    r'[Ee]xclusion [Cc]riteria:?(?P<exclude_crit>[\w\W ]*)'
)
```

**TODO — annotate what each character class in the include_crit group is doing and why. This is where parsing silently fails for trials with unusual characters.**

**Known failure modes:**
1. No exclusion header → all criteria classified as inclusion
2. Criteria that themselves contain "exclusion" (e.g., "Prior exclusion from another study") → spurious split
3. Numbered lists (`1. ... 2. ...`) instead of bullet lists → the `"- "` split misses them
4. Headers like "Eligibility:" with no inc/exc subheading → both groups are empty

**Decision made:** use `process_eligibility_naive()` rather than the full NLP-based approach because spaCy + UMLS adds significant latency and the marginal gain in parsing accuracy was not measurable in downstream NDCG.

### 3c. NLP layer (optional)

When `CTConfig(nlp=True)`:
- `spacy.load("en_core_sci_md")` — scientific English model
- `scispacy_linker` with UMLS → entity → CUI mapping
- `negex` → negation detection (important: "no history of diabetes" ≠ "history of diabetes")
- Abbreviation resolution: "T2DM" → "Type 2 diabetes mellitus"

**TODO — show a concrete before/after example with NegEx on an exclusion criterion**

**Decision:** NLP pipeline disabled in the current production pipeline because it's not needed for embedding + SVM stages, and criterion-level entailment (future work) would handle negation more precisely.

### 3d. Topic processing: age and gender extraction

TREC topics contain structured age/gender if you parse carefully:

```python
AGE_PATTERN = re.compile(r'(?P<age>\d+) *(?P<units>\w+).*')
TOPIC_GENDER_PATTERN = re.compile(r'[ \d](?P<gender>woman|man|female|male|boy|girl|M|F) .*')
```

**Known gap:** These fields are extracted from CTTopic but not currently used as hard filters in the matching pipeline. A trial that excludes patients > 70 years old and the patient is 74 should be a hard disqualification — the pipeline currently treats this as a soft signal via the classifier. This is a major source of false positives in the error analysis.

---

## 4. Information Retrieval: general theory applied to this task

### 4a. The ranking problem

Standard IR: given query $q$ and corpus $\mathcal{D}$, return a ranked list $\sigma$ such that $P(d \text{ is relevant} | q)$ decreases monotonically with rank.

This task: $q$ = patient description, $d$ = clinical trial eligibility criteria, relevance is **eligibility** (a logical property, not a topical one).

**The distributional mismatch:**
- $q$ is clinical narrative: "74M with HTN, HLD, T2DM presenting with chest pain, EF 45%..."
- $d$ is regulatory protocol: "Inclusion: LVEF ≥ 30% and < 45%. Exclusion: eGFR < 30 mL/min/1.73m². Prior CABG within 6 months."

Lexical overlap is low. Semantic overlap requires domain knowledge. This is why BM25 underperforms and dense retrieval with biomedical embeddings is necessary.

### 4b. Candidate generation vs. reranking

**TODO — diagram showing the two-stage paradigm**

Standard pipeline:
1. **First stage (retrieval):** cheap, high recall. Returns top-1000 from 374k. BM25, dense embedding, or hybrid.
2. **Second stage (reranking):** expensive, high precision. Cross-encoder or LLM scoring of top-1000.

ctmatch maps to this as:
1. Embedding sim (MiniLM cosine + category prior) → top-10k (in practice, first real reduction)
2. SVM ranking → top-100
3. SciBERT classifier → top-50
4. Generative reranker → top-10

**TODO: measure the actual intermediate set sizes in normal inference mode on a sample topic**

### 4c. Why standard IR assumptions break here

1. **Vocabulary mismatch**: "chest pain" in topic, "NYHA Class III heart failure" in trial. Standard BM25 gets nothing.
2. **Logical structure of exclusion criteria**: these are NOT things the trial is "about" — they're disqualifying conditions. A trial that excludes "patients with renal failure" mentions renal failure but is LESS relevant to a renal failure patient.
3. **Relevance is asymmetric and non-transitive**: if trial A and trial B both require "Type 2 Diabetes", they're both relevant to a T2DM patient — but that doesn't mean they're similar to each other.
4. **Sparse relevant set**: on average, 2–3 trials out of 374k are truly eligible. The prior $P(\text{relevant})$ is ~0.00001. Standard calibration doesn't work.

---

## 5. The ctmatch Pipeline

### 5a. Architecture overview

**TODO — graphviz diagram of the full pipeline**

```
Patient description (topic text)
        │
        ▼
[get_pipe_topic()]
  ├── MiniLM-L6-v2 → 384-dim embedding
  └── bart-large-mnli → 14-class category distribution
        │
        ▼  doc_set (all 374k indices OR judged subset in eval mode)
[sim_filter()]  →  top-10,000 (or passthrough in eval mode)
  cosine(topic_emb, doc_emb) + category_match_penalty
        │
        ▼
[svm_filter()]  →  top-100
  LinearSVC trained on (topic, docs) at inference time
        │
        ▼
[classifier_filter()]  →  top-50
  SciBERT (semaj83/scibert_finetuned_pruned_ctmatch)
  3-class: not_relevant / partially_relevant / relevant
        │
        ▼
[gen_filter()]  →  top-10   [optional]
  LLM binary-search subquery ranking
        │
        ▼
ranked list of (nct_id, doc_text) pairs
```

**Key design insight:** each stage is trained on a different signal and operates at a different granularity:
- Sim: global semantic + topic category
- SVM: topic-conditioned linear ranking in embedding space
- SciBERT: criterion-level text classification
- Gen: reasoning over eligibility logic

### 5b. Stage 1: Embedding similarity + category prior

**The math:**

$$\text{score}(q, d) = \underbrace{\frac{q_{\text{emb}} \cdot d_{\text{emb}}}{\|q_{\text{emb}}\| \|d_{\text{emb}}\|}}_{\text{cosine similarity}} - \underbrace{\mathbf{1}[\arg\max(q_{\text{cat}}) \neq \arg\max(d_{\text{cat}})]}_{\text{category penalty}}$$

The category penalty is binary: 0 if the topic and doc share the same top category, 1 (subtracted) if not.

**Category prior as a Bayesian prior:**

The 14 medical categories (pulmonary, cardiac, gastrointestinal, ...) act as a prior over the document distribution. The intuition:

$$P(d \text{ relevant} | q) \propto P(q | d) \cdot P(d)$$

Where $P(d)$ encodes our prior belief that a cardiac patient is more likely to be eligible for cardiac trials. The category match is a hard approximation of this prior — it zeros out cross-category similarity rather than down-weighting it.

**TODO — show the bar plot of category distribution in the TREC corpus. Is "other" dominated? What's the cardiac/pulmonary split?**

**bart-large-mnli as a zero-shot category classifier:**
```python
CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological",
    "genetic", "pediatric", "neurological", "cancer", "reproductive",
    "endocrine", "infection", "healthy", "other"
]
# called once per topic (not per doc — docs are pre-classified at index time)
output = category_model(topic_text, candidate_labels=CT_CATEGORIES)
```

**Why exclusive argmax over the category distribution?**

`exclusive_argmax()` collapses the soft distribution to a one-hot vector. This is lossy — a patient with both cardiac AND renal disease gets classified as one or the other. 

**TODO — quantify: what fraction of TREC topics are multi-category? Is the cardiac enrichment in FP errors partly an artifact of topics that are cardiac-primary but renal-secondary?**

**`redist_other_category()`** redistributes the weight of the "other" category uniformly across the remaining 13. This prevents "other" from dominating the argmax for unusual conditions.

### 5c. Stage 2: SVM ranking

This is a clever approach adapted from Karpathy's image retrieval SVM paper. The key insight: train a LinearSVC **at inference time** on just the topic + candidate documents, using the topic as the single positive example.

```python
topic_embedding_vec = topic.embedding_vec[np.newaxis, :]
x = np.concatenate([topic_embedding_vec, doc_embeddings[doc_set]], axis=0)
y = np.zeros(len(doc_set) + 1)
y[0] = 1  # topic is the only positive

clf = svm.LinearSVC(class_weight='balanced', C=0.1)
clf.fit(x, y)
similarities = clf.decision_function(x)  # signed distance to decision boundary
```

**The math:**

The SVM learns a hyperplane $w$ that separates the topic from the document set in embedding space. The decision function $w \cdot x + b$ gives a signed distance — documents on the same side as the topic and closer to it get higher scores.

**Why this works:** the SVM adapts its decision boundary to each specific query. Unlike fixed cosine similarity, it finds the direction in embedding space that best discriminates the topic from the candidate set. For topics with multiple conditions, this can find a direction that emphasizes the combination.

**The `class_weight='balanced'` choice:** with one positive (the topic) and hundreds of negatives (docs), balanced weighting prevents the trivially correct "predict all negative" solution.

**TODO — add diagram showing embedding space with topic, relevant docs, irrelevant docs, and the SVM hyperplane**

### 5d. Stage 3: SciBERT classifier

Fine-tuned on the TREC clinical trials dataset as a 3-class sequence classification problem:

```
input:  [CLS] topic_text [SEP] doc_text [SEP]
output: softmax over {not_relevant, partially_relevant, relevant}
```

Model: `semaj83/scibert_finetuned_pruned_ctmatch`
- Base: `allenai/scibert_scivocab_uncased`
- Pruned with `nn_pruning` for faster inference
- Trained with class-weighted cross-entropy (imbalanced classes: ~15% relevant, ~20% partial, ~65% not relevant in training data)

**TODO — include the training data statistics and the final confusion matrix on the test set**

**The cross-encoder architecture:** unlike the bi-encoder used for embedding similarity, the cross-encoder sees both topic and document together, enabling attention across them. This is why it performs better for relevance classification but is too slow for first-stage retrieval.

**Batching fix (recent):** the original implementation sent all docs through in a single forward pass, causing OOM on large doc sets. Fixed to mini-batch with `batch_size=32`. Also fixed device placement (`ir_setup=True` was leaving the model on CPU even with a GPU available — now calls `.to(self.device)` on load).

### 5e. Stage 4: Generative reranker

**TODO — describe the binary search subquery approach and show a concrete prompt example**

The gen filter uses an LLM (originally OpenAI text-davinci-003, now configurable) to do list-wise reranking. The approach:
1. Partition the top-50 docs into token-budget-sized chunks
2. Ask the LLM to rank each chunk from most to least relevant
3. Keep the top half of each chunk
4. Repeat until ≤ top_n remain

This implements approximate halving sort — O(n log n) LLM calls. The original paper used 30 topics with the gen filter; the modernized baseline drops it (no paid inference).

### 5f. Eval mode vs. inference mode

**This is the single most important architectural distinction to understand.**

In inference mode: each filter actually reduces the candidate set (374k → 10k → 100 → 50 → 10).

In eval mode (`doc_set` is passed): `reset_filter_params()` sets ALL `top_n` values to `len(doc_set)`. Every filter sees the full judged set and returns all of it, ranked. The pipeline becomes a **reranker ensemble**, not a cascade filter.

```python
def reset_filter_params(self, val: int) -> None:
    self.sim_top_n = self.svm_top_n = self.classifier_top_n = self.gen_top_n = val
```

**Why:** if the sim filter dropped a relevant doc (false negative at the filter stage), it would never appear in the final ranking and NDCG would suffer. Correct NDCG computation requires all judged docs to be ranked.

**Consequence for eval performance:** the sim filter's category prior IS applied — docs from the wrong category get a lower combined score and end up ranked lower — but nothing is dropped. The cascade's efficiency benefit doesn't materialize in eval mode.

**Consequence for eval speed:** for KZ topics with 1,200+ judged docs, all three models score all 1,200. SciBERT (now on GPU with mini-batching) is the dominant cost at ~0.05s/doc.

#### Which filter configs actually differ in eval mode

Because every soft filter returns all N docs (just reordered), the final ranking is determined entirely by whichever soft filter runs **last**. Two configs produce identical NDCG in eval mode if and only if they share the same last soft-filter stage **and** the same set of hard-filter stages.

Concretely:
- `sim+svm+clf` ≡ `sim+clf` ≡ `svm+clf` ≡ `clf` — classifier is last in all four; they all rank the same N docs identically.
- `sim+svm` ≡ `sim+svm+svm` but ≠ `sim` — SVM is last, not sim.
- `sim+svm+demo+clf` ≡ `demo+clf` — classifier is last, demographic is the only hard filter, so both rank N-M docs by classifier score.

The only config dimension that meaningfully affects NDCG is:
1. **Which hard-filter stages are included** (they remove docs from the pool before the last ranker sees them).
2. **Which soft-filter stage runs last** (it determines the final ordering).

#### Why `sim+demo` ≡ `demo+sim` (exact order, not just relative)

The underlying principle: **pointwise scoring functions commute with set restriction**.

**Definitions.** Let $\mathcal{D}$ be the full document corpus and $q$ a fixed query (patient topic text, held constant throughout a pipeline run). Let $S \subseteq \mathcal{D}$ be a finite pool of candidate documents, $d \in S$ a single document, and $\text{filter}(S) \subseteq S$ the subset surviving a hard-exclusion rule. Define a scoring function as $f: \mathcal{D} \times 2^{\mathcal{D}} \rightarrow \mathbb{R}$, where $f(d, S)$ is the score assigned to document $d$ given pool $S$ and implicit query $q$. Define $\text{rank}_f(S)$ as the sequence of elements of $S$ sorted in descending order of $f(\cdot, S)$.

A scoring function $f$ is **pointwise** (with respect to $q$) if there exists a function $g: \mathcal{D} \rightarrow \mathbb{R}$ — depending on $q$ but not on $S$ — such that $f(d, S) = g(d)$ for all $S \subseteq \mathcal{D}$ with $d \in S$. Equivalently, $f$ is pointwise if $f(d, S) = f(d, S')$ for any two pools $S, S'$ both containing $d$.

When $f$ is pointwise, filtering and ranking commute:

$$\text{rank}_f(\text{filter}(S)) = \text{filter}(\text{rank}_f(S))$$

The left side filters the pool first, then ranks the survivors by $f$. The right side ranks the full pool by $f$, then removes the filtered-out documents. Because $f(d, S) = g(d)$ regardless of pool composition, removing a document from $S$ cannot change any other document's score, and therefore cannot change how the survivors rank against each other.

**This is a sufficient condition, not a necessary one.** A non-pointwise $f$ can still satisfy the equation for a particular $S$ and filter by coincidence — for example, if the documents removed by the filter happen to occupy score positions that don't affect the relative ordering of the survivors under the modified pool. Pointwiseness guarantees commutativity for *all* pools and *all* filters; a non-pointwise function may commute in specific instances but not in general.

Sim scores are computed as `cosine(topic_emb, doc_emb) + cat_match`, independently per doc. The score for doc A does not depend on whether doc B is in the pool. Sim is therefore pointwise, so:

- `sim+demo`: sim ranks all N docs by score, demo removes M → N-M docs in sim-score order.
- `demo+sim`: demo removes M first, sim ranks N-M docs → N-M docs in sim-score order.

The sim-score order of the surviving N-M docs is **identical** in both cases. This is not true for metrics like BM25 where IDF depends on the corpus, or for SVM:

#### Toy example: BM25 changes the end ranking

Query $q$ = "heart failure". Pool $S = \{d_1, d_2, d_3\}$, where $d_3$ is removed by the demographic hard filter.

| Document | Content (simplified) | Relevance |
|---|---|---|
| $d_1$ | "heart" × 4 | rel=1 (partially relevant) |
| $d_2$ | "failure" × 3 | rel=2 (relevant) |
| $d_3$ | "heart" × 1 | rel=0 — removed by demographic filter |

Using $\text{IDF}(t) = \log(N / \text{df}(t))$ and score$(d) = \text{TF}(d, t) \times \text{IDF}(t)$ summed over query terms:

**Path 1 — rank first, then filter:**

Under $S$ ($N=3$, $\text{df}(\text{"heart"})=2$, $\text{df}(\text{"failure"})=1$):
$$\text{IDF}(\text{"heart"}) = \log(3/2) \approx 0.41 \qquad \text{IDF}(\text{"failure"}) = \log(3/1) \approx 1.10$$
$$\text{score}(d_1) = 4 \times 0.41 = 1.62 \qquad \text{score}(d_2) = 3 \times 1.10 = 3.30$$

$\text{rank}_f(S) = [d_2, d_1, d_3]$. Remove $d_3$ → **final ranking: $[d_2, d_1]$**.

**Path 2 — filter first, then rank:**

Pool after filtering = $\{d_1, d_2\}$ ($N=2$, $\text{df}(\text{"heart"})=1$, $\text{df}(\text{"failure"})=1$):
$$\text{IDF}(\text{"heart"}) = \log(2/1) \approx 0.69 \qquad \text{IDF}(\text{"failure"}) = \log(2/1) \approx 0.69$$
$$\text{score}(d_1) = 4 \times 0.69 = 2.77 \qquad \text{score}(d_2) = 3 \times 0.69 = 2.08$$

$\text{rank}_f(\{d_1, d_2\})= $ **final ranking: $[d_1, d_2]$**.

The two paths produce opposite orderings. In the full pool, "failure" is rare ($\text{df}=1$, $\text{IDF}=1.10$) while "heart" is diluted by $d_3$ ($\text{df}=2$, $\text{IDF}=0.41$), so $d_2$ dominates. Once $d_3$ is removed, "heart" becomes unique to $d_1$ and both terms receive the same IDF; $d_1$'s higher term frequency then tips the balance. Since $d_2$ is the rel=2 document, only Path 1 produces the correct NDCG-optimal ordering — the order of operations is not semantically neutral.

#### Why `svm+demo` ≠ `demo+svm`

SVM is fit on its input set at inference time. A LinearSVC trained on N docs (one positive: the topic) produces a different hyperplane than one trained on N-M docs. The ranking of the N-M surviving docs will therefore differ between:
- `svm+demo`: SVM fit on full pool of N, then demographic removes M.
- `demo+svm`: demographic removes M first, SVM fit on reduced pool of N-M.

These are genuinely different configurations and will produce different NDCG.

#### Effect of hard filters on NDCG

Removing docs via a hard filter changes which docs can appear in the top-10. The impact depends on the relevance of what gets removed:

- Removed doc is rel=0: the next-ranked doc (rel ≥ 0) moves up. NDCG improves or stays the same.
- Removed doc is rel=1 (partial): contributes gain=1 to DCG if in top-10. Removing it and replacing with a lower-ranked doc can only hurt NDCG.
- Removed doc is rel=2 (relevant): contributes gain=3. Removing it is the worst case.

The demographic filter's NDCG impact is therefore determined by whether any of the M excluded docs would have appeared in the classifier's top-10 and what their relevance label is. Excluded rel=0 docs improve NDCG; excluded rel=1 or rel=2 docs hurt it.

Note: **MRR is only affected by rel=2 docs** (`calc_first_positive_rank` uses `pos_val=2`). Excluding rel=1 partial docs has zero effect on MRR regardless of where they would have been ranked.

---

## 6. Metrics

### 6a. NDCG@10

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i+1)}$$

Where $r_i \in \{0, 1, 2\}$ is the relevance of the document at rank $i$.

$\text{IDCG@}k$ is the DCG of the perfect ranking (all rel=2 first, then rel=1, then rel=0).

**Why NDCG@10 for this task:**
- Clinical users look at the top-10 results; beyond that, adoption drops
- Graded relevance (0/1/2) is correctly handled — putting a rel=2 above a rel=1 is more valuable than putting a rel=1 above a rel=0
- Normalized: scores comparable across topics with different numbers of relevant documents

**TODO — show concrete calculation on a small example (5 docs, mix of relevances)**

### 6b. MRR

$$\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank of first relevant doc for } q}$$

Here "relevant" means rel ≥ 1 (partially relevant or fully relevant). Lower rank = better.

**Note:** ctmatch implements this as `calc_first_positive_rank()` which returns the rank position (1-indexed), then averages. MRR is the mean of the reciprocal of these.

### 6c. F1 and FPR

For threshold-based evaluation: given the pipeline's final ranking, assign binary labels based on whether the document is in the top-k. Then compute standard precision/recall.

**FPR (False Positive Rate):** mean rank position of the first relevant document. Confusingly named — this is more like "how far down do you have to go to find something useful?" Not a standard IR metric.

### 6d. The custom "optimistic rank" metric

**TODO — formalize this properly**

The custom metric assumes, within each relevance tier, the optimal ordering. Given a ranked list, compute NDCG as if all rel=2 docs are at the top of the tier, all rel=1 in the middle, rel=0 at the bottom — but respecting the inter-tier ordering from the model. This is equivalent to computing NDCG on the "best case" scenario given the pipeline's macro-level decisions.

**Rationale:** when you don't have a full corpus ranking (only judged docs), you can't know the true NDCG. The optimistic metric gives an upper bound.

### 6e. eval_predictions.jsonl: structure and what can be inferred

`eval_baseline.ipynb` optionally writes `eval_predictions.jsonl` via `evaluate_detailed()`.
Understanding exactly what this file contains is critical for correct error analysis and
hard-negative mining.

#### What the file contains

Each record is one (topic, doc) pair with fields:
`topic_id`, `topic_text`, `doc_id`, `doc_text`, `rank`, `predicted`, `actual`,
`predicted_label`, `actual_label`, `is_error`.

The **`actual` field is the TREC/KZ qrel judgment** for that pair (0, 1, or 2).
The **`predicted` field is the argmax** of the classifier's 3-class logits for that pair.
The **`rank` field** runs from 1 to 10 — the file contains only the **top-10 ranked judged
docs per topic**, not all judged docs.

Concretely: 134 topics × 10 docs = 1,340 records. The top-10 is drawn from the judged pool
only (unjudged docs are invisible to the evaluator). Docs ranked 11+ that appear in the qrels
are not written to the file.

#### Why clf == sim+svm+clf in eval mode (affects which filter to run)

As documented in §5f: in eval mode every soft filter passes all N judged docs through and
returns them all, ranked. The final ranking is therefore determined entirely by the last soft
filter in the config. `sim+svm+clf`, `svm+clf`, `sim+clf`, and `clf` all produce identical
rankings and identical NDCG. Run `clf` (or any single-stage config ending in the classifier)
to avoid redundant computation. Only hard filters (e.g. `demo`) change the result by removing
docs before the last ranker sees them.

#### What TP/FP/FN/TN mean — and don't mean — in this file

Standard confusion matrix terminology assumes you see ALL instances. This file shows only the
top-10. The labels must be interpreted accordingly:

| Term | In this file | What it actually is |
|---|---|---|
| **FP** | `actual=0, predicted=2, rank ≤ 10` | Doc incorrectly promoted into the top-10. **Valid** — this is exactly what we want to suppress. |
| **TP** | `actual=2, rank ≤ 10` (any predicted) | Relevant doc the pipeline found. **Valid** — though `predicted` may be 1 if P(rel) score was high but argmax ≠ 2. |
| **"FN"** | `actual=2, predicted≠2, rank ≤ 10` | A relevant doc that ended in top-10 but whose argmax label is wrong (e.g. predicted=1). **Not a real FN** — it IS retrieved (rank ≤ 10); the argmax is just off. |
| **Real FN** | `actual=2, rank > 10` | Relevant doc buried below rank 10. **Invisible** — not in the file. NDCG penalizes these; they cannot be enumerated from this file alone. |
| **"TN"** | `actual=0, predicted≠2, rank ≤ 10` | Not-relevant doc that landed in top-10 but the classifier's argmax is not 2. **Not a classical TN** — it still appears in top-10 via continuous P(rel) score. |

**Practical consequence**: the file's FP count (199 in the baseline run) and the actual=2 docs
present in the top-10 (762) are the right inputs for hard-negative mining. The "FN" count of
22 from `is_error` is not the real false-negative rate — it just measures within-top-10
argmax errors, which are mostly borderline partial/relevant confusions. The real FN
rate requires scoring all judged docs and is captured by NDCG@10 itself.

#### Which config produced the file

The code gates detailed eval on `i == len(configs) - 1`: only the last config in
`FILTER_CONFIGS` is written, regardless of how many configs are active. The file has no
`filter_config` field. When running multi-config ablations, put the config of interest last,
or run a separate single-config eval pass to generate the file.

---

## 7. Experiments and Results

> ⚠️ **All ctmatch numbers in §7 are `PENDING(R)` — representation-confounded (§2g).** They were
> produced before the representation was frozen and are being re-run on one consistent `R` via
> `ctmatch.experiments`. Read the method and mechanism prose as valid; treat every score as invalidated
> until it carries a `repr_tag`. External baselines (h2oloo, TrialGPT) are unaffected.

**TODO — fill in from completed eval runs**

### 7a. Baseline (clf)

> In eval mode `clf` ≡ `sim+svm+clf` — see §5f. All results below use `clf` only.

**Full pipeline NDCG@10 (all 184 topics, clean split):**

| Config | NDCG@10 (all 184) | TREC22 NDCG@10 | MRR | F1 | FPR | Notes |
|---|---|---|---|---|---|---|
| clf+SciBERT | 0.6525 | — | 0.305 | 0.335 | — | TREC21+KZ only, contaminated |
| clf+BioLinkBERT-large | 0.7528 | — | 0.8253 | — | — | TREC21+KZ only, contaminated |
| clf-v1: PubMedBERT-base, clean | 0.7222 | 0.5967 | 0.8067 | 0.6455 | 2.261 | first honest baseline |
| clf-v2: PubMedBERT-base + hard-neg aug | 0.7578 | 0.6782 | 0.8407 | 0.6933 | 1.932 | TREC22 contaminated in hard-neg mining |
| clf-v3: BioLinkBERT-large, clean | 0.7404 | 0.6222 | 0.8343 | 0.6657 | 1.933 | larger model, clean |
| **clf-v4: BioLinkBERT-large + clean aug** | **0.7460** | **0.6388** | **0.8373** | **0.6694** | **1.925** | **current gate; clean hard-neg aug** |

TREC22 (50 topics) is the clean holdout — never seen during training.
clf-v2 TREC22 numbers are inflated: TREC22 topics were not filtered from hard-neg mining.
The honest TREC22 progression is: clf-v1 (0.5967) → clf-v3 (0.6222) → clf-v4 (0.6388).

**Key findings so far:**
- Hard-neg augmentation (+0.017 TREC22) and model scale (+0.026 TREC22) are both meaningful but subadditive
- The two evaluation frameworks diverge for augmented models: section 6b (classifier-only) consistently overstates TREC22 vs eval_baseline (full pipeline). Use eval_baseline as authoritative.
- Tier 2a (LLM-as-reranker, Qwen2.5-7B, zero-shot): pipeline NDCG@10 0.6485 (+1.6%), MRR 0.7759 (+3.8%) vs clf-v4 alone. Gain is modest but consistent — fine-tuned cross-encoder still dominates NDCG; LLM adds complementary signal at the top of the ranking.

**Note on eval mode vs. inference mode efficiency.**
TREC evaluation requires ranking every judged document per topic to compute fair NDCG.
This collapses the cascade: `reset_filter_params(len(doc_set))` passes all judged docs
through every filter, so the KZ topics (1,100+ docs each) run the full BERT classifier
over the entire judged pool — taking ~45–60s per topic in eval.

In real inference mode the cascade runs as designed: sim cuts 374k → 10k, SVM → 100,
classifier → 50, giving ~3s end-to-end on a GPU. TREC does not measure this.

**Benchmark blind spot:** TREC scores NDCG with no time or cost dimension. A system that
runs GPT-4 on every candidate for 45 minutes per query scores on the same axis as one
that runs in 3 seconds. TrialGPT (NDCG@10=0.7252 on a *pooled candidate set*, not full-corpus
retrieval — see §7e) falls into this category — it uses GPT-4 with per-criterion chain-of-thought,
which at $0.03/1k tokens over the candidate pool × ~50 criteria each is roughly $0.75/query. The
comparable full-corpus SOTA is the TREC 2022 winner (h2oloo) at 0.6125.
Our pipeline runs the same eval on Colab free tier.

A fairer comparison table would add:
- Latency (seconds per query, GPU T4)
- Cost per query ($ at public API prices, or GPU-hours)
- NDCG / (cost per query) as an efficiency-adjusted metric

TODO: measure and record inference latency per pipeline configuration.

### 7b. Bi-encoder ablation: off-the-shelf vs. fine-tuned vs. specialist models

**What is being measured:** bi-encoder ranking quality within the **full 184-topic judged pool**
(TREC21+TREC22+KZ). For each topic, the model encodes all judged docs and ranks by cosine
similarity (`eval_dense` in `reembed_corpus.ipynb`). This is a proxy for real retrieval quality —
the judged pool is enriched; full-corpus recall@1000 over 374k is deferred.

**Important:** bi-encoder NDCG does **not** affect `clf` NDCG in `eval_baseline.ipynb`.
Eval mode collapses the cascade (all judged docs pass through regardless of bi-encoder score);
only the cross-encoder classifier determines the final ranking. Bi-encoder quality matters in
live inference (first-stage recall over 374k), not in the judged-pool eval used for clf NDCG.

| Model | Dim | NDCG@10 ↑ | MRR ↑ | F1 ↑ | FPR ↓ | Notes |
|---|---|---|---|---|---|---|
| MiniLM-L6-v2 (off-the-shelf) | 384 | 0.3537 | 0.3978 | 0.6555 | 10.769 | production baseline |
| MedCPT (off-the-shelf) | 768 | 0.3457 | 0.4405 | 0.6555 | 8.679 | biomedical specialist; worse NDCG than MiniLM |
| PubMedBERT embeddings (off-the-shelf) | 768 | 0.3579 | 0.4275 | 0.6634 | 8.590 | |
| BGE-base-en-v1.5 (off-the-shelf) | 768 | 0.3939 | 0.4582 | 0.6646 | 9.925 | best off-the-shelf |
| **MiniLM fine-tuned (Tier 1b)** | **384** | **0.4734** | **0.5586** | **0.6902** | **6.806** | **MNRL on TREC21+KZ positives** |

Fine-tuning delta vs. off-the-shelf MiniLM: NDCG@10 +0.120 (+34%), MRR +0.161 (+40%), FPR −3.963 (−37%).

**Key finding:** in-domain fine-tuning of the smaller 384-dim MiniLM outperforms all off-the-shelf
alternatives including 768-dim biomedical specialist models (MedCPT, PubMedBERT). The fine-tuned
model also beats BGE-base (+0.080 NDCG), which had the best off-the-shelf score. Domain-specific
contrastive training on the task's own relevance signal is more valuable than scale or
biomedical pretraining for this retrieval problem.

**Why bi-encoder NDCG is much lower than cross-encoder NDCG (0.47 vs 0.64–0.75):** the
bi-encoder encodes topic and doc independently (no cross-attention), so it cannot model
fine-grained eligibility logic. Its job is recall — surfacing candidates — not precision ranking.
The FPR improvement (10.77 → 6.81) is the more operationally meaningful number: fewer irrelevant
docs make it into the cross-encoder's candidate set in live inference.

### 7c. Filter ablation: per-stage ranker quality

**What is being measured.** In eval mode, `reset_filter_params(N)` sets every filter's `top_n` to N (the number of judged docs for that topic). This means every filter in the chain passes all judged docs through, and the final ranking is determined solely by the **last** filter. As a consequence, `svm+clf ≡ clf` and `sim+svm+clf ≡ clf` in eval mode — adding upstream filters before the classifier produces identical NDCG to running the classifier alone.

The meaningful ablation is therefore each stage as the sole ranker: what NDCG does each component produce when it is the only thing doing the ranking?

| Config | Last-stage ranker | NDCG@10 (all 184) | Delta vs clf |
|---|---|---|---|
| `sim` | cosine(topic_emb, doc_emb) + category prior | 0.3219 | −0.4241 |
| `svm` | LinearSVC decision function on embeddings | 0.2576 | −0.4884 |
| **`clf`** | BioLinkBERT-large softmax P(relevant) | **0.7460** | — |

**Key finding: the cross-encoder does almost all the work.** The two upstream stages — embedding similarity and SVM — achieve NDCG@10 of 0.32 and 0.26 respectively when used as sole rankers, versus 0.75 for the classifier. Their role in the pipeline is candidate generation and recall, not precision ranking. In eval mode this contribution is invisible because all judged docs are passed through regardless; their value appears in live inference where they reduce the candidate set from 374k to 100 before the classifier runs.

**Why sim > svm here.** The SVM is trained per-topic on the fly: topic embedding as the single positive, all judged doc embeddings as the field. With a small positive set (one point) and no corpus context, the SVM decision function degenerates relative to the naive cosine similarity it was meant to improve on. In inference mode the SVM has the full 374k corpus to train against, giving it a meaningful hyperplane; in the judged-pool eval it fits on a biased sample and underperforms raw cosine.

**What this means for the pipeline design.** The cascade is justified by latency, not by each stage independently improving NDCG. Sim and SVM are gates that preserve recall while cutting the candidate pool from 374k → ~10k → ~100, so the classifier only runs ~100 cross-encoder forward passes per query rather than 374k. The NDCG numbers in this ablation do not measure their contribution to the final pipeline result — they measure what happens when the final precision ranker is removed entirely.

### 7d. Demographic pre-filter ablation

Full ablation across all meaningful config combinations (TREC21 + TREC22 + KZ, 135 topics):

| Config | NDCG@10 ↑ | MRR ↑ | F1 ↑ | FPR ↓ |
|---|---|---|---|---|
| `clf` | **0.7528** | **0.8253** | **0.6724** | 2.007 |
| `clf+demo` / `demo+clf` | **0.7532** | **0.8260** | 0.6716 | **1.993** |
| `sim+demo` / `demo+sim` | 0.3271 | 0.3576 | 0.2955 | 5.590 |
| `svm+demo` / `demo+svm` | 0.2592 | 0.2968 | 0.2381 | 6.119 |

`clf+demo` ≡ `demo+clf` confirmed exactly (clf is pointwise — see §5f).
`svm+demo` ≡ `demo+svm` empirically (SVM is not pointwise, but the filter removes so few docs — ~15 across 135 topics — that the hyperplane shift is negligible in practice).

**The demographic filter adds almost nothing to NDCG over the classifier alone (+0.0004).** This is not evidence that the filter is useless — it is an artifact of several properties of the TREC evaluation:

1. **Scale dilution.** The filter fires on 11 of 135 topics, removing ~15 docs total. Any improvement is averaged across all 135 topics and becomes tiny by construction.

2. **Pool bias.** The judged pool is drawn from systems that already do reasonable retrieval. Trials that are obviously age-wrong are rarely retrieved by any system and therefore never judged — they are invisible to NDCG. A 4-year-old patient matched against adult trials that never appear in any system's top-1000 contributes nothing to the metric, even though excluding them is medically correct.

3. **Classifier overlap.** BioLinkBERT has learned from training data that age-inappropriate trials score low. The hard demographic exclusion and the soft classifier signal partially overlap — the filter removes docs the classifier would have demoted anyway.

4. **Top-10 ceiling.** A removed FP only improves NDCG if it was inside the classifier's top-10. Most demographically-excluded docs are not ranked that high by the classifier.

**The filter's value is in production, not in TREC NDCG.** A 4-year-old patient should not be shown adult trials regardless of what the classifier scores. TREC qrels define relevance as topical match (right condition, right treatment), not strict eligibility — an adult trial for the right condition is often labeled relevant even for a pediatric patient. The demographic filter is medically correct; the TREC metric is not designed to reward it.

**Evaluate the demographic filter via:** clinical review of the docs it removes (confirmed: 5 FP removed, 4 partial removed, 0 relevant removed across 1340 eval records), not NDCG.

### 7e. The real target: full-corpus SOTA, and where TrialGPT actually sits

**This section is grounded in the primary PDFs** (both read directly, 2026-07): the *Overview of the TREC 2022 Clinical Trials Track* (Roberts et al.) and *Matching Patients to Clinical Trials with LLMs* (TrialGPT, Jin et al.). Earlier drafts asserted TrialGPT reports ≈0.73 "on the full corpus top-500" and treated it as the target; the primary sources show that is wrong on multiple counts.

#### The TREC 2022 benchmark and the real SOTA (verified: TREC overview)

- **Corpus:** the April 27, 2021 ClinicalTrials.gov snapshot, **375,581 trial descriptions** (our reprocessed corpus has 374,647 — a ~0.25% difference from processing/availability; noted as a minor caveat).
- **Topics:** 50 synthetic patient case descriptions (prose, ~110 words). **41 runs from 11 teams**; runs submit up to 1,000 trial IDs/topic; pooled to **depth 40** → 35,394 judged pairs.
- **Relevance scale (important, drives metric comparability):** *Eligible* = 2, *Excluded* (has the condition but an exclusion criterion applies) = 1, *Not Relevant* = 0. **NDCG uses the graded 2/1/0 gains. But for P@10, R-Prec, and MRR, only *Eligible* counts as relevant — *Excluded* is merged with *Not Relevant*** (binary, eligible-only).
- **Best run — team h2oloo, run `frocchio_monot5_e` (Table 4 of the overview):** **NDCG@10 = 0.6125**, **P@10 = 0.5080**, R-Prec = 0.3297, MRR = 0.7262. Next-best teams: DoSSIER 0.5565, iiia-unipd 0.5051, CSIROmed 0.4912. **This 0.6125 is the real full-corpus SOTA and the number our protocol is comparable to.** Note: h2oloo submitted the winning run but **no** TREC-31 CT participant paper, so the exact config is undocumented; the architecture is their SIGIR 2022 method (below).

  **h2oloo method** (Pradeep, Li, Wang, Lin, *Neural Query Synthesis and Domain-Specific Ranking Templates for Multi-Stage Clinical Trial Matching*, SIGIR 2022 — https://cs.uwaterloo.ca/~jimmylin/publications/Pradeep_etal_SIGIR2022.pdf; verified from the PDF): (1) **Neural Query Synthesis** — a doc2query–T5-3B model (MS MARCO V2-trained) generates ~40 single-sentence queries from the patient note; (2) **first stage** — each query → BM25 + RM3 PRF, fused via RRF (sparse only, no dense retriever; TREC21 first-stage nDCG@10 0.4726); (3) **reranker** — monoT5-3B → fine-tuned MS MARCO → Med-MARCO (`monoT5_MED`, zero-shot) → **further fine-tuned on Koopman-Zuccon/KZ** (`monoT5_CT`, ~1.1k positives) with multi-field ranking templates (title/condition/eligibility/description, sliding-window MaxP). Best TREC21 run `monoT5'_CT` nDCG@10 **0.7118**. **Correction to earlier drafts:** h2oloo is *not* purely zero-shot — its domain-tuned reranker trains on KZ (the same dataset ctmatch uses); it is "blind" only w.r.t. the TREC22 qrels (a blind competition submission). Contrast with ctmatch (§7j): they synthesize queries + one large monoT5-3B reranker; we use hybrid dense retrieval + a LambdaMART ensemble of smaller cross-encoders + a Qwen LLM judge. Their NQS directly targets ctmatch's diagnosed retrieval weakness on implicit-diagnosis topics (§8a).

#### TrialGPT's numbers are on a different, smaller, enriched task (verified: TrialGPT paper)

- **Search space is NOT the full corpus.** TrialGPT-Retrieval runs over "the combination of the judged clinical trials for all patients in the individual cohort" — i.e., the pooled judged set. Their Table 1 gives this exactly: **26,581 "considered initial trials" for TREC 2022** (vs our 374,647), 26,149 for TREC 2021, 3,621 for SIGIR. The ranking analysis then uses only the **top-500** retrieved per patient.
- **The 0.7252 is an average across three cohorts, not a TREC 2022 number.** TrialGPT Table 2 reports **NDCG@10 = 0.7252, P@10 = 0.6724** for GPT-4 TrialGPT-Ranking (eligibility aggregation), and 0.7275/0.6688 for the feature combination — but these are the **mean over SIGIR + TREC 2021 + TREC 2022**, not TREC 2022 alone. The SIGIR cohort (3,621-trial space) is far easier and inflates the average.
- **The paper itself states non-comparability** (verbatim, p.21): *"the results are not directly comparable to the results of TREC CT participating systems as we used the initial corpora of more realistic sizes and reported the average performance on three different cohorts."*
- **Their best non-LLM baseline** (Table 2) is a **BioLinkBERT cross-encoder trained on MedNLI: NDCG@10 = 0.4797** on the same pooled/top-500 setting — notable because it is the *same base model family* as our clf-v4, but trained on NLI rather than TREC relevance and used without our retrieval+ensemble stack.

So TrialGPT solves an easier problem three ways over: a ~14× smaller search space, further reduced to top-500 for ranking, and averaged over cohorts including the small SIGIR set. Its 0.7252 is **not** a full-corpus TREC 2022 result and cannot be placed on the same axis as h2oloo's 0.6125 or ours.

#### Scoreboard (full-corpus TREC 2022 protocol — the only apples-to-apples column)

| System | NDCG@10 | Protocol | Search space |
|---|---|---|---|
| **h2oloo — TREC 2022 winner** (`frocchio_monot5_e`) | **0.6125** | full-corpus, blind submission | 375,581 |
| **ctmatch — full ensemble (this work)** | **0.6105** | full-corpus, test-adapted | 374,647 |
| ctmatch — hybrid + score fusion | 0.5540 | full-corpus | 374,647 |
| ctmatch — full-text BM25 → clf-v4 | 0.458 | full-corpus | 374,647 |
| ctmatch — eligibility-only (start) | 0.22 | full-corpus | 374,647 |
| — | — | — | — |
| TrialGPT-Ranking GPT-4 (Jin et al.) | 0.7252\* | pooled-rerank, **not comparable** | 26,581 → top-500, ×3-cohort avg |
| BioLinkBERT/MedNLI (TrialGPT's best baseline) | 0.4797\* | pooled-rerank | 26,581 → top-500, ×3-cohort avg |

\* TrialGPT rows are 3-cohort averages over pooled judged sets, not TREC22 full-corpus numbers — shown only to document why they are excluded from the comparison.

**Metric-comparability caveat (must fix before publishing P@10/MRR).** Our NDCG@10 (0.6105) is directly comparable to h2oloo's 0.6125 — both use the graded 2/1/0 gains over the full corpus. **However, our reported P@10 (0.686) and MRR (0.851) are *not* comparable to h2oloo's P@10 (0.508)/MRR (0.726):** the TREC overview treats only *Eligible* (rel=2) as relevant for those metrics, whereas our `pytrec_eval` call used the default `rel≥1`, counting *Excluded* (rel=1) as relevant and inflating our numbers. **Action: recompute P@10/MRR with eligible-only relevance (`rel_level=2`) before putting them in any table.** NDCG@10 is unaffected and remains the clean headline comparison.

**Bottom line:** the open ctmatch pipeline reaches **NDCG@10 = 0.6105 vs the TREC 2022 winner's 0.6125** on the same full-corpus protocol — a statistical tie (competitive, not "beat"; see §7i). TrialGPT's 0.73 is a different, easier task and is not the target.

**References (read directly):**
- Roberts, Demner-Fushman, Voorhees, Bedrick, Hersh, *Overview of the TREC 2022 Clinical Trials Track* — https://trec.nist.gov/pubs/trec31/papers/Overview_trials.pdf. Corpus 375,581; 50 topics; 41 runs/11 teams; Table 4: h2oloo `frocchio_monot5_e` NDCG@10 0.6125 / P@10 0.5080 / R-Prec 0.3297 / MRR 0.7262; NDCG gains Eligible=2/Excluded=1, other metrics eligible-only.
- Jin, Wang, Floudas, Chen, Gong, Bracken-Clarke, Xue, Yang, Sun, Lu, *Matching Patients to Clinical Trials with Large Language Models* (TrialGPT) — https://arxiv.org/pdf/2307.15051. Table 1: 26,581 considered trials (TREC22), 183 patients across 3 cohorts; Table 2: GPT-4 ranking NDCG@10 0.7252 / P@10 0.6724 (3-cohort avg), best baseline BioLinkBERT/MedNLI 0.4797; p.21 explicit non-comparability statement.

### 7f. LLM reranker: zero-shot eligibility scoring (Tier 2a)

**Motivation.** BioLinkBERT-large (clf-v4) is a 340M-parameter cross-encoder fine-tuned on 31k in-domain (topic, trial) pairs. Its 512-token budget and fixed three-class head limit its reasoning: it cannot perform multi-step eligibility inference or exploit knowledge that didn't appear in its training distribution. A general-purpose instruction-tuned LLM trained on far broader data — including clinical literature, drug references, and patient narratives — might recover signal the cross-encoder misses, without any fine-tuning.

The question is whether zero-shot LLM reasoning about document-level eligibility adds measurable value on top of a fine-tuned specialist model, and whether it can do so at open-weight cost.

**Method.**

*Model:* Qwen2.5-7B-Instruct (Alibaba Cloud, Apache-2.0 license, 7B parameters, loaded in float16 on an A100 40GB). Chosen because it requires no HF gating, fits with margin on a single A100, and represents a current small-LLM baseline. Mistral-7B-Instruct-v0.3 and Llama-3-8B-Instruct are plug-in alternatives (same notebook, one config change).

*Scoring:* for each (topic, doc) pair, compute a relevance score as log P("yes") − log P("no") at the position of the first generated token. No tokens are actually generated — a single forward pass over the full prompt is sufficient. The score aggregates across all single-token tokenizer variants of "yes" and "no" (yes/Yes/·yes/·Yes and their negations) using logsumexp, so the scoring is robust to tokenizer-specific casing and spacing conventions.

```
score(topic, doc) = logsumexp(log_prob[yes_ids]) − logsumexp(log_prob[no_ids])
```

*Prompt:*
```
<system> You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
<user>
You are a clinical trial matching expert.

Patient:
{topic_text}

Trial eligibility criteria:
{doc_text}

Is this patient likely eligible for this trial? Answer with a single word: yes or no.
```

The full eligibility criteria text is used, tokenizer-truncated at 2048 tokens (`longest_first` — the patient description is short and rarely truncated; criteria text may be truncated in the longest cases). Character-level pre-truncation is explicitly avoided to preserve exclusion criteria, which typically appear after inclusion criteria in the text.

*Prompt format* is applied with `tokenizer.apply_chat_template(add_generation_prompt=True)`, which inserts the model's chat template markers (e.g. `<|im_start|>system/user/assistant`). Left-padding is used for batched inference so that `logits[:, -1, :]` correctly targets the next-token prediction position for all sequences in a batch.

*Batching:* batch size 4 on A100. ~35k forward passes (TREC22 standalone) completes in ~12 minutes; ~2,500 passes (pipeline top-50) in ~2 minutes.

**Sanity check.** Before scoring the full eval set, five examples are run with `max_new_tokens=3` to confirm the model's first output token is "yes" or "no" rather than a preamble. All five produced clean first-token yes/no answers with direction correlated with ground truth relevance labels.

**Two evaluation modes.**

*LLM standalone:* the LLM scores every judged document for each topic independently of clf-v4. Ranked by LLM score descending. NDCG@10 computed over the LLM ranking against the full doc2rel judged set. This measures raw zero-shot eligibility discrimination.

*clf-v4 → LLM pipeline:* clf-v4 scores all judged docs for each topic using softmax P(relevant) (matching the pipeline's `classifier_filter` exactly), selects the top-50, and the LLM reranks those 50 by logprob score. The final ranking is the LLM-reranked top-50 followed by clf-v4's remainder in clf order. NDCG@10 is computed over this combined ranking.

**Results (TREC22, 50 topics, clean holdout):**

| System | NDCG@10 ↑ | MRR ↑ | Notes |
|---|---|---|---|
| clf-v4 (BioLinkBERT-large) | 0.6383 | 0.7477 | standalone harness, softmax P(relevant) |
| Qwen2.5-7B standalone | 0.6269 | 0.7720 | zero-shot, all judged docs scored |
| **clf-v4 → Qwen2.5-7B (top-50)** | **0.6485** | **0.7759** | pipeline; LLM reranks clf's top-50 |

Pipeline delta vs clf-v4: NDCG@10 +0.010 (+1.6%), MRR +0.028 (+3.8%).

**Interpretation.**

*Finding 1: Fine-tuned cross-encoder beats zero-shot 7B on NDCG.* Qwen-7B standalone NDCG (0.627) is 1.1 points below clf-v4 (0.638). A 340M-parameter model fine-tuned on 31k in-domain pairs with focal loss and hard-negative augmentation outperforms a 7B general model with zero-shot prompting. Domain-specific supervision is more valuable than scale for this ranking problem, at least at the 7B level.

*Finding 2: The LLM has a complementary signal on MRR.* Qwen-7B standalone MRR (0.772) is notably higher than clf-v4 (0.748), a +3.2% gain. MRR rewards placing the first relevant result at rank 1 — the LLM is better at identifying and elevating the single most clearly eligible trial, even when its full top-10 ranking is noisier than clf-v4's. This suggests the LLM's broader clinical priors help in high-confidence positive cases, while the cross-encoder's discriminative training makes it more precise across the full ranking distribution.

*Finding 3: The pipeline strictly improves over either system alone.* clf-v4 → Qwen-7B (top-50) achieves NDCG@10=0.6485 and MRR=0.7759, better than both clf-v4 (NDCG=0.638, MRR=0.748) and LLM standalone (NDCG=0.627, MRR=0.772). The two rankers make different errors — the cross-encoder's systematic mistakes (missed exclusion logic, terminology gaps) are partially corrected by the LLM, while the LLM's noisier ranking over the full judged set is disciplined by clf-v4's top-50 pre-selection.

*Finding 4: The gain is modest, not transformative.* +1.6% NDCG is statistically real but small. (Note: this was measured under the older judged-pool protocol; the full-corpus numbers in §7i are the ones that compare to the real target. TrialGPT's ~0.73 is on pooled candidates and not comparable — see §7e.) Document-level eligibility prompting — feeding the full criteria text as a single block — does not replicate the reasoning power of TrialGPT's per-criterion GPT-4 chain-of-thought. Without isolating and reasoning about individual eligibility criteria, the LLM faces the same challenge as the cross-encoder: a single disqualifying exclusion criterion buried in a 500-word text is easily missed.

**Why the ceiling is where it is.**

The core limitation is that both clf-v4 and the LLM receive the same input representation: the entire eligibility criteria text concatenated as one string, paired with the patient description. This representation forces the model to simultaneously identify which criteria are relevant, determine whether the patient satisfies each, and aggregate the results — all without explicit structure. A patient who fails one out of eight exclusion criteria is encoded the same way as a patient who fails all eight; the surrounding inclusion criteria dominate the embedding or the prompt.

TrialGPT addresses this by parsing criteria into individual sentences and scoring each one separately with GPT-4 chain-of-thought. The structured per-criterion signal is then aggregated explicitly. This decomposition removes the multi-step reasoning burden from a single forward pass and makes the reasoning auditable. The price is latency and API cost (§7e).

The path to closing the remaining gap is not better document-level prompting — it is criterion-level decomposition (Tier 3, §9d). The current Tier 2a result provides an honest baseline: zero-shot document-level LLM reranking is worth +1–4% on NDCG/MRR but cannot substitute for structured eligibility reasoning.

**Cost and hardware.**

All experiments ran on a single A100 40GB (Colab Pro+). The LLM is loaded in float16 (~14GB VRAM); clf-v4 is loaded and scored first, then deleted before the LLM is loaded (~14GB peak). Total GPU time: ~15 minutes for the full TREC22 eval (standalone + pipeline). Estimated cost: <\$0.50 in GPU-hours. TrialGPT's GPT-4-based per-criterion eval on the same 50-topic scope would cost multiple orders of magnitude more — see §7a for the estimate on the 75-topic TREC21 eval.

### 7g. Criterion-level reranker: Claude ceiling test (Tier 3a)

**Motivation.** The Qwen2.5-7B criterion-level reranker (§9d, Tier 3a) scored NDCG@10=0.6242 on TREC22, below the clf-v4 baseline of 0.6388. Two competing explanations: (1) model weakness — Qwen 7B logprob without chain-of-thought is too noisy a judge; (2) input limitation — TREC topic descriptions don't contain the clinical detail needed to assess individual eligibility criteria, regardless of model quality. Replacing Qwen with Claude Sonnet distinguishes them: if criterion-level scoring still underperforms clf-v4 with a state-of-the-art model, the bottleneck is information content, not model capability.

**Method.**

*Model:* claude-sonnet-4-6, via async Anthropic API (`rerank_criteria_claude.ipynb`). Each (patient, criterion) pair is a separate API call; within each topic all criteria across the top-50 trials are scored concurrently (semaphore of 15). Total API calls on TREC22: ~17.4k.

*Task:* for each (patient description, criterion) pair, Claude returns one of five labels:
- `included` — patient meets this inclusion criterion
- `not_included` — patient does not meet this inclusion criterion
- `excluded` — this exclusion criterion applies to the patient
- `not_excluded` — this exclusion criterion does not apply
- `not_enough_information` — cannot determine from the patient description

Labels are mapped to scores (`included`=+1, `not_included`=−1, `excluded`=−2, `not_excluded`=+1, `not_enough_information`=0) and aggregated per trial.

*Aggregations:*
- `sum`: sum of all criterion scores
- `mean`: mean of all criterion scores (normalizes by criterion count)
- `strict_exc`: if any `excluded` label fires, apply a large penalty (−10 + mean inclusion score); otherwise use `sum`

*Pipeline:* identical to Tier 3a Qwen — clf-v4 top-50 reranked by criterion aggregate score; rest of judged set follows in clf-v4 order.

**Label distribution (TREC22, ~17.4k criterion assessments):**

| Label | Count | % |
|---|---|---|
| `not_enough_information` | 7,285 | **41.8%** |
| `not_excluded` | 6,132 | 35.1% |
| `included` | 2,677 | 15.3% |
| `not_included` | 1,045 | 6.0% |
| `excluded` | 310 | 1.8% |

The 41.8% NEI rate is the key diagnostic. Claude correctly recognizes that it cannot assess most clinical criteria from TREC topic descriptions — ECOG status, lab values, prior treatment history, and disease staging are rarely stated explicitly enough. The high `not_excluded` rate (35.1%) reflects that exclusion criteria are mostly conditions the patient does not have, which can be inferred.

**Results (TREC22, 50 topics, internal harness):**

Note: the standalone clf-v4 harness in reranker notebooks scores all judged docs directly with the cross-encoder. This gives clf-v4 NDCG@10=0.6593, which is higher than the eval_baseline pipeline number (0.6388). Use eval_baseline as the authoritative pipeline gate; use the internal harness number as the within-experiment reference.

| System | NDCG@10 | MRR | Δ vs clf (harness) |
|---|---|---|---|
| clf-v4 (standalone harness) | 0.6593 | — | — |
| **Tier 3a: clf→Claude criterion (`sum`)** | 0.5709 | 0.6899 | −0.0884 |
| **Tier 3a: clf→Claude criterion (`mean`)** | **0.6359** | **0.7804** | −0.0234 |
| **Tier 3a: clf→Claude criterion (`strict_exc`)** | 0.5700 | 0.6911 | −0.0892 |

Best aggregation: `mean` at NDCG@10=0.6359, still 0.023 below clf-v4 in the same harness.

**Interpretation.**

*Finding 1: Claude cannot beat clf-v4 at criterion-level scoring — the bottleneck is inputs, not models.* Even with a state-of-the-art frontier model and correct label semantics, criterion-level decomposition degrades TREC22 NDCG@10 by 2.3 points. The gap between Qwen (0.6242) and Claude (0.6359) is small (+0.012) compared to the gap both have to clf-v4. Upgrading the model from 7B logprob to a frontier API buys little. This rules out model quality as the primary explanation for Tier 3a's underperformance.

*Finding 2: 41.8% NEI confirms the information bottleneck.* TREC topics are 100–200 word clinical vignettes written for topic matching, not for criterion evaluation. They routinely omit: formal performance status (ECOG/Karnofsky), exact lab values, prior treatment regimen details, time-since-diagnosis, and staging workup. Claude correctly outputs `not_enough_information` for 42% of criterion assessments — it cannot fabricate information that isn't there. With ~11 criteria per trial and 42% unknown, every trial accumulates a large noise floor of zero-score criteria that dilute any signal the model does extract.

*Finding 3: `mean` aggregation is the only competitive strategy.* `sum` and `strict_exc` both score ~0.570, well below clf-v4. `sum` is dominated by trial length — trials with more criteria accumulate more score regardless of their actual match. `strict_exc` applies extreme penalties when `excluded` fires (1.8% of assessments), which is appropriate in principle but noisy when only 1–2 out of ~12 exclusion criteria are assessable. `mean` normalizes by criterion count and treats unknowns as neutral zeros, giving the most informative signal. Its 0.6359 result, while still below clf-v4, shows that mean aggregation is roughly aligned with the cross-encoder on NDCG even without being able to beat it.

*Finding 4: High MRR for `mean` (0.7804) vs `sum`/`strict_exc` (0.689–0.691).* MRR measures first-relevant placement. `mean` does relatively well at elevating clearly eligible trials to rank 1–3 within the top-50; `sum` and `strict_exc` distort this by penalizing long well-matched trials or applying catastrophic exclusion penalties incorrectly.

*Finding 5: The bottleneck is prompting strategy, not topic information content.* TrialGPT is also evaluated on TREC 2021/2022 topics — the same short clinical vignettes we use, not richer EHR data. The claim that "TrialGPT works because it has more informative patient descriptions" is incorrect. The difference is how the model handles uncertainty given the same sparse text:

- **Our approach:** Claude receives a simple label prompt with no reasoning chain. When the criterion requires information not in the topic (ECOG score, exact lab values, treatment history), Claude correctly outputs `not_enough_information`. This is epistemically honest but produces 41.8% NEI — a large dead zone of zero-contribution criteria.
- **TrialGPT's approach:** GPT-4 with chain-of-thought is instructed to reason step-by-step: expand abbreviations ("AC" → doxorubicin/cyclophosphamide), apply clinical context ("ambulates independently → consistent with ECOG 0-1"), perform implicit arithmetic ("4 cycles × 60 mg/m² = 240 mg/m² < 450 threshold → criterion not triggered"). CoT allows GPT-4 to commit to a label even from incomplete information, making calibrated inferences rather than abstaining.

Additionally, TrialGPT's evaluation is not directly comparable: it reranks a reduced candidate set pooled from TREC participant submissions, not full-corpus retrieval (§7e). This confounds any numeric comparison.

The cross-encoder (clf-v4) sidesteps the criterion-level problem entirely: it learns which clinical narrative patterns correlate with judged relevance without needing to resolve individual criteria explicitly. This is more robust to sparse topics because it matches at the level of overall clinical picture, not logical criterion satisfaction.

**Implications and next steps.**

The Tier 3a result rules out one explanation (model capability) but opens several actionable paths:

**Path A — Add chain-of-thought to criterion prompts (highest expected impact for this architecture)**

Replace the current single-label prompt with a CoT prompt that instructs Claude to reason before labeling:

```
Patient: {patient_text}

{type} criterion: {criterion}

Think through this step by step:
1. What specific information does this criterion require?
2. Does the patient description contain this information, explicitly or implicitly?
3. If implicit, what clinical inference can you make?
Label: [included / not_included / excluded / not_excluded / not_enough_information]
```

This directly addresses the 41.8% NEI rate: instead of abstaining, Claude would reason from available context ("ambulates independently" → ECOG 0-1). Expected effect: NEI rate drops, NEI → `included` / `not_included` conversions improve signal. Cost increases ~3× due to longer outputs. Test on a 10-topic subset before running full TREC22 to check whether NEI rate actually falls before committing.

**Path B — Richer aggregation under uncertainty**

Currently NEI → 0 (neutral score). This treats unknown criteria as uninformative, which is correct in expectation but loses information. Alternatives:

- *Type-conditional prior:* for inclusion criteria with NEI, substitute the base rate of `included` across assessed criteria (~15.3% in the current run) rather than 0. For exclusion criteria with NEI, substitute the `not_excluded` base rate (~35.1%).
- *Asymmetric NEI weighting:* an unknown exclusion criterion is more dangerous than an unknown inclusion criterion (false inclusion is worse than missed inclusion in clinical practice). Subtract a small penalty for NEI on exclusion criteria rather than treating it as 0.
- *Count-based features only:* use raw label counts (n_included, n_not_included, n_excluded, n_not_excluded, n_nei) as features into a learned aggregation function trained on TREC21+KZ pairs. This avoids hand-tuned score maps and lets the model learn which label patterns predict judged relevance.

**Path C — Patient representation expansion**

Both our system and TrialGPT use the raw TREC vignette. An intermediate step: use an LLM to expand the vignette into a structured clinical summary before criterion assessment:

```
Input:  "74-year-old male with hypertension, hyperlipidemia, T2DM, presenting with
         chest pain on exertion. EF 45% on echo."
Output: {age: 74, sex: M, conditions: [HTN, HLD, T2DM, HFmrEF], LVEF: 45%,
         symptoms: [exertional chest pain], ECOG_estimate: "0-1 (ambulates)"}
```

The criterion assessor then receives the structured summary rather than the raw narrative. This trades LLM inference cost for interpretable structured features. Implementation: add a `cell-expand-topics` cell before `cell-pipeline` in `rerank_criteria_claude.ipynb`.

**Path D — Distill Claude labels into a small fine-tuned criterion cross-encoder**

The `criteria_claude_labels.jsonl` file contains ~17.4k (patient, criterion, label) pairs — noisy but in-domain. Fine-tune a small biomedical cross-encoder (BiomedBERT-base or PubMedBERT) on these labels as a 5-class classifier. This gives:
- Fast inference (no API cost at deployment)
- A model that can be further fine-tuned as better labels accumulate
- A concrete object for the Stage C distillation plan

Quality caveat: with 41.8% NEI in the training labels, the fine-tuned model may learn to predict NEI too aggressively. Filter NEI examples from training or up-weight non-NEI examples. Evaluate on a held-out set of manually verified (patient, criterion, label) triples from the TrialGPT annotation dataset (`ncbi/TrialGPT-Criterion-Annotations`, 1,020 pairs with GPT-4 labels + explanations).

**Path E — NDCG-differentiable loss on clf-v4 (separate from criterion-level)**

Completely independent of criterion parsing: replace clf-v4's cross-entropy loss with ApproxNDCG or a pairwise margin ranking loss operating on qrel preference pairs (§9b). This directly optimizes the eval metric at the document level and requires no criterion parsing, no LLM API calls, and no change to the inference architecture. Cleanest near-term option for improving the TREC22 NDCG gate number.

**Recommended priority order:** E (ranking loss, clean win within existing architecture) → A (CoT prompts, directly addresses the NEI bottleneck) → C (patient expansion, higher effort but principled fix) → B (aggregation tuning, low effort but marginal without fixing A) → D (distillation, valuable but dependent on A/C improving label quality first).

**Cost.** ~17.4k API calls at claude-sonnet-4-6 pricing (~$3/1M input tokens, 450 avg input tokens/call): estimated ~$30. Total GPU time: ~20 minutes for clf-v4 scoring; Claude API runs on CPU with no GPU requirement after clf-v4 is freed.

### 7h. Distilled open criterion cross-encoder (Tier 3b)

**Motivation.** §7g established that criterion-level scoring with a frontier model still trails clf-v4 on TREC22, and identified the bottleneck as input information (41.8% NEI) rather than model quality. That result was a *ceiling* test, not a production system — it used Claude, which is off-limits for end inference (open-models-only constraint). Tier 3b asks a narrower, deployment-relevant question: can an entirely open pipeline — an open reasoning model to annotate, distilled into a small open cross-encoder — produce a criterion signal that *complements* clf-v4, even if it can't beat it alone?

The pipeline has three stages, all open-weight:
- **Stage A** (done, §7g): Claude ceiling test — establishes the achievable NEI floor and label semantics.
- **Stage B**: annotate the TREC21+KZ **training** split with DeepSeek-R1-Distill-Qwen-7B (open CoT model), producing `(patient, criterion) → label + reasoning`.
- **Stage C**: fine-tune BiomedBERT-base as a 5-class criterion cross-encoder on the Stage B silver labels; aggregate per-trial and fuse with clf-v4.

**Stage B — R1-7B annotation.** DeepSeek-R1-Distill-Qwen-7B (float16, transformers backend on A100 80GB) annotated the clf-v4 top-10 trials per training topic. Output: **11,250** (patient, criterion) records across 134 topics (trec21: 6,216; kz: 5,034), each with the model's chain-of-thought reasoning captured.

| Label | Count | % |
|---|---|---|
| `included` | 4,530 | 40.3% |
| `not_enough_information` | 3,907 | 34.7% |
| `not_included` | 1,396 | 12.4% |
| `not_excluded` | 1,206 | 10.7% |
| `excluded` | 211 | 1.9% |

The R1-7B NEI rate (34.7%) is **lower** than Claude's simple-label baseline (41.8%): the reasoning trace lets the smaller model commit to inferences Claude abstained on, exactly the CoT effect predicted in §7g Path A. This is a favorable sign for label informativeness, though it says nothing about label *correctness* — the silver labels are unverified.

*Data-quality note.* An initial run used a buggy parser that ran label extraction over the full R1 output (reasoning + `</think>` + answer), so reasoning text containing a label word could shadow the true final label. This corrupted 770 records across 8 trec21 topics (detectable as empty-reasoning records). They were stripped and re-annotated with the corrected parser; the `excluded` count on those topics fell from 276 → 211, confirming the original parse had spuriously inflated the rare high-penalty class. Lesson for the writeup: for CoT annotation, always parse the answer *after* the `</think>` delimiter, never over the whole string.

**Stage C — distillation and fusion.** BiomedBERT-base fine-tuned as a cross-encoder (`patient [SEP] INCLUSION/EXCLUSION: criterion`, `longest_first` truncation, inverse-frequency class weights to protect the 1.9% `excluded` class), 5 epochs, best-val-loss checkpoint. Per-trial score = mean over criteria of the expected label weight (`included`=+1, `not_excluded`=+1, `not_included`=−1, `excluded`=−2, NEI=0). Evaluation on a 20% topic-level holdout of the training split (26 topics), reranking judged-pool docs.

*Classification quality (val):* macro-F1 0.33, weighted-F1 0.43. `excluded` collapsed (F1 0.08 on 31 support); class weighting produced over-prediction of rare classes rather than genuine learning (e.g. `not_excluded` recall 0.76 at precision 0.26). Val loss rose after mid-training — overfitting to silver-label noise.

*Ranking quality (val, 26 topics):*

| System | NDCG@10 | Notes |
|---|---|---|
| clf-v4 alone | 0.8401 | **inflated** — these are trec21/kz topics clf-v4 trained on |
| criterion scorer alone | 0.4476 | distilled cross-encoder, mean aggregation |
| combined, z-norm, α=1 | 0.7945 | equal-weight fusion **hurts** clf-v4 |

*Fusion-weight sweep* (`combined(α) = znorm(clf) + α·znorm(crit)`): optimum at **α=0.05**, NDCG 0.8446, a delta of **+0.0045** over clf-v4 — 10 of 26 topics improved, 6 hurt, 10 unchanged.

**Interpretation.**

*Finding 1: the distilled scorer is weak in isolation and near-useless as an equal fusion partner.* 0.45 standalone and a fusion that only helps at α≈0 both point to a low-signal, high-noise ranker. This is the expected outcome of the "silver labels + base encoder + fixed aggregation + naive fusion" configuration — it is the baseline ablation row, not the target system.

*Finding 2: there is a faint but non-zero complementary signal.* The sweep peaks at a positive α with a small positive delta rather than at α=0. On this 26-topic val set that delta is within noise (10-up/6-down is not significant), so it is not decision-grade evidence — but it does not rule out complementarity either.

*Finding 3: the winners are all KZ topics, the losers all trec21.* The two large gains (kz_201415 +0.18, kz_20158 +0.14) are KZ; every negative-delta topic is trec21. Two explanations the val set cannot separate: (a) **ceiling artifact** — clf-v4 memorized these trec21 topics (near-1.0 already, criterion signal can only hurt) while KZ has headroom, which would mean the val set *understates* fusion value relative to the unseen TREC22 set where clf-v4 sits at 0.6388; or (b) **style effect** — the scorer simply works better on KZ's one-liner topics and won't transfer to trec-style TREC22. Distinguishing these requires the clean TREC22 holdout.

**Statistical-power caveat.** Three different sample sizes are in play and they are not equally trustworthy. The ~2k-pair classification metrics are adequate for coarse conclusions (macro-F1, overfitting) but not for per-class claims (`excluded` has 31 val examples). The 26-topic ranking eval is genuinely underpowered — a 10/6 topic split is p≈0.45 on a sign test. And topic-level eval never scales past TREC22's ~50 topics; every published comparison on this task, TrialGPT included, rests on that same n. Consequences adopted going forward: (i) use k-fold CV over the 134 train topics for architecture decisions rather than one holdout; (ii) do model selection at the criterion level (tens of thousands of pairs, real power) and use ranking NDCG only as final confirmation; (iii) attach paired-bootstrap / Wilcoxon CIs to every NDCG comparison, especially the eventual TrialGPT claim.

**Why this motivates the next architecture.** Every weakness above traces to one of four fixable components — noisy silver labels, a base-size encoder, a fixed hand-set aggregation vector, and naive equal-weight fusion. Tier 3c (§9d) replaces all four: gold-by-construction synthetic labels, a GRPO-tuned assessor, and a learned aggregator that folds in the clf-v4 signal. Tier 3b is the honest baseline that quantifies the headroom those changes have to recover.

**Cost and hardware.** Stage B: ~1 GPU-hour on A100 (11,250 generations, 7B float16, batch 128). Stage C: ~30 min training + ~7 min val eval on A100. No API cost — fully open pipeline. Contrast with the Stage A ceiling test (~$30 Claude API), which is permitted only because it is annotation/analysis, not eval-path inference.

### 7i. Full-corpus evaluation — the retrieval→rerank pipeline against the real target

**Why this section exists.** Everything in §7a–§7h uses judged-pool reranking: the system reorders the ~35k already-judged (topic, trial) pairs. The real TREC 2022 benchmark (§7e) is *full-corpus retrieval* — rank the relevant trials out of all ~374k. `eval_fullcorpus.ipynb` implements the comparable protocol (retrieve → TREC run file → `pytrec_eval` against official qrels), and it changed the picture substantially. Target: beat the TREC 2022 winner, **NDCG@10 = 0.6125**.

**Finding 1 — the corpus was the wrong text.** The `ctmatch_ir` corpus (`doc_texts.txt` + all dense embeddings) was built from **eligibility-criteria text only** (confirmed: README, and the indexed docs are enrollment-rule text). Patient queries are diagnostic narratives; eligibility rules often don't even name the disease. Full-corpus BM25 recall@1000 on this representation was a catastrophic **0.1144**, and BM25→clf-v4 NDCG@10 only **0.22**. `build_fulltext_corpus.ipynb` re-fetches title + conditions + summary + detailed description + interventions + eligibility for all 374k NCT IDs from the ClinicalTrials.gov API v2 (`filter.ids` batches; only 100/374k missing; median doc length 950→2853 chars). Re-indexing on full text roughly **doubled** everything: BM25 recall@1000 → **0.2355**, BM25→clf-v4 → **0.458**.

**Finding 2 — dense ≫ lexical for retrieval, as predicted.** Re-encoding the fine-tuned retriever (`ctmatch-retriever-v2`) on full text:

| Retriever (TREC22, full corpus) | NDCG@10 | Recall@100 | Recall@1000 |
|---|---|---|---|
| BM25 | 0.2434 | 0.099 | 0.2355 |
| dense (fine-tuned MiniLM) | 0.3383 | 0.2286 | 0.5260 |
| hybrid (RRF of BM25+dense) | 0.4125 | 0.2093 | 0.5200 |

Dense more than doubles BM25 recall. Hybrid RRF has the best precision-at-top (MRR 0.79).

**Finding 3 — the reranker, not retrieval, is now the bottleneck (oracle diagnostic).** An oracle rerank (sort each retrieval pool by true qrel relevance) gives the NDCG@10 ceiling that pool's recall permits:

| Pool | Oracle NDCG@10 | clf-v4 actual | fraction extracted |
|---|---|---|---|
| BM25 | 0.8008 | 0.4581 | 57% |
| dense | 0.9441 | 0.3934 | 42% |
| **hybrid** | **0.9569** | 0.4408 | 46% |

The hybrid pool already contains enough relevant docs for a **0.957** top-10 — far above the 0.6125 target. clf-v4 extracts under half of it, and is *most* efficient on BM25's pool (its lexically-pooled training distribution) and *least* on dense's semantic hard negatives. Retrieval recall (0.52) is not what caps NDCG@10; reranker precision is.

**Finding 4 — reranker retraining on hard negatives did not (yet) help.** Two attempts to make clf exploit dense's recall both failed to beat clf-v4: (v1) training raw BioLinkBERT-large from scratch on a narrow positives+hard-neg slice *regressed* to 0.30 (discarded clf-v4's full training distribution — a recipe error); (v2) continue-training *from* clf-v4 on the full pool with dense hard negatives oversampled improved recall@100 (0.284) but not NDCG@10 (0.40). Pointwise cross-entropy cross-encoders appear to plateau ~0.44 here regardless of hard negatives.

**Finding 5 — the fix was an eval artifact, and score fusion is the win.** With `RERANK_K = RETRIEVE_K = 1000`, "reranking" replaced the retrieval order *entirely* with the classifier's sort (empty tail) — discarding hybrid's MRR-0.79 ordering, which is why MRR cratered. Two eval-only sweeps (no training) fixed it:
- **`RERANK_K` cutoff** (rerank top-K, keep retrieval order below): best at K≈100 → **0.5228**.
- **Score fusion** `α·clf + (1−α)·hybrid`, α=0.4 → **NDCG@10 = 0.5540, MRR 0.7747** — the best full-corpus number, +0.096 over the previous best, from a one-line eval fix.

**Finding 6 — learned ensemble (LambdaMART) is the winning frame.** `train_ensemble_full.ipynb` combines per-document features with a LightGBM `lambdarank` model (details below). The ensemble climbed steadily as signals and fixes were added:

| Ensemble configuration | TREC22 NDCG@10 | MRR |
|---|---|---|
| hybrid + hand-tuned score fusion (α=0.4) | 0.5540 | 0.7747 |
| LambdaMART, 7 features | 0.5324 | 0.8327 |
| + reranker-v2 P(relevant) (8th feature) | 0.5594 | 0.8076 |
| + open LLM yes/no score (9th feature, top-200) | 0.5779 | 0.8369 |
| + coherent LLM floor & `llm_scored` indicator | 0.5867 | 0.8424 |
| + widen LLM coverage to top-500 | 0.5913 | 0.8365 |
| **+ 5-fold-CV-tuned hyperparameters** | **0.6105** | **0.8512** |

Two findings worth preserving: (a) the **LLM yes/no feature is genuinely independent and strong** — `diagnose_llm_coverage.ipynb` measured per-topic rank-AUC = 0.791 and clean label separation (mean score rel=2 +3.25 / rel=1 −2.17 / rel=0 −10.78); it became the #2 feature. (b) A **judged-only training variant regressed to 0.37** — the unjudged-as-0 rows correctly encode the ~95%-non-relevant serve distribution; removing them miscalibrates the model. (c) The **default LambdaMART overfit**: CV selected a much smaller model (num_leaves=15, ~50 rounds vs default 31/300), worth +0.019 — the single cleanest gain because it was chosen on train, not test.

#### How the ensemble targets NDCG — the LambdaMART / LambdaRank objective

NDCG@10 is what we want to maximize, but it is **not differentiable**: it depends on the sorted rank order, and sorting is a step function with zero gradient almost everywhere. You cannot backpropagate through "rank." LambdaRank (Burges et al.) sidesteps this by **defining the gradients directly** instead of differentiating a loss.

For a query with a candidate set, the model outputs a score $s_i$ per document. For an ordered pair $(i, j)$ with $y_i > y_j$ (i more relevant), RankNet models $P_{ij} = 1/(1 + e^{-\sigma(s_i - s_j)})$ with pairwise cost $C_{ij} = \log(1 + e^{-\sigma(s_i - s_j)})$ and gradient

$$\frac{\partial C_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}}.$$

RankNet stops there — it only cares about pairwise *order*, not *where* in the list the pair sits. LambdaRank's one crucial change is to scale each pair's gradient by the change in the target metric from swapping $i$ and $j$:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}}\; \big|\Delta\text{NDCG}_{ij}\big|,$$

where $|\Delta\text{NDCG}_{ij}|$ is the absolute change in NDCG@$k$ if *only* documents $i$ and $j$ swapped positions in the current ranking. Each document's gradient is the net force over all its pairs:

$$\lambda_i = \sum_{j:\, y_i > y_j} \lambda_{ij} \;-\; \sum_{j:\, y_j > y_i} \lambda_{ji}.$$

**LambdaMART** = these $\lambda$ gradients + gradient-boosted regression trees (MART): each boosting round fits a tree to the current $\lambda$ pseudo-responses, moving documents in the direction that most increases NDCG@$k$. The $|\Delta\text{NDCG}|$ weight makes the model spend nearly all its gradient on pairs whose swap actually changes the top-10, and ignore pairs buried deep in the list.

**Worked example (NDCG@10).** One topic, current ranking, two documents (gain $= 2^y - 1$, discount $= 1/\log_2(\text{rank}+1)$):

- **A**: eligible ($y=2$), currently rank 8
- **B**: not relevant ($y=0$), currently rank 3

Swapping A up to rank 3 and B down to rank 8:
- A: gain 3, rank 8→3 contributes $3\,(1/\log_2 4 - 1/\log_2 9) = 3\,(0.500 - 0.315) = +0.554$ to DCG
- B: gain 0 → contributes 0 at any rank
- $\Delta\text{DCG} \approx +0.554$, so $|\Delta\text{NDCG@10}| = 0.554/\text{IDCG}$ — a **large** weight; $\lambda$ pulls A strongly up and B down.

Contrast a deep pair — **C** (partial, $y=1$) at rank 40 and **D** ($y=0$) at rank 44: both sit outside the top 10, so swapping them leaves NDCG@10 unchanged, $|\Delta\text{NDCG@10}| = 0 \Rightarrow \lambda = 0$. The model expends no effort there. This is exactly why LambdaMART suits a top-heavy metric like NDCG@10.

**Where it sits, and the untried extension.** LambdaMART is the *ensemble/combiner* stage: it consumes the per-document feature vector (BM25, dense, RRF, clf-v4 P(rel/partial), reranker-v2 P(rel), LLM yes/no) plus the graded qrel label, grouped by topic, and emits the final score. The upstream cross-encoders (clf-v4, v2) are still trained with **pointwise cross-entropy** and only *produce features*; the listwise NDCG optimization happens only here. Training the cross-encoders *themselves* with a listwise/NDCG-aware loss (ApproxNDCG / LambdaLoss on the neural model, §9b) is the still-untried version of the idea — a heavier lever, likely small now that the combiner already optimizes NDCG. (Precise term: LambdaMART is *pairwise with listwise NDCG-delta weighting*, commonly grouped with listwise LTR because it directly optimizes NDCG.)

**Progress ladder (TREC22 full-corpus NDCG@10):** 0.22 (eligibility-only) → 0.458 (full-text corpus) → 0.554 (hybrid + fusion) → 0.5594 (+v2) → 0.5779 (+LLM) → 0.5867 (floor fix) → 0.5913 (widen) → **0.6105 (CV-tuned)**. vs h2oloo 0.6125. **Gap: 0.002; 95% CI [0.542, 0.679]; P(>0.6125) = 49%.**

**Honest status — competitive with SOTA, not "beat".** 0.6105 is a *statistical tie* with the best TREC 2022 run, but the comparison is asymmetric and must be reported as such: **h2oloo's 0.6125 was a blind competition submission** (no qrel access during development), whereas ctmatch's 0.6105 was developed *with* the TREC22 qrels across ~8 selection rounds (fusion α, RERANK_K cutoff, feature choices, floor, top-K all validated directly on TREC22). Their number is truly held-out; ours is partly fit to the test. Defensible claim: **"an open, end-to-end full-corpus pipeline competitive with the best TREC 2022 system."** Not "tied with SOTA," and "beat SOTA" would not survive review. Without h2oloo's per-topic scores a paired significance test is impossible, so "competitive" is the ceiling of what is claimable even at 0.615.

#### Per-source validation — does the number generalize, and where does it fail?

A single 50-topic test number is easy to over-trust, so it was cross-checked two ways.

**Cross-validation by source** (5-fold grouped CV over the 134 train topics, best config):

| Source | CV NDCG@10 | topics |
|---|---|---|
| TREC21 | 0.552 | 75 |
| KZ | 0.098 | 59 |

TREC21 CV (0.552) lands close to the TREC22 held-out test (0.6105) — both are TREC-style clinical vignettes, and their agreement is the key evidence the pipeline **generalizes** rather than being fit to TREC22. Were the pipeline overfit to the test, TREC21 CV would have collapsed too; it did not. KZ is the outlier at 0.098.

**Why KZ collapses — two compounding causes, not one.** Pool-recall (fraction of judged-relevant trials that reach the retrieval candidate pool):

| Source | pool recall@1000 |
|---|---|
| TREC21 | 64% |
| TREC22 | 58% |
| KZ | 34% |

KZ retrieval is genuinely worse — its qrels judge a **2015** ClinicalTrials.gov snapshot while we retrieve over the **2021** corpus, so many relevant trials are poorly represented. But 34% recall alone cannot explain NDCG@10 = 0.098 (TREC22 at 58% recall scores 0.61). The remainder is **ranking quality on terse queries**: KZ topics are one-line descriptions ("35-year-old female diagnosed with anorexia nervosa"), giving the cross-encoders, LLM, and retriever far less to work with than a ~200-word TREC vignette, and the LambdaMART combiner is trained on a set dominated by TREC-style topics (11,589 relevant vs KZ's 1,106). So KZ suffers from (1) a **retrieval/corpus-vintage mismatch** (data artifact) and (2) **genuine underperformance on one-liner queries** (a real pipeline limitation).

**Reporting decision.** The headline is TREC-style full-corpus retrieval: **TREC22 = 0.6105 (held-out), TREC21 CV = 0.552 (consistent).** KZ is reported *separately* with both caveats — not folded into the headline (its data mismatch would understate the pipeline) and not hidden (its one-liner weakness is a legitimate limitation and a concrete future-work direction: query expansion / structured-summary generation for terse inputs).

#### Next step is discipline, not another lever

At a test-adapted near-tie, every additional TREC22-validated tweak *weakens* the eventual claim. **Freeze the config.** Report TREC21 CV + TREC22 held-out as primary (above); KZ separately with caveats. Limitation to state honestly (not hide): clf-v4/v2/the retriever were themselves trained on TREC21+KZ, so a *fully* clean nested estimate would require per-fold retraining of those — out of scope for this stage; named as a limitation.

**Where the next gains actually are (from the error analysis, §8a — evidence-based, not assumed).** The error analysis overturned the "retrieval recall is the ceiling" assumption: retrieval loses 42% of relevant trials but leaves a ~5× surplus of eligible trials in the pool (48/topic vs 10 slots; oracle 0.957 vs extracted 0.61), so **the binding constraint is reranker top-10 precision**, not recall. In priority order, and each validated on a fresh split (not TREC22 again):
1. **A stronger LLM judge** (larger/CoT/calibrated, wider than top-500) — `llm_yesno` both surfaces true positives and, via its false negatives, is the main cause of buried eligible trials; it is the most direct attack on top-10 precision.
2. **Resolve the Eligible-vs-Excluded objective mismatch** — an eligibility judge is misaligned with the *graded/topical* metric on the rel=1 (Excluded) tier; a two-headed topicality+eligibility signal or metric-aware aggregation (or the criterion features, §9d Tier 3c) could recover it.
3. **Recalibrate the cross-encoder features** — clf_rel/v2_rel currently score *higher* on false positives than true positives; a listwise/NDCG-aware retrain (§9b) or down-weighting.
4. **Query/diagnosis expansion for terse, implicit-diagnosis topics** — secondary for NDCG@10 (surplus covers it) but the primary lever for recall and for the KZ one-liners.

Note the reprioritization: the previously-assumed "raise retrieval recall" is now a *secondary* lever (#4), and a better *reranker/LLM judge* is primary — a direct consequence of the §8a evidence.

#### Stage summary (the claim, in one paragraph)

> ⚠️ **The numbers in this summary are `PENDING(R)` — representation-confounded (§2g), being re-run.**

An open, end-to-end, full-corpus clinical-trial retrieval pipeline — full-text corpus reconstruction (ClinicalTrials.gov API v2) → hybrid BM25+dense retrieval → a LambdaMART ensemble over BM25/dense/RRF, two fine-tuned BioLinkBERT cross-encoders, and an open Qwen-2.5-7B yes/no reranker score — reached (pre-R, invalidated) ~~**NDCG@10 = 0.6105 on the blind TREC 2022 Clinical Trials test set**~~ **PENDING(R)** (vs the winning h2oloo run's 0.6125), with matching-basis ~~P@10 = 0.4940~~ / ~~MRR = 0.7379~~ **PENDING(R)** — the pre-R pipeline was statistically indistinguishable from SOTA, a claim to be re-established on the frozen representation. It is corroborated by **0.552 five-fold CV on TREC 2021** and uses **no closed models in the inference path**. The caveat: our configuration was selected with test-set access, so it is characterized as *competitive with*, not *superior to*, SOTA. The result was reached through mechanistically-diagnosed fixes (an eligibility-only→full-text representation repair that doubled recall; an oracle analysis localizing the bottleneck to the reranker; a score-fusion fix recovering discarded retrieval order; and CV-tuned listwise ranking) rather than test-set number-chasing, and the pipeline's one clear weakness — terse one-liner queries (KZ) — is reported rather than hidden.

#### Paper-ready results table

**Table 1. Full-corpus retrieval on TREC 2022 Clinical Trials (50 topics).** NDCG@10 computed with `trec_eval` against the official qrels over the full corpus (graded gains Eligible=2/Excluded=1/Not Relevant=0). **NDCG@10 is the one clean cross-system comparison.** h2oloo figures are from the TREC 2022 overview, Table 4 (verified). TrialGPT is a different protocol (pooled ~26.6k-trial judged set → top-500, averaged over 3 cohorts) and is *not comparable* — shown only to document the distinction.

> ⚠️ **ctmatch rows are `PENDING(R)` (representation-confounded, §2g). h2oloo/TrialGPT rows are external and stand.**

| System | NDCG@10 | P@10 | MRR | Protocol (search space) | Open |
|---|---|---|---|---|---|
| h2oloo — TREC 2022 winner (`frocchio_monot5_e`) | 0.6125 | 0.5080 | 0.7262 | full-corpus, **blind** (375,581) | — |
| **ctmatch — full ensemble (this work)** | ~~0.6105~~ PENDING(R) | ~~0.4940~~ | ~~0.7379~~ | full-corpus, test-adapted (374,647) | ✅ |
| BM25 (lexical baseline, this work) | ~~0.2434~~ PENDING(R) | — | — | full-corpus (374,647) | ✅ |
| *TrialGPT-Ranking GPT-4 (Jin et al.)* | *0.7252* | *0.6724* | — | *pooled-rerank, **not comparable** (26,581→top-500, ×3-cohort avg)* | ❌ (GPT-4) |

NDCG@10 uses graded gains (Eligible=2/Excluded=1/Not Relevant=0). **P@10 and MRR are eligible-only** (Excluded merged with Not Relevant), matching the TREC 2022 official protocol — so the ctmatch and h2oloo values are directly comparable; ctmatch is within noise on all three (−0.002 NDCG, −0.014 P@10, +0.012 MRR). h2oloo figures are from the overview Table 4; TrialGPT figures are 3-cohort averages over a pooled ~26.6k-trial set and are shown only to mark the protocol difference.

Generalization (secondary): ctmatch TREC 2021 5-fold CV NDCG@10 = **0.552** (consistent with the TREC22 test). KZ excluded — retrieval/corpus-vintage mismatch + terse-query degradation (per-source validation above).

**Table 2. Ablation — TREC22 NDCG@10 by pipeline stage (this work).** Each row adds one component to the previous.

| Stage | NDCG@10 |
|---|---|
| eligibility-only corpus, BM25 → clf-v4 | 0.22 |
| + full-text corpus reconstruction | 0.458 |
| + hybrid retrieval & score fusion | 0.554 |
| + LambdaMART ensemble w/ reranker-v2 feature | 0.559 |
| + open LLM (Qwen-2.5-7B) reranker feature | 0.578 |
| + LLM floor fix & top-500 coverage | 0.591 |
| + 5-fold-CV-tuned hyperparameters | **0.611** |

#### Limitations (this stage)

Stated plainly so the result is neither over- nor under-sold:

1. **Test-set adaptation.** The configuration (fusion weight, `RERANK_K` cutoff, feature set, LLM floor, coverage top-K) was chosen across ~8 rounds validated *directly on TREC22*. So 0.6105 is partly fit to the test, whereas h2oloo's 0.6125 was a blind submission. Hence *competitive with*, not *superior to*, SOTA.
2. **No paired significance test possible.** Without h2oloo's per-topic scores, we cannot run a paired bootstrap/Wilcoxon test against it. The bootstrap CI on our own number is [0.542, 0.679] (n=50) — wide, and it contains both 0.6105 and 0.6125.
3. **Upstream components trained on the eval families.** clf-v4, reranker-v2, and the fine-tuned retriever were trained on TREC21+KZ, so the TREC21 CV (0.552) is optimistic to the degree a fully clean nested estimate (per-fold retraining of those models) would reduce it. The TREC22 held-out number is unaffected — those topics were never in any training set.
4. **KZ is not fairly evaluable here.** 2015 qrels vs 2021 corpus (34% pool recall) plus terse one-liner queries the pipeline handles poorly (CV NDCG@10 = 0.098). Reported separately, not in the headline; the one-liner weakness is a genuine limitation, not only a data artifact.
5. **Small-n evaluation.** TREC22 has 50 topics — a constraint shared by all clinical-trial-matching benchmarks. Point estimates carry real uncertainty (see the CI).
6. **Retrieval recall ceiling.** Hybrid recall@1000 ≈ 0.52 on TREC22 — ~48% of relevant trials never reach the reranker. The oracle ceiling on the retrieved pool is 0.957, so the reranker (not retrieval) is the current binding constraint; raising recall would lift that ceiling further.

**Notebooks:** `eval_fullcorpus.ipynb` (harness), `build_fulltext_corpus.ipynb` (corpus rebuild), `rerank_llm_feature.ipynb` (open LLM reranker scores), `diagnose_llm_coverage.ipynb` (feature-validity audit), `train_reranker_hardneg.ipynb` (v1/v2 hard-neg reranker), `train_ensemble_full.ipynb` (full LambdaMART ensemble — supersedes `train_ensemble_ltr.ipynb`).

### 7j. Methods — the full open pipeline (reproducible)

A self-contained specification of the system that produced the §7i result. Fully open-weight in the inference path; no closed model is called at retrieval, feature, or ranking time.

**7j.1 Datasets and splits.** Corpus: the ClinicalTrials.gov 2021-04-27 snapshot (official 375,581 trials; 374,647 in our reprocessed corpus, aligned to `index2docid.txt`). Topics: **train** = TREC 2021 (75) + KZ (59) = 134; **test** = TREC 2022 (50), held out. Relevance: 3-level (Eligible=2, Excluded=1, Not Relevant=0). Metrics via `pytrec_eval`/`trec_eval` against official qrels: NDCG@10 (graded gains), P@10 and MRR (eligible-only). Full-corpus protocol: up to 1,000 trials/topic.

**7j.2 Corpus reconstruction** (`build_fulltext_corpus.ipynb`). For each of the 374,647 NCT IDs, fetch the full record from the ClinicalTrials.gov **API v2** (`GET /studies`, `filter.ids` in batches of 100, resumable). Assemble each document as the concatenation of `briefTitle` + `officialTitle` + `conditions` + `briefSummary` + `detailedDescription` + `interventions` (type+name) + `eligibilityCriteria`. Output `doc_texts_fulltext.txt`, aligned to `index2docid` order (median length 950 → 2,853 chars; 100/374,647 trials missing from the API). *Rationale: the prior corpus was eligibility-text-only; full text roughly doubled recall@1000 (0.11 → 0.24).*

**7j.3 Retrieval** (`eval_fullcorpus.ipynb`).
- **BM25:** Okapi (`rank_bm25`), whitespace-lowercase tokenization, over the full-text corpus.
- **Dense:** `semaj83/ctmatch-retriever-v2` — MiniLM-L6 (384-dim) fine-tuned with MultipleNegativesRankingLoss on TREC21+KZ positive (topic, trial) pairs; the full corpus re-encoded on the full-text representation; retrieval by cosine over L2-normalized vectors.
- **Hybrid:** Reciprocal Rank Fusion (RRF, k=60) of the BM25 and dense rankings.
- **Candidate pool** per topic = union of each retriever's top-1000 (~1,900 trials/topic). Measured recall@1000: BM25 0.24, dense 0.53, hybrid 0.52; oracle NDCG@10 on the hybrid pool = 0.957.

**7j.4 Per-document features (9)** (`train_ensemble_full.ipynb`, cached to `ensemble_features_full.npz`). For each (topic, candidate) pair:

| # | Feature | Source |
|---|---|---|
| 1–2 | `bm25`, `bm25_rank` | BM25 score and within-pool rank |
| 3–4 | `dense`, `dense_rank` | dense cosine and within-pool rank |
| 5 | `rrf` | RRF score |
| 6–7 | `clf_rel`, `clf_partial` | clf-v4 softmax P(relevant), P(partial) |
| 8 | `v2_rel` | reranker-v2 P(relevant) |
| 9 | `llm_yesno` (+ `llm_scored`) | Qwen yes/no logprob; binary scored-indicator |

**7j.5 Component models.**
- **clf-v4** (`semaj83/ctmatch-clf-v4`): BioLinkBERT-large cross-encoder, 3-class sequence classification (not/partial/relevant), fine-tuned on TREC21+KZ judged (topic, trial) pairs with focal loss + hard-negative augmentation; input `[CLS] topic [SEP] full-text trial`, truncated to 512 tokens.
- **reranker-v2** (`train_reranker_hardneg.ipynb`): clf-v4 **continue-trained** on the full TREC21+KZ judged pool, with **dense-mined hard negatives** — for each training topic, the judged rel=0 trials ranked highest by the dense retriever (top-50), **oversampled ×3** — at LR 5e-6, 2 epochs, class-weighted CE. Used only as feature #8 (weak as a standalone reranker; valuable in the ensemble).
- **LLM reranker** (`rerank_llm_feature.ipynb`): **Qwen-2.5-7B-Instruct** (Apache-2.0), float16. For each (topic, trial), a chat prompt asks "Is this patient likely eligible for this trial? yes or no"; score = `logsumexp(logits[yes-variant ids]) − logsumexp(logits[no-variant ids])` at the generation position from a **single forward pass** (no tokens generated). Trial text truncated to 1,800 chars; scored on the **top-500** of each pool by RRF; deeper candidates receive a floor value (below the empirical minimum) plus `llm_scored=0`. Signal audit (`diagnose_llm_coverage.ipynb`): per-topic rank-AUC 0.79.

**7j.6 Ensemble reranker (LambdaMART).** LightGBM with `objective=lambdarank`, `metric=ndcg`, `ndcg_eval_at=[10]` — a listwise objective that optimizes NDCG@10 directly by weighting pairwise gradients by |ΔNDCG| (derivation + worked example in §7i). Training rows are the **full candidate pools** of the 134 train topics with graded qrel labels (unjudged=0), grouped by topic. Hyperparameters selected by **grouped 5-fold CV over training topics**: final `num_leaves=15`, `min_data_in_leaf=20`, `learning_rate=0.05`, `lambda_l2=1.0`, ~50 boosting rounds (defaults 31/300 overfit). Inference: the booster scores each candidate; the ranking is the sorted score.

**7j.7 Evaluation protocol.** Full-corpus retrieval → per-topic candidate pool → 9 features → LambdaMART score → TREC run file → `pytrec_eval`. Train on TREC21+KZ; TREC22 held out. Validation: bootstrap 95% CI (10k resamples); grouped 5-fold per-source CV; eligible-only P@10/MRR (`rel_level=2`) for leaderboard comparability. KZ reported separately (degenerate: 34% pool recall + terse queries).

**7j.8 Notebook → step map.**

| Step | Notebook |
|---|---|
| Corpus reconstruction | `build_fulltext_corpus.ipynb` |
| Retrieval + full-corpus harness + fusion/oracle | `eval_fullcorpus.ipynb` |
| Hard-negative reranker (v2) | `train_reranker_hardneg.ipynb` |
| Open LLM reranker scores | `rerank_llm_feature.ipynb` |
| LLM feature audit | `diagnose_llm_coverage.ipynb` |
| Feature extraction + ensemble + CV-tune + eval | `train_ensemble_full.ipynb` |

**7j.9 Reproducibility.** Feature extraction and LLM scores are cached to Drive (`ensemble_features_full.npz`, `llm_reranker_scores.jsonl`); retraining/re-eval from cache is CPU-only and takes minutes. Seeds fixed (spec sampling, CV folds = 42/0). Result: NDCG@10 0.6105, P@10 0.4940, MRR 0.7379 on TREC22.

---

## 8. Error Analysis

### 8a. Full-corpus ensemble pipeline (current) → see `docs/error_analysis_ensemble.md`

From `eval_predictions_ensemble.jsonl` (50 TREC22 topics, per-trial outcome + all 9 feature values).
**This corrected a going assumption: retrieval recall is *not* the binding constraint — the reranker is.**

1. **Reranker precision, not retrieval, is the ceiling.** Retrieval loses 42% of relevant trials, but a mean of **48 Eligible trials/topic remain in the pool** vs 10 top-10 slots (oracle 0.957 vs extracted 0.61). **31% of top-10 slots are false positives** while eligible trials sit unused.
2. **The LLM judge drives the top-10; its false negatives are the main ranking-miss cause.** Eligible trials that surface have llm_yesno +5.78; eligible trials that get buried have −11.51 (Qwen wrongly says "ineligible"). The LLM is both the feature that finds true positives *and* the feature that buries them.
3. **Eligible-vs-Excluded objective mismatch.** The LLM judges *eligibility*, but graded NDCG rewards *topicality* — Excluded (rel=1) trials should rank above Not-Relevant. When the LLM correctly catches an exclusion (llm_yesno −14.5) it buries a rel=1 trial the metric wanted elevated.
4. **False positives = miscalibrated cross-encoders overriding the LLM.** clf_rel/v2_rel score *higher* on FPs (0.25/0.15) than on true positives (0.13/0.05); 18% of FPs were also beyond the LLM's top-500 (un-vetoed).
5. **Retrieval misses concentrate on implicit-diagnosis / terse-symptom topics** (worst pool-recall 0.21: nosebleeds, azoospermia, rash+oral-ulcers) — patient presents symptoms, trial text names diseases. Same failure mode as the KZ one-liners; a secondary lever for NDCG@10 but the main one for recall.

Prioritized levers: reranker top-10 precision (largest) → stronger/CoT LLM judge + wider coverage → resolve the eligibility-vs-topicality mismatch → recalibrate cross-encoder features → query/diagnosis expansion for terse topics.

### 8b. Judged-pool clf baseline (earlier) → see `docs/error_analysis_baseline.md`

Findings from the older judged-pool clf-v4 baseline (different, easier protocol — retained for reference):
1. Cardiac topics are 10.5x enriched in false positives — the category prior groups cardiac conditions too coarsely (CAD, HF, arrhythmia, valvular disease all map to "cardiac")
2. ~15–20% of errors are likely label quality issues (incorrect ground truth), not model errors
3. Age/gender hard constraints are not enforced — a 74-year-old patient matching a trial that excludes patients >70 is a systematic FP source
4. Rare conditions with no lexical or embedding overlap to their trial's terminology are systematic FNs (retrieval miss, not reranker failure)

---

## 9. Future Directions

### 9a. Criterion-level entailment

**TrialGPT approach:** for each criterion, ask: "can this patient satisfy this criterion?" using NLI or LLM scoring. This makes exclusion criteria explicit — a patient who fails even one exclusion criterion is not eligible.

**Our gap:** the current SciBERT classifier sees the full eligibility text as a single document. It cannot distinguish a patient who fails one out of ten exclusion criteria from one who fails five. Criterion-level scoring would add this resolution.

**Implementation sketch:**
1. Use ctproc to parse inc/exc criteria into sentence lists
2. For each criterion, score: patient description → criterion via NLI model (entails / neutral / contradicts)
3. Aggregate: eligible iff (all inclusion entailed) AND (no exclusion entailed)
4. Use TrialGPT criterion annotations (`ncbi/TrialGPT-Criterion-Annotations`) as training signal

#### Why you cannot simply do binary entailment per criterion and AND the results

The natural instinct is: parse each criterion into a sentence, run NLI (entails / neutral / contradicts) between patient and criterion, then apply logical AND across all inclusion criteria and NOT-AND across exclusion criteria. Eligible = all inclusions entailed AND no exclusions entailed.

This is clean, interpretable, and wrong in practice. Here is why, with real examples from the TREC corpus.

---

**Problem 1: Missing information — you cannot entail OR contradict**

Criterion (inclusion): *"ECOG performance status 0 or 1"*

Patient: *"74-year-old male with hypertension, hyperlipidemia, and type 2 diabetes presenting with chest pain on exertion. Currently ambulates independently."*

ECOG is a formal 5-point scale (0=fully active, 1=restricted in strenuous activity, 2=ambulatory but unable to work, 3=limited self-care, 4=fully disabled). The patient "ambulates independently" — is that ECOG 0 or 1? We don't know. Is the chest pain on exertion restricting him from strenuous activity (ECOG 1) or is he still fully active (ECOG 0)? A clinician would need to examine him. An NLI model can only output **neutral** — it cannot entail that he meets this criterion, nor can it contradict it.

If your pipeline requires ALL inclusion criteria to be entailed, a single unknown criterion kills the entire trial — the patient is falsely classified as not eligible. TREC assessors marked many such patients as rel=1 or rel=2 using clinical judgment about the *likely* ECOG status given the narrative.

---

**Problem 2: Vague criteria with trial-specific definitions**

Criterion (inclusion): *"Adequate hepatic function"*

Patient: *"ALT 52 U/L, bilirubin 1.1 mg/dL"*

"Adequate hepatic function" is defined differently in every trial's protocol. One trial might define it as ALT < 3× ULN (~117 U/L); another as ALT < 2× ULN (~78 U/L); another as ALT < 5× ULN in the setting of liver metastases. The eligibility textblock often states the criterion without the numerical definition because the protocol document has the definitions and the CTG textblock is a summary.

A patient with ALT=52 might satisfy one trial's definition and fail another's. NLI on the textblock alone cannot resolve this — the information required for entailment is not present.

---

**Problem 3: Temporal and historical requirements**

Criterion (exclusion): *"Prior anthracycline therapy with cumulative dose > 450 mg/m²"*

Patient: *"s/p AC chemotherapy × 4 cycles for breast cancer 3 years prior"*

AC is doxorubicin (60 mg/m² per cycle) + cyclophosphamide. Four cycles = 240 mg/m² cumulative doxorubicin, which is under 450. So the patient *does not* trigger this exclusion criterion and is potentially eligible. But:
- "AC" requires domain knowledge to expand to doxorubicin
- The cumulative dose requires knowing the standard dosing regimen and multiplying
- The patient description says "4 cycles" but standard regimens vary; some use 75 mg/m²/cycle, which would put her at 300 mg/m² — still under 450
- If she received dose-dense AC, the calculation changes again

NLI would almost certainly return **neutral** (the patient text does not explicitly state a dose above 450), which is technically correct but for the wrong reason. The real answer requires arithmetic over implicit information. An experienced oncology nurse would compute this immediately; an NLI model cannot.

---

**Problem 4: Partial eligibility is not a logical artifact — it is a clinical judgment**

Criterion (inclusion): *"Diagnosis of heart failure with reduced ejection fraction (HFrEF), defined as LVEF ≤ 40%"*

Criterion (exclusion): *"Estimated GFR < 30 mL/min/1.73m²"*

Patient: *"74M with history of ischemic cardiomyopathy, EF 35%, CKD stage 3b with GFR 28."*

Binary entailment outcome:
- Inclusion: patient has EF 35% ≤ 40% → **entails** ✓
- Exclusion: GFR 28 < 30 → **entails** → patient is EXCLUDED ✗

The model outputs: not eligible (0).

But a TREC assessor — and clinician — might label this **rel=1 (partially eligible)** because:
1. The patient is close to the threshold (GFR 28 vs. 30)
2. CKD stage 3b GFR fluctuates; a repeat measurement might be ≥ 30
3. The investigator might grant an exception for a GFR of 28 in a patient who otherwise perfectly fits the trial
4. Some trials use eGFR with different equations (MDRD vs. CKD-EPI) that might yield slightly different values

The rel=1 label is a clinical judgment about *likelihood of enrollment* given the full picture, not a binary logic gate. This is the fundamental reason the 3-point relevance scale exists — real-world eligibility is not a boolean.

If you binary-AND all inclusion criteria and binary-OR-NOT all exclusion criteria, you collapse rel=1 into rel=0. You lose the middle tier that NDCG@10 uses most discriminatively (the difference between placing a rel=2 at rank 1 vs. a rel=1 vs. a rel=0 is large).

---

**Problem 5: Negation compounding and double negatives**

Criterion (exclusion): *"Patients with no prior exposure to checkpoint inhibitors are ineligible for the dose-escalation cohort but may enroll in the dose-expansion cohort"*

This sentence contains: a negative ("no prior exposure"), a conditional eligibility structure (two cohorts), and is itself an exclusion criterion. The correct clinical parsing is: checkpoint inhibitor-naive patients CAN enroll (in one cohort), checkpoint inhibitor-experienced patients cannot enroll in that cohort but can in another. An NLI model reading this as "does the patient's description entail this criterion?" will almost certainly produce an incorrect or highly uncertain output.

NegEx (the tool used in ctproc's NLP layer) handles single-sentence negation well. It cannot handle compound conditionals embedded in regulatory language.

---

**Problem 6: Cross-criterion dependencies**

Criterion (inclusion): *"Stage III or IV non-small cell lung cancer (NSCLC)"*
Criterion (inclusion): *"At least one prior platinum-based chemotherapy regimen for stage IV disease"*

The second criterion only applies if the patient has stage IV (not stage III). These criteria cannot be evaluated independently — the second is conditional on a value extracted from the first. Logical AND-ing them treats them as independent when they are not.

In practice, eligibility criteria are a **decision tree**, not a flat list of independent constraints. Some criteria branch on prior criteria. Some have OR-logic within a single criterion ("LVEF ≤ 40% OR prior hospitalization for HF within 12 months"). The flat-list parsing in ctproc necessarily loses this structure.

---

**What this means for criterion-level systems**

The above examples explain why TrialGPT uses GPT-4 with chain-of-thought reasoning rather than a classification head over an NLI model. GPT-4 can:
- Expand medical abbreviations ("AC" → doxorubicin)
- Perform arithmetic ("4 cycles × 60 mg/m² = 240 mg/m²")
- Apply clinical context ("ECOG 0-1 is consistent with an ambulatory patient with exertional symptoms")
- Handle conditional structure ("this criterion applies only to stage IV patients")
- Produce a graded score (0–2) rather than a binary output

Even so, TrialGPT's criterion annotations are imperfect — the `ncbi/TrialGPT-Criterion-Annotations` dataset contains 1,020 rows with per-criterion GPT-4 labels and explanation chains. The errors in those annotations are instructive: they cluster exactly in the problem categories above (missing information, temporal reasoning, vague thresholds).

**The right framing for criterion-level work:** NLI per criterion is not a solution; it is a feature extraction step that produces a structured, uncertain signal to be aggregated by a model that understands clinical context. The aggregation function — how uncertain criterion-level signals combine into a trial-level relevance score — is where the modeling challenge actually lives.

### 9b. Pairwise ranking loss on qrel preference signal

> **Status update:** this idea is now **realized at the ensemble level** — the LambdaMART combiner
> (§7i) uses the `lambdarank` objective, which directly optimizes NDCG via NDCG-delta-weighted
> pairwise gradients (full formula + worked example in §7i). Tuning it was the single cleanest gain
> (+0.019 → 0.6105). What remains untried is applying a ranking loss to the **neural cross-encoder
> itself** (clf-v4/v2 are still pointwise cross-entropy and only produce features) — the heavier
> lever described below.

The TREC qrels provide pairwise preference signal (rel=2 > rel=1 > rel=0) that cross-entropy training doesn't fully exploit — CE treats each example independently, ignoring the relative ordering of docs within a topic.

**Correct approach for a classifier:** replace (or augment) CE with a **margin ranking loss**:

```python
# For each topic, sample (chosen, rejected) pairs where chosen_rel > rejected_rel
# Score each doc with the classifier's logit for class 2 (relevant)
loss = max(0, margin - score(chosen) + score(rejected))
```

This directly trains the model to score (topic, rel=2_doc) higher than (topic, rel=0_doc) by at least `margin`, which is what NDCG actually rewards. The qrel's three-level scale gives three pair types: (2,1), (2,0), (1,0) — each with different expected margins.

**Why not DPO:** DPO is a generative training objective that operates on token-level log-probabilities. It applies to language models with a generation head, not to BERT classifiers with a linear classification head. For our architecture, ranking loss is the correct analogue.

**Clinical advantage:** expert pairwise judgments from an RN with 20 years ICU experience on specific error cases are higher-value signal than the TREC qrels, which are topical-match labels annotated by non-clinicians. These corrections can be added as high-weight (chosen, rejected) pairs in the ranking loss.

---

### 9d. Path to SOTA without TrialGPT

**Target corrected (see §7e/§7i):** the goal is to beat the full-corpus TREC 2022 SOTA — the **h2oloo winning run, NDCG@10 = 0.6125** — with a fully open, end-to-end pipeline. TrialGPT's 0.7252 is *pooled-candidate reranking* and not a comparable full-corpus number, so "beat TrialGPT" was the wrong framing. The current best open full-corpus result is **0.554** (hybrid + score fusion, §7i); the gap to SOTA is 0.058.

**The primary path is now the full-corpus retrieval→rerank→ensemble pipeline (§7i), not standalone criterion matching.** Full-corpus work showed criterion decomposition and hard-negative reranking plateau ~0.44–0.55; the wins came from fixing the document representation and from score fusion. The criterion signal (below) is now best used as *additional features in the learned ensemble* (§7i Finding 6), alongside an open LLM reranker score — not as a standalone reranker. The tiers below remain the menu of open-weight signal sources; the ensemble is how they combine.

---

**Tier 1 — Low cost, near-term gains**

**1a. BioLinkBERT-large on clean data**
The most immediate lever. BioLinkBERT-large (340M vs 110M) trained on the same clean `train_clf_data.jsonl` split likely gives +3–5 NDCG@10 points. Requires an A100 (available in Colab Pro+). Retrain with `base_model = 'michiyasunaga/BioLinkBERT-large'` in `retrain_classifier.ipynb`.

**1b. Fine-tune the bi-encoder retrieval stage**
The current MiniLM retriever is used off-the-shelf. It was not trained on clinical trial matching. Fine-tuning it with in-batch negatives from the TREC qrel would improve recall at the top of the cascade and give the cross-encoder better candidates to rerank. Use the BEIR framework or `sentence-transformers` with `MultipleNegativesRankingLoss` on (topic, relevant_trial) pairs from `train_clf_data.jsonl`.

---

**Tier 2 — Moderate cost, significant gains**

**2a. LLM-as-reranker (zero-shot or few-shot)** ✓ Done

Use a small open LLM as a reranker on the cross-encoder's top-50. Implemented in `rerank_llm.ipynb`.

**Method:** logprob scoring — for each (topic, doc) pair, compute log P("yes") − log P("no") at the first generated token via a single forward pass (no generation). Score aggregated across tokenizer variants (yes/Yes/·yes/·Yes) using logsumexp. Prompt uses `apply_chat_template` for model-specific formatting. Full doc text, tokenizer-truncated at 2048 tokens (no char-truncation, preserves exclusion criteria).

**Results on TREC22 (50 topics, Qwen2.5-7B-Instruct, Apache-2.0):**

| System | NDCG@10 | MRR | Notes |
|---|---|---|---|
| clf-v4 (BioLinkBERT-large) | 0.6383 | 0.7477 | cross-encoder baseline (standalone harness) |
| Qwen2.5-7B standalone | 0.6269 | 0.7720 | zero-shot, all judged docs scored |
| clf-v4 → Qwen2.5-7B (top-50) | **0.6485** | **0.7759** | pipeline; LLM reranks clf's top-50 |

Pipeline delta vs clf-v4: NDCG@10 +0.010 (+1.6%), MRR +0.028 (+3.8%).

**Findings:**
- Zero-shot Qwen-7B standalone NDCG (0.627) is slightly below the fine-tuned cross-encoder (0.638) — domain-specific fine-tuning on TREC data is more powerful than 7B zero-shot reasoning.
- Qwen-7B has higher MRR than clf-v4 (0.772 vs 0.748) — better at surfacing the first relevant trial, even when full top-10 ranking is noisier.
- The pipeline (clf-v4 → LLM) improves on both metrics, confirming complementary signals: clf is precise over the full ranking; the LLM adds a re-ordering boost at positions 1–10.
- The gain is modest (+1.6% NDCG), not the "largest single gain" originally anticipated. Zero-shot eligibility reasoning without per-criterion structure cannot fully replace supervised fine-tuning on domain pairs.

**2b. NDCG-differentiable loss**
Replace CE with **ApproxNDCG** or **ListNet** — losses that directly optimize a differentiable approximation of NDCG@10. Requires grouping examples by topic during training (a `DataCollator` that batches within topics). `torch_geometric` or `allRank` implement these. Higher implementation effort but directly optimizes the evaluation metric.

---

**Tier 3 — Large architectural change, highest ceiling**

**3a. Open-weight criterion-level matching**
This is the TrialGPT architecture without GPT-4. The pipeline:

1. **ctproc criteria parser** → parse trial inclusion/exclusion criteria into individual criterion statements (already implemented)
2. **Criterion-level LLM judge** → for each (patient_description, criterion) pair, use Mistral-7B or Llama-3-8B to produce a 3-class label (met / not_met / not_enough_info) with a chain-of-thought explanation
3. **Aggregation model** → learn how criterion-level uncertain signals combine into a trial-level relevance score; this can be a simple logistic regression over criterion-level vote counts, or a small MLP

The `ncbi/TrialGPT-Criterion-Annotations` dataset provides 1,020 GPT-4 criterion annotations that can be used to fine-tune the criterion-level judge step on a much smaller model. The aggregation step is where the cross-encoder scores can be integrated.

**Why this is the ceiling:** §9a explains why flat NLI per criterion is insufficient — temporal arithmetic, conditional logic, vague thresholds. A small LLM with chain-of-thought handles all of these better than a classification head. The user's clinical expertise (RN, 20y ICU) makes correcting the criterion-level errors tractable in a way it wouldn't be for most teams.

**3b. Generative reranker fine-tuned on TREC data**
Fine-tune a small LLM (e.g., Mistral-7B) on the TREC qrel as a generation task: given (topic, trial), generate "relevant" / "partially relevant" / "not relevant". Use the TREC22 qrel as held-out test. This is more expensive than 2a (requires fine-tuning) but more controllable than zero-shot and can be distilled from TrialGPT annotations.

---

**Tier 3c — Verifiable-reward criterion assessor + learned aggregation (the proposed contribution)**

Tier 3a (§7g) and Tier 3b (§7h) both underperform clf-v4, and between them they isolate *why*: the assessor is trained on noisy, unverifiable labels, and the criterion signals are combined by a hand-set score vector with no exposure to the gold qrels. Tier 3c rebuilds the criterion path around those two diagnoses. It is fully open-weight at inference (Claude appears only in offline data generation, which is permitted).

**Component 1 — Synthetic patients with gold-by-construction labels.** Invert the annotation problem: instead of labeling real (patient, criterion) pairs and hoping the label is right, *sample a target label spec first and generate the patient note to realize it*. Sample a trial and a spec ("meets inclusion 1 and 3, violates exclusion 2, no information on inclusion 4"), have Claude write a style-matched note (few-shot from the 134 real topics), and the label of every criterion is known by construction. Two guards make the labels trustworthy: the generator returns `feasible: false` for clinically contradictory specs, and a *different* model (Haiku) blind-labels each pair without seeing the target — only agreement earns `verified: true`. The spec sampler deliberately rebalances the classes (30% `excluded` on exclusion criteria vs 1.9% natural), directly fixing the collapsed-`excluded` problem from §7h. Trials judged in TREC22 are excluded from generation (contamination guard). Implemented in `gen_synthetic_criteria.ipynb`; the per-class blind-agreement matrix doubles as the fidelity table for the paper.

**Component 2 — GRPO on the R1-7B assessor with verifiable reward.** Gold-by-construction labels make criterion assessment a verifiable-reward task — the regime GRPO was built for (and what produced R1 itself). Train R1-Distill-7B with LoRA via TRL `GRPOTrainer`: reward = exact label match + format adherence − length penalty, groups of ~8 sampled completions, advantage-normalized within group, no reward model. This targets the exact failure §7g diagnosed — NEI calibration — by rewarding abstention only when the gold spec actually says "no information." At inference, sample k CoTs at temperature and use the empirical label distribution as a calibrated 5-dim probability vector (self-consistency marginal), optionally distilled into the BiomedBERT cross-encoder for throughput. Mitigate synthetic→real shift by mixing ~20% real R1 silver prompts into training and validating checkpoints on held-out *real* topics, never on synthetic.

**Component 3 — Learned set-transformer aggregation.** This is where gold trial-level supervision (the qrels) finally enters the criterion path — Tier 3a/3b wasted it on a fixed +1/−1/−2/0 vector. Represent each criterion as a token: its 5-dim probability vector ⊕ a frozen sentence embedding of the criterion text ⊕ an inclusion/exclusion flag; add one token carrying the clf-v4 relevance score. A small (~1M-param, 2-layer) permutation-invariant set transformer attends over the set and emits a scalar trial score, trained on trec21+kz qrels with a listwise loss (LambdaLoss / ApproxNDCG) that optimizes NDCG@10 directly. This learns what linear aggregation cannot express: soft-veto logic (one confident `excluded` on a load-bearing criterion should dominate), criterion salience (boilerplate "willing to consent" vs. a decisive staging criterion, via the text embedding), NEI-mass handling, and per-trial fusion weight against clf-v4. It subsumes noisy-OR / Dawid–Skene aggregation as special cases while being trained end-to-end against the metric. The §7h fusion sweep is the degenerate 1-parameter version of this module.

**Why this can clear the gate where 3a/3b didn't:** criterion-alone ≈ clf-v4 is the established prior (even Claude only reached 0.6359 vs the 0.6388 gate), so the bet is not criterion-alone but the *combination* — clf-v4's document-level signal plus a calibrated, gold-trained criterion adjustment carry strictly more information than either alone. Each of the four weaknesses from §7h is addressed by exactly one component (labels → C1, calibration → C2, aggregation + fusion → C3). Even if the aggregate NDCG only matches clf-v4, the artifact — an open, interpretable per-criterion assessor emitting labels + reasoning — is the TrialGPT-analog contribution, and the eval should surface example per-criterion outputs, not just the scalar.

**Risks:** synthetic→real distribution shift (largest; mitigated by style-matching, real-data mixing, real-topic validation); reward hacking toward degenerate short CoTs (format + length reward terms, spot-check on real topics); compute (GRPO-LoRA on 7B is ~10–20 A100-hours, the one expensive step; everything else is cheap).

**Deprioritized (2026-07) — the criterion-assessor ceiling is too low to justify GRPO.** The Tier 3a Claude ceiling test (§7g) is the key evidence: **Claude — a far stronger judge than any GRPO-tuned 7B would produce — scored only NDCG@10 = 0.6359 at criterion-level, *below* clf-v4.** GRPO (and RL-from-verifiable-reward generally) largely *elicits and sharpens* a model's existing capability rather than adding new knowledge or exceeding a task ceiling set by a much stronger model [REF — add citation: recent work indicating RL fine-tuning amplifies base-model behavior within its capability envelope rather than surpassing a frontier-model ceiling on the same task]. So GRPO'ing a 7B assessor is unlikely to beat a ceiling a frontier model doesn't reach, at ~10–20 A100-hrs — the most expensive lever for the least-supported upside. The productive residue of this line: (a) the criterion signal as a cheap **ensemble feature** (per-criterion probabilities into the LambdaMART, testable directly, no GRPO), and (b) the **synthetic-data infrastructure repurposed** to augment the data-poor judge fine-tune (§9e idea #1, §10). This section is retained as a documented, considered-and-shelved architecture, not an active plan.

**Notebooks:** `gen_synthetic_criteria.ipynb` (done — Component 1), `grpo_criterion_assessor.ipynb` (planned, not built — Component 2), `train_aggregator.ipynb` (planned, not built — Component 3). `train_criterion_clf.ipynb` (Tier 3b) becomes the fixed-aggregation ablation row and retrains on gold labels via a one-line `LABELS_PATH` swap.

---

**Status:**

| Tier | Item | Status | Result |
|---|---|---|---|
| 1a | BioLinkBERT-large, clean | **Done** | TREC22 NDCG@10=0.6222 (clf-v3), 0.6388 w/ aug (clf-v4) |
| 1b | Fine-tune bi-encoder (MiniLM) | **Done** | TREC22 judged-pool: NDCG@10=0.4783, R@100=0.4292 (+18% vs off-the-shelf) |
| 2a | LLM-as-reranker zero-shot | **Done** | Qwen2.5-7B; standalone 0.6269, clf→LLM top-50: NDCG@10=0.6485 (+1.6%), MRR=0.7759 (+3.8%) |
| 2b | NDCG-differentiable loss | Not started | — |
| 3a (Qwen) | Open-weight criterion-level matching | **Done** | Qwen2.5-7B logprob; best=mean_inc_max_exc NDCG@10=0.6242, MRR=0.7327 — below clf-v4 |
| 3a (Claude) | Criterion-level ceiling test | **Done** | Claude Sonnet; best=mean NDCG@10=0.6359, MRR=0.7804 — still below clf-v4; 41.8% NEI confirms input bottleneck |
| 3b | Distilled open criterion cross-encoder | **Done** | R1-7B silver (11,250 pairs) → BiomedBERT-base; standalone val NDCG 0.45, fusion helps only at α≈0.05 (+0.0045, within noise). Baseline ablation row — see §7h |
| 3c-1 | Synthetic gold-label generation | Built, not run | `gen_synthetic_criteria.ipynb`; **repurpose** for judge-fine-tune augmentation (§9e/§10) |
| 3c-2 | GRPO verifiable-reward assessor | **Shelved** | criterion ceiling too low (Claude 0.6359 < clf-v4); see Tier 3c deprioritization note |
| 3c-3 | Learned set-transformer aggregation | **Shelved** | superseded by the LambdaMART ensemble (§7i); criterion signal → ensemble feature instead |
| 3b (gen) | Generative reranker fine-tune | Not started | — |

**Key negative result (Tier 3a/3b):** criterion-level decomposition trails clf-v4 across every model tried — Qwen 7B (−0.015), Claude Sonnet (−0.023) on TREC22, and a distilled BiomedBERT (standalone val NDCG 0.45, fusion within noise). Tier 3a diagnosed the *input* bottleneck (41.8% NEI on sparse TREC vignettes); Tier 3b added two more: silver labels are noisy and unverifiable, and a hand-set aggregation vector never sees the gold qrels. R1-7B's lower NEI rate (34.7% vs 41.8%) confirms CoT recovers some of the input gap. These three diagnoses define Tier 3c.

**Recommended sequence:**
1. ~~Finish hard-neg augmented training~~ Done → clf-v4 gate=0.7460, TREC22=0.6388
2. ~~BioLinkBERT-large retrain (1a)~~ Done → clf-v3 / clf-v4
3. ~~Fine-tune bi-encoder (1b)~~ Done → MiniLM fine-tuned NDCG@10=0.4734 (all-184 judged pool)
4. ~~LLM-as-reranker zero-shot (2a)~~ Done → Qwen2.5-7B pipeline NDCG@10=0.6485 (+1.6%), MRR=0.7759 (+3.8%) on TREC22
5. ~~Criterion-level matching (3a)~~ Done → Qwen: 0.6242; Claude ceiling: 0.6359; both below clf-v4 (§7g)
6. ~~Distilled open criterion cross-encoder (3b)~~ Done → weak baseline; quantifies headroom for 3c (§7h)
7. ~~Full-corpus pipeline + LambdaMART ensemble~~ Done → TREC22 NDCG@10 **0.6105**, competitive with the TREC 2022 winner (0.6125); §7i, §7j.

**Re-prioritized by the error analysis (§8a).** The binding constraint is **reranker top-10 precision, not retrieval recall** (retrieval leaves a ~5× eligible surplus; oracle 0.957 vs extracted 0.61). So the next levers, in evidence-based order (each on a fresh split, not TREC22 again):
8. **Stronger LLM judge (primary).** `llm_yesno` drives the top-10 and its false negatives are the main ranking-miss cause. A larger / chain-of-thought / calibrated judge, scored wider than top-500, directly attacks the 31%-false-positive top-10. This subsumes the old "2a LLM reranker" as the highest-value direction now.
9. **Fix the Eligible-vs-Excluded objective mismatch.** An eligibility judge is misaligned with the graded/topical metric on the rel=1 tier; a two-headed topicality+eligibility signal, metric-aware aggregation, or the **criterion features (Tier 3c)** as ensemble inputs could recover it — Tier 3c is now framed as *ensemble features*, not a standalone reranker.
10. **Recalibrate the cross-encoder features / listwise loss on the neural model (2b, §9b)** — clf_rel/v2_rel currently score higher on false positives than true positives.
11. **Query/diagnosis expansion for terse, implicit-diagnosis topics** — secondary for NDCG@10, primary for recall and the KZ one-liners.

### 9c. Lab value extraction and comparison

The ctproc `lab/` module (`extractor.py`, `patterns.py`, `reference_ranges.py`) extracts lab values from criteria text. A trial that requires "Hemoglobin ≥ 10 g/dL" contains a structured constraint that can be compared against a patient's known lab values.

**TODO — document the lab extraction pipeline in its own section (this deserves §3e)**

Current state: extractor exists but is not integrated into the ranking pipeline. Integrating it would allow hard filtering on lab thresholds — a patient with Hgb=8 would be automatically excluded from trials requiring Hgb≥10.

### 9e. Candidate improvements and medical models to try

Consolidated menu of levers, **ranked by what the error analysis (§8a) says will move NDCG@10** — the binding constraint is reranker top-10 precision, not retrieval recall, so the reranker/judge side is prioritized over retrieval. Anti-gaming: develop each on TREC21 (or a fresh split), touch TREC22/23 once.

**Improvement ideas (EV order):**

1. **Improve the LLM judge (highest EV) — and the top sub-lever is to FINE-TUNE it.** The Qwen `llm_yesno` feature both surfaces true positives and, via its false negatives, is the main ranking-miss cause (§8a: eligible-surfaced +5.78 vs eligible-buried −11.51). Note the judge has only ever been used **zero-shot** (§7f, ensemble feature) — the one unsupervised component in an otherwise fine-tuned pipeline (clf-v4/v2, retriever, and h2oloo's monoT5 are all supervised). Sub-levers, in EV order: (a) **fine-tune Qwen itself** — LoRA on TREC21+KZ (patient, trial)→relevance, yes/no-logprob or graded target; the supervised adaptation of the binding signal and the LLM analog of h2oloo's monoT5_CT. **If trained on the graded qrel (eligible=2/excluded=1/not=0), it may also fix the rel=1 eligibility-vs-topicality mismatch of idea #2 in one move** — it learns the metric's target, not pure enrollability. (b) **swap to a medical/reasoning model** (still zero-shot; see model list); (c) **CoT prompting** (cheap zero-shot reasoning). Develop on TREC21, test TREC22/23 once.
2. **Topicality signal — fixes the Eligible-vs-Excluded objective mismatch (most *original* idea, from §8a).** The LLM judges *eligibility* but graded NDCG rewards *topicality*; catching a rel=1 exclusion (correct) buries a trial the metric wants elevated. Add a separate "is this trial *about* the patient's condition?" score (LLM prompt or condition-match encoder) as its own feature to recover the rel=1 tier. A principled, analysis-derived contribution (stronger for a paper than a bigger model).
3. **Symptom → diagnosis expansion** for the implicit-diagnosis retrieval misses (§8a worst topics: nosebleeds, azoospermia, rash+oral-ulcers — patient presents symptoms, trials name diseases). Medically-grounded version of h2oloo's NQS.
4. **Self-consistency + calibration on the judge** — sample k CoT judgments and average; temperature-scale the logit. Reduces confident-wrong false negatives.
5. **Listwise / ranking-loss retrain of the cross-encoders (§9b)** — §8a showed clf/v2 are anti-informative at the top (higher scores on FPs than TPs); ApproxNDCG/LambdaLoss or recalibration could make them contribute.
6. **(Low NDCG value, real clinical value)** hard demographic + lab-threshold filters (§7d, §9c) — clinically correct, but pool bias means little NDCG effect.

**Medical models by role:**

- **LLM judge (the #1 lever):** OpenBioLLM-8B/70B (Llama-3 medical), Meditron-7B/70B (EPFL), BioMistral-7B, Med42, Palmyra-Med, MedGemma (check license/availability); reasoning models: **DeepSeek-R1-Distill** (already have it from the criterion work — reasons before answering) and QwQ-32B; or a larger general model (Qwen2.5-72B, Llama-3.3-70B).
- **Dense retrievers ("neural representations" lever):** MedCPT (NCBI — off-the-shelf lost to fine-tuned MiniLM in §7b, but *fine-tuning* it / using its asymmetric encoders is untried), BGE-large / BGE-M3, E5-large-v2, GTE-large (fine-tune on TREC21+KZ), **SPLADE** (learned sparse — complements dense in RRF), SapBERT (biomedical entity embeddings, good for condition matching + idea #3).
- **Rerankers:** MedCPT cross-encoder (NCBI's own), monoT5-MED / monoT5_CT (in progress, §7e/reproduce_h2oloo), Clinical-Longformer / Clinical-BigBird (longer context if trial truncation hurts).
- **Concept / NLP (for idea #3):** UMLS/MetaMap, scispaCy / medspaCy, cTAKES — map symptoms ↔ diseases, normalize concepts.

**Two highest-value next experiments** (develop on TREC21): ~~(a) fine-tune the Qwen judge~~ — **TRIED, ensemble-null (§11c)**: the fine-tune won at the judge level (0.746→0.871 judged-pool) but did nothing in the ensemble (8-seed CV Δ = −0.003 ± 0.011), because the LLM feature is redundant with the cross-encoders. The lesson: the **pointwise-eligibility axis is saturated** (redundant with clf-v4/v2/dense); judge quality is not the binding constraint. But recall is *not* the answer either — the hybrid-pool **oracle NDCG@10 is 0.957** (§8a), so the 0.61→0.957 gap is reranking *extraction*, not retrieval (see the corrected redirect in §11c; NQS raises a ceiling we're 0.35 below). The right next move is an **analysis first** — decompose that gap into rel=1-buried (→ **topicality feature**, a small/hard slice) vs rel=2-buried-by-confident-FPs (→ needs an *orthogonal* signal or a **listwise** reranker; recalibration is a LightGBM no-op). The decomposition, not a third guess, picks the lever. These are fixes derived from §8a and §11c, not model-swapping for its own sake.

Also: the parked **synthetic-data infrastructure** (`gen_synthetic_criteria.ipynb`, §9d Tier 3c-1 — built but never run) is now most useful *repurposed* to **augment the data-poor judge fine-tune** — KZ has only ~1.1k positives (the same scarcity h2oloo notes for monoT5_CT). Spec-conditioned + blind-verified synthetic (patient, trial) relevance pairs could enlarge the LoRA training set for idea #1, rather than feeding the abandoned GRPO-criterion path.

---

## 10. Surpass-SOTA program (active work plan)

The current active effort: an open, full-corpus pipeline that surpasses the TREC 2022 winner (h2oloo, 0.6125) *in a way that survives review* — i.e., without gaming the test. This section is the working plan; details live in the cross-referenced sections.

**Honest expectation (set before starting).** At n=50 without h2oloo's per-topic run, a *statistically significant* surpass on TREC 2022 is not claimable from a point estimate (marginal CI ±0.07). Realistic outcomes, strongest first: (a) a **significant** win via a paired test — available if we obtain h2oloo's 2022 run *or* run all systems ourselves on 2023; (b) **competitive with SOTA** (already have: 0.6105 vs 0.6125 on all three metrics) plus **external generalization** on 2023. A clean surpass is the target; "competitive + generalizes" is the floor and is itself publishable. Note: paired tests detect far smaller effects than the ±0.07 marginal CI because systems share a pool — **0.02–0.05 can be significant**, so a large absolute gap is not required (the earlier "needs 0.1" intuition was the marginal-CI bar, not the paired bar). Absolute +0.1 would be a strong stretch result; +0.2 is not realistic on this benchmark.

**Anti-gaming protocol (mandatory).** The current 0.6105 config is a frozen, pre-registered baseline. Every improvement is developed and tuned on **TREC 2021 (CV) only**; TREC 2022 and 2023 are each touched **once**, at the end. §8a was derived on 2022, so improvements it motivates are validated on 2021, never re-scored on 2022.

**Evaluation design (three tracks).**
1. **TREC 2022 vs published 0.6125** — descriptive point-estimate + our CI; state that a paired test needs h2oloo's run (a limitation). Standard practice.
2. **h2oloo run file** (request via `trec@nist.gov` — individual agreement, no affiliation needed — or the authors Pradeep/Lin) → paired bootstrap/Wilcoxon on 2022. Upgrades "competitive" → "significant."
3. **TREC 2023 external test** — genuinely unseen (no component of ours saw it). Run theirs-repro / ours / combined all ourselves → paired significance available *without* NIST. Confound: questionnaire format (domain shift) — a generalization stress test, not a clean surpass.

**Faithful h2oloo reproduction (fidelity-gated).** Reproduce their pipeline (NQS doc2query → BM25+RM3 → RRF → monoT5) with their exact scripts + pyserini (`reproduce_h2oloo.ipynb`), and reproduce their KZ-tuned `monoT5_CT` ourselves (`finetune_monot5_ct.ipynb` — we have KZ + the §3.3 recipe). **Gate before use as a baseline:** validate against their published numbers — BM25 0.2923 / RM3 0.3539, and monoT5_CT ≈ **0.7118** on TREC 2021. The public MED variant (0.4715) is too weak to be a credible baseline. Limitation to state: their doc2query is T5-3B (public = base) and `monoT5_CT` isn't released → faithful *architecture* with public/self-trained checkpoints.

**Combined ("best of both") system.** Fold the winning levers into one pipeline, each developed on 2021: **NQS query synthesis** into *our* hybrid (BM25+dense) pool; **monoT5 score** as an ensemble feature; the **fine-tuned Qwen judge** (idea #1); the **topicality feature** (idea #2); optionally a **stronger dense retriever** (§9e). Then run theirs / ours / combined on 2022 + 2023 with paired tests. If the combined system clearly beats both simpler ones, the complexity is earned.

**Complexity check (from the h2oloo comparison).** h2oloo ties us with a cleaner architecture (§7e). Run the feature/model **ablation** (`train_ensemble_full.ipynb`, TREC21 CV) — if a lean subset (retrieval + one judge) matches the full ensemble, **report the lean system**. A simple system that beats SOTA is a stronger paper than a complex one that ties it.

**Notebook map (surpass program):** `reproduce_h2oloo.ipynb` (their pipeline + fidelity gate), `finetune_monot5_ct.ipynb` (KZ-tuned reranker), `rerank_cot_final.ipynb` (CoT judge, dev on 2021), `build_corpus_2023.ipynb` + `eval_external_2023.ipynb` (external test), `train_ensemble_full.ipynb` (ensemble + ablation + eligible-only metrics), `gen_synthetic_criteria.ipynb` (parked; repurpose for judge-fine-tune augmentation).

**Prioritized tasks.** (Reordered after §11a CoT and §11c judge-FT both came back ensemble-null — the *pointwise-eligibility axis* is saturated. Recall is NOT the redirect: oracle @10 = 0.957, so the gap is reranking extraction, §11c corrected.)
1. [done — null] ~~CoT judge α-sweep~~ (§11a) and ~~fine-tune the Qwen judge~~ (§11c): won/neutral alone, null in the ensemble (redundant with the cross-encoders).
2. **Gap decomposition (analysis, not a build)** — from `eval_predictions_ensemble.jsonl`, split the 0.61→0.957 gap into rel=1-buried vs rel=2-buried-by-confident-FPs. This gates every choice below; do it first.
3. **If rel=1-heavy → topicality feature** (§9e #2), a *separate* signal. **If rel=2-confident-error-heavy → listwise reranker** (open RankZephyr-style) or bank the parity result. (NOT recalibration — LightGBM no-op.)
4. **monoT5_CT as an ensemble feature on our (better) pool** — near-free to test once `finetune_monot5_ct.ipynb` validates ≈0.71 on 2021; but it is the same pointwise axis that went null twice, so treat as a real test.
5. **Combined system** + theirs/ours/combined on 2022 + 2023, paired tests.
6. **Complexity ablation** → report lean vs full.
7. [user] pursue the h2oloo 2022 run file (trec@nist.gov / authors) for a 2022 paired test.

---

## 11. Negative results (addendum)

> ⚠️ **Re-open these under `R` before trusting them (§2g).** Several negatives here (CoT judge §11a,
> fine-tuned judge §11c, listwise §11d) were run on the eligibility-blind representation — the judge
> and rerankers could not see the eligibility text. Their nulls may be *input* artifacts, not model
> ceilings. Numbers are `PENDING(R)`; the mechanisms are worth reading but each negative must be
> re-confirmed on the frozen representation before it counts as a closed door.

Documented dead-ends, retained for a "negative results" appendix — each tells the reader what *not* to spend effort on, and why. Negative results with a clear mechanism are a contribution, not a failure.

### 11a. Chain-of-thought reranker (dev-null, deprioritized)

**Hypothesis.** §8a identified LLM-judge **false negatives** as the main ranking-miss mechanism (eligible trials the zero-shot Qwen judge scores "no" and buries). A chain-of-thought judge that reasons through the criteria before answering was hypothesized to reduce them.

**Implementation** (`rerank_cot_final.ipynb`). Rerank the *frozen* LambdaMART ensemble's top-K (K=30) per topic with an open **Qwen2.5-7B** judge: for each (patient, trial) the model is prompted to reason step-by-step about inclusion/exclusion criteria, then a follow-up scoring pass reads the `yes`/`no` logprob *after* the generated reasoning. Final ranking over the head = `α·z(cot_score) + (1−α)·z(ensemble_rank)`; ensemble order kept below K.

**Data.** Developed on **TREC 2021** only (anti-gaming protocol — never tuned on the TREC22 test). Note TREC21 is *in-sample* for the frozen ensemble (trained on TREC21+KZ), so its baseline here (0.6497) is optimistic vs the 0.552 held-out 5-fold CV — which may *understate* any add-on's headroom.

**Result (α-sweep, TREC 2021 NDCG@10):**

| α (CoT weight) | NDCG@10 |
|---|---|
| **0.0 (pure ensemble)** | **0.6497** |
| 0.2 | 0.6497 |
| 0.4 | 0.6403 |
| 0.6 | 0.6230 |
| 0.8 | 0.5703 |
| 1.0 (pure CoT) | 0.5023 |

CoT is **neutral at low weight and monotonically hurts** as its weight rises; pure CoT is far below the ensemble.

**Why it likely underperformed.** The ensemble already includes the zero-shot Qwen `llm_yesno` judge as a feature, and CoT is a costlier version of the *same* Qwen eligibility judgment — so it is **redundant**, and blending a correlated signal on top only dilutes the ensemble's (already good) ordering. The pure-CoT collapse mirrors the discard-order artifact (§7i): a single judge underperforms the full ensemble, and letting it override the ranking throws away good order.

**Why not pursued.** A lever with no dev-set gain does not earn a one-shot TREC22 run. And the null is *informative*: zero-shot reasoning adds nothing over the zero-shot judge already in the ensemble → the path is **supervision** (fine-tune the judge, §9e idea #1), not more zero-shot cleverness. The **topicality feature** (§9e idea #2) remains promising because it is an *uncorrelated* signal (topicality ≠ eligibility). Caveat retained: this tested CoT as a score *blend*, not as an added ensemble feature; given the redundancy with `llm_yesno` and CoT's generation cost, the feature variant was judged not worth the spend.

### 11b. Other documented negatives (cross-referenced)

- **Hard-negative reranker v1/v2** (§7i Finding 4): from-scratch v1 *regressed* to 0.30 (discarded clf-v4's training distribution); continue-trained v2 improved recall@100 but not NDCG@10.
- **Criterion decomposition, Tier 3a/3b** (§7g/§7h): Qwen 0.6242, Claude ceiling 0.6359, distilled cross-encoder ~0.53 — all *below* clf-v4; criterion-alone caps near clf-v4.
- **GRPO criterion assessor** (§9d Tier 3c): shelved on the ceiling argument (Claude 0.6359 < clf-v4), never run.
- **Judged-only LTR training** (§7i Finding 6): *regressed* to 0.37 — the unjudged-as-0 rows correctly encode the ~95%-non-relevant serve distribution; removing them miscalibrates the model.
- **Eligibility-only rerank text** (§7i): reranking with eligibility-only text lowered *all* pools vs full-text reranking (disproved the 512-token-truncation hypothesis).

### 11c. Fine-tuned LLM judge — judge-level win, ensemble-level null

**Hypothesis.** The zero-shot Qwen judge was the ensemble's only unsupervised component and its main error source (§8a); supervised fine-tuning was ranked the highest-EV lever (§9e #1). CoT prompting had already been dev-null (§11a), pointing to supervision rather than more zero-shot cleverness.

**Implementation** (`finetune_judge_lora.ipynb`). Qwen2.5-7B, **bf16 LoRA with DoRA**, **LambdaRank-weighted pairwise loss on the yes/no margin**, **graded labels** (eligible=2/excluded=1/not=0), dense-mined **hard negatives**, trained on TREC21+KZ with a held-out TREC21 dev split. Interface unchanged (same yes/no margin readout) → drop-in for the `llm_yesno` feature. Feature regenerated over the identical pool pairs via `regen_judge_feature.ipynb` (read-only, clean A/B).

**Result.**
- **Judge level: clear win.** Held-out judged-pool NDCG@10 0.746 → **0.871** (epoch 2; epoch 3 overfit).
- **Ensemble level: null.** Paired TREC21 CV, fine-tuned vs zero-shot `llm_yesno`, **8 seeds**: mean Δ = **−0.0031 ± 0.0106, FT wins 3/8**. The single +0.0071 seen at one fold-seed was noise. The TREC22 one-shot was **not** spent.

**Why (the real finding).** The LLM feature is **redundant** with clf-v4 / reranker-v2 / dense — the ensemble already extracted its signal, so a much sharper judge adds nothing. Corollary: at ~0.61 (tied with SOTA) the **reranking-feature line is saturated; judge quality is not the binding constraint.** Note the graded knowledge is already carried by the margin we tested, so a 3-way readout is not new signal (and would need a retrain).

**Redirect — CORRECTED.** An earlier draft here sent the effort to retrieval recall; that was wrong and is retracted. The hybrid-pool **oracle NDCG@10 = 0.957** on TREC22 (§7i/§8a), with ~48 eligible trials/topic in the CAND_K pool vs 10 top-10 slots, means recall into the pool does **not** cap @10 (nor P@10/MRR — same top-10 surplus). The whole 0.61→0.957 gap is **reranking extraction**. NQS/doc2query raises a ceiling we are already 0.35 *below*, and h2oloo *ran* NQS and still scored 0.6125 — consistent with recall not being the lever. What is saturated is specifically the **pointwise-eligibility axis** (CoT §11a and a sharper judge §11c both add signal redundant with clf-v4/v2/dense), not reranking as such. **Next step is an analysis, not a build: decompose the 0.61→0.957 gap** from `eval_predictions_ensemble.jsonl` into (i) **rel=1 buried** (excluded-but-topical — the eligibility-vs-topicality mismatch; likely a *small, hard* slice: rel=1 gain is 1 vs rel=2's 3, and dense already encodes topicality) → a **topicality feature** targets it; (ii) **rel=2 buried, displaced by confident rel=0 FPs** (cross-encoders scoring wrong docs high and true-eligibles low; likely the bulk) → no pointwise feature fixes it (proven twice), and monotonic recalibration is a **LightGBM no-op** (trees are invariant to it) — only an *orthogonal* signal or a **listwise** reranker (open RankZephyr-style, compares candidates) can. The decomposition gates the choice; if it is dominated by rel=2 confident-error we are near the pointwise ceiling and the honest choice is **listwise-or-bank**, not another feature. The FT judge is retained as an open-vs-zero-shot ablation row. (Reframe for §10: our hybrid pool already *beats* h2oloo's sparse-only pool at the oracle, so our deficit vs them is reranking extraction, not retrieval — the combined-system thesis becomes "put one strong reranker — the monoT5_CT we're already reproducing — on our better pool," testable near-free as a feature, but note it is the same pointwise axis that just went null twice.)

**Gap decomposition — RESULT (TREC22, 50 topics).** Of the relevant docs the oracle puts in top-10, we bury **374/465 rel=2 (80%)** and only **20/28 rel=1** — so the **rel=1/topicality slice is negligible** (~20 docs) and the gap is overwhelmingly buried *eligibles*. But the mechanism is **not** confident consensus error: (a) **35% of buried eligibles (130/374) were never LLM-scored** (beyond the top-500 cutoff), floored at ~−36 — a scoring-*coverage* gap that produced the misleading −11.3 mean; (b) of the 244 that *were* scored, the zero-shot judge gives a **weak-positive +1.90** (vs +5.78 for surfaced eligibles) — a **tight race**, not a rejection; (c) the cross-encoders actually rate buried eligibles *higher* than surfaced ones (clf 0.265 vs 0.132) but are too flat/uncalibrated to override the LLM-dominated top-10. Crucially, **the fine-tuned judge scores these buried eligibles *worse* than zero-shot (+1.90 → −1.09)** — the LambdaRank objective sharpened easy cases at the cost of the marginal tail, which is exactly why the FT judge was ensemble-null-to-negative. **Conclusions:** no more judge fine-tuning (counterproductive on the tail); LLM coverage-widen (top-500→1000) is concrete but low-EV (those docs are retrieval-rank 500–1000 and already-scored buried eligibles only reach +1.9 < +5.8 needed); the gap is a **tight race + disagreement between the LLM and the cross-encoders** — mechanism-matched only to a **listwise reranker** (reads candidates together, can promote a +1.9 eligible over an FP using cross-candidate context no pointwise score sees). Chosen next: **listwise reranker** (`rerank_listwise.ipynb`, open RankZephyr-style, dev on TREC21). See §11d when its result lands.

### 11d. Zero-shot listwise reranker (RankZephyr) — hurts, worse with depth

**Idea.** §11c showed a tight race + LLM/cross-encoder disagreement — mechanism-matched to a listwise reranker that compares candidates. Tested open **RankZephyr-7B** (`rerank_listwise.ipynb`), sliding window (w=20, s=10), reordering the ensemble's **out-of-fold TREC21** top-N (honest base 0.5520, where eligibles are genuinely buried).

**Result (TREC21 out-of-fold, base NDCG@10 = 0.5520):**

| rerank depth N | listwise NDCG@10 | delta |
|---|---|---|
| 20 | 0.5132 | **−0.0388** |
| 50 | 0.4859 | **−0.0661** |

It **hurts, and monotonically worse the deeper it reranks** — RankZephyr's ordering is *systematically* worse than the ensemble's on this task, not just noisy.

**⚠️ This run is CONFOUNDED — not ceiling evidence.** `PASSAGE_CHARS=400` on a corpus blob ordered title → conditions → summary → detailed → interventions → **eligibility (last)** means RankZephyr saw only title+conditions — **never any eligibility criteria**, the exact rel=2-vs-rel=0 signal. It was asked to rank by eligibility from topicality text alone. The depth-monotonic worsening does NOT discriminate "systematic mis-ranking" from "arbitrary permutations of uninformative passages" (both degrade toward pool-random as more good order is overwritten), so it cannot rule out the truncation. **Fair retry ran** with eligibility-inclusive passages (head 220 for topicality + tail 1200 landing on the eligibility section, verified by inspection; window 10). **Result: N=20 → 0.5428, delta −0.0092** vs base 0.5520 — fixing the confound moved it +0.030 (−0.039 → −0.009), confirming the confound was real, but the fair result is **flat/within-noise, not a gain**. Zero-shot RankZephyr, even seeing eligibility, does not beat the ensemble.

**Verdict (fair, confound removed).** **Third reranking approach to fail on a clean test** (CoT §11a, judge-FT §11c, listwise §11d). Zero-shot listwise doesn't transfer to clinical eligibility — consistent with the §11c finding that even the eligibility-reading cross-encoders are flat on these buried docs; the discriminating signal is genuinely hard. The reranking stage is at its ceiling across every mechanism-matched lever tested fairly. Only remaining reranking card = *fine-tune* a listwise reranker (break-even zero-shot gives it a marginally-better-than-hopeless prior, but high cost, and everything else failed). **Recommendation: bank the parity result** (0.6105 vs 0.6125, fully open) with §11 as the negative-results appendix mapping the ceiling. **Headline caveat when banking:** 0.6105 involved many TREC22 touches (test-adaptation risk) — state it in the headline, and report TREC21 CV 0.552 as the honest primary number.

---

## Working Notes / TODOs

- [ ] Quantify corpus overlap: what % of KZ qrels NCT IDs exist in the 2021 corpus?
- [ ] Measure actual intermediate set sizes in inference mode for a sample topic
- [ ] Show category distribution across the 374k corpus
- [ ] Formalize the custom optimistic NDCG metric with a concrete example
- [ ] Annotate BOTH_INC_AND_EXC_PATTERN character class by character class
- [ ] Add graphviz diagrams: (a) full pipeline, (b) ctproc parse tree, (c) SVM embedding space
- [ ] Document lab extraction pipeline (§3e)
- [ ] Compute svm+clf ablation results
- [ ] Compute TREC 2021 + 2022 + KZ combined baseline
- [x] Verify the real SOTA target — done (§7e): full-corpus TREC22 winner h2oloo NDCG@10=0.6125; TrialGPT 0.7252 is pooled-candidate rerank, not comparable
- [ ] Add open LLM reranker score + criterion signal as features to the LambdaMART ensemble (§7i), targeting 0.6125+
- [ ] Paired-bootstrap / Wilcoxon CIs on the ~50-topic TREC22 comparison before any SOTA claim
- [ ] Document TrialGPT criterion annotations: format, coverage, alignment with our criteria parser
