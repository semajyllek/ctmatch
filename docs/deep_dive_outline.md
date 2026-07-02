# Clinical Trial Matching: A Deep Dive
## From Raw Eligibility Criteria to Ranked Retrieval

> **Status:** Outline + working notes. Expand each section into full prose, code blocks, diagrams, and math.
> Intended audience: graduate-level ML/NLP, or a clinical informaticist who can read Python.

---

## Document map

```
1. The Problem
2. The Data
   2a. ClinicalTrials.gov structure
   2b. TREC 2021 / 2022 datasets
   2c. KZ (Koopman-Zuccon) dataset
   2d. Structural differences: judged pool, topic format, relevance scale
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
   7e. Comparison to TrialGPT
8. Error Analysis
   8a. Confusion matrix breakdown
   8b. FP error patterns (cardiac enrichment, label quality)
   8c. FN error patterns
   8d. Data quality: label errors vs. model errors
9. Future Directions
   9a. Criterion-level entailment
   9b. RLHF / DPO on clinical judgments
   9c. Lab value extraction and comparison
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

**TODO — table showing dataset stats:**

| | TREC 2021 | TREC 2022 |
|---|---|---|
| Topics | 75 | 50 |
| Topic format | XML `<topic number="N">free text</topic>` | same |
| Corpus | ClinicalTrials.gov 2021-04-27 dump | same |
| Judged docs/topic (approx) | 40–120 | 500–900 |
| Relevance scale | 0/1/2 | 0/1/2 |
| Qrels file | `trec_21_judgments.txt` | `qrels2022.txt` |

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
- TREC 2022 has far more judged docs per topic (~708 avg vs. ~80 avg for 2021). This dramatically increases eval time for any pipeline that processes the full judged set.
- Same corpus — results are directly comparable between years
- Relevance: 2=eligible, 1=partially eligible, 0=not eligible

### 2c. KZ (Koopman-Zuccon 2016) dataset

**TODO — show actual topic file format:**

```
<TOP>
<NUM>1</NUM>
<TITLE>35 year old female diagnosed with anorexia nervosa...</TITLE>
</TOP>
```

This is pseudo-XML: no proper root element, parsed line-by-line in ctproc.

| | KZ |
|---|---|
| Topics | 60 |
| Topic format | pseudo-XML `<NUM>/<TITLE>` |
| Corpus | ClinicalTrials.gov 2015 dump |
| Judged docs/topic | ~1,000–1,400 |
| Relevance scale | 0/1/2 |
| Source | SIGIR 2016 paper |

**Notes:**
- KZ topics are shorter/simpler than TREC ("35 year old female with anorexia nervosa" vs. a full clinical narrative)
- The KZ judged pool is far larger per topic. This likely reflects a more systematic pooling strategy (deeper pool = more docs judged), not that there are more relevant trials
- The 2015 corpus vs. 2021 corpus means some NCT IDs in KZ qrels may not appear in the ctmatch corpus — those are silently dropped by `get_doc_indices()`
- **This corpus mismatch is a real issue** — worth quantifying what fraction of KZ judgments are found in the 2021 corpus

### 2d. Structural differences summary

**TODO — graphviz diagram showing the three dataset shapes side by side**

Key differences that affect system design:
1. **Topic length**: TREC topics are full clinical narratives (~100–200 words); KZ topics are one-liners. The category prior (bart-large-mnli) works better on longer text.
2. **Judged pool depth**: KZ ~1,200, TREC 2022 ~700, TREC 2021 ~80. Eval time scales linearly with pool depth × pipeline cost.
3. **Corpus vintage**: KZ uses 2015 dump. ~15–20% of KZ judged trial IDs may be absent from the 2021 corpus.
4. **Annotation provenance**: TREC judgments made by NIST assessors; KZ by the paper authors. Different inter-annotator agreement, different thresholds for partial relevance.

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

This is the standard and correct way to evaluate against TREC qrels. It is meaningfully different from TrialGPT's evaluation, which ran their system on the full corpus top-500 — a procedure that requires running the full pipeline on 374k documents per topic.

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

---

## 7. Experiments and Results

**TODO — fill in from completed eval runs**

### 7a. Baseline (sim+svm+clf)

| Dataset | NDCG@10 | MRR | F1 | FPR | Latency (inference) |
|---|---|---|---|---|---|
| TREC 2021 + KZ (49 topics) | 0.6525 | 0.305 | 0.335 | — | ~3s/query |
| TREC 2021 + TREC 2022 + KZ (TBD) | TBD | TBD | TBD | TBD | TBD |

**Note on eval mode vs. inference mode efficiency.**
TREC evaluation requires ranking every judged document per topic to compute fair NDCG.
This collapses the cascade: `reset_filter_params(len(doc_set))` passes all judged docs
through every filter, so the KZ topics (1,100+ docs each) run the full BERT classifier
over the entire judged pool — taking ~45–60s per topic in eval.

In real inference mode the cascade runs as designed: sim cuts 374k → 10k, SVM → 100,
classifier → 50, giving ~3s end-to-end on a GPU. TREC does not measure this.

**Benchmark blind spot:** TREC scores NDCG with no time or cost dimension. A system that
runs GPT-4 on every candidate for 45 minutes per query scores on the same axis as one
that runs in 3 seconds. TrialGPT (NDCG@10=0.73 on full corpus top-500) falls into this
category — it uses GPT-4 with per-criterion chain-of-thought, which at $0.03/1k tokens
over 500 docs × ~50 criteria each is roughly $0.75/query, or ~$56 for a 75-topic eval.
Our pipeline runs the same eval on Colab free tier.

A fairer comparison table would add:
- Latency (seconds per query, GPU T4)
- Cost per query ($ at public API prices, or GPU-hours)
- NDCG / (cost per query) as an efficiency-adjusted metric

TODO: measure and record inference latency per pipeline configuration.

### 7b. Embedding ablation: MiniLM vs. MedCPT

| Model | FPR ↓ | MRR ↑ | F1 ↑ |
|---|---|---|---|
| MiniLM-L6-v2 (384-dim, baseline) | 5.987 | 0.524 | 0.688 |
| MedCPT (768-dim, biomedical) | 4.813 | 0.554 | 0.663 |

Surprising result: MedCPT improves MRR and FPR (better at finding the first relevant trial) but slightly hurts F1. Possible explanation: MedCPT's asymmetric encoders (Article vs. Query) are better calibrated for retrieval but the classifier downstream isn't adapted to the higher-dimensional embeddings.

### 7c. Filter ablation: sim+svm+clf vs. svm+clf

**TODO — pending results from svm+clf run**

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

### 7e. Comparison to TrialGPT

**Critical comparability note:** TrialGPT reports NDCG@10=0.7275 on the full corpus top-500, not the judged pool. Our eval uses the judged pool only (standard TREC evaluation). These are NOT directly comparable:
- Full corpus: the model ranks all 374k trials; NDCG is computed against qrels on those rankings
- Judged pool: the model ranks only the ~80–1200 judged trials; NDCG is computed on those

**TODO — if access to TrialGPT outputs can be obtained, run our model on the same full corpus top-500 for direct comparison**

---

## 8. Error Analysis

→ See `docs/error_analysis_baseline.md` for full analysis.

Key findings:
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

### 9b. RLHF / DPO on clinical judgments

The TREC qrels provide pairwise preference signal: rel=2 > rel=1 > rel=0. DPO (Direct Preference Optimization) can use this to fine-tune a smaller model without a separate reward model.

**Clinical advantage:** as an RN with 20 years ICU experience, the user can provide authoritative pairwise judgments on specific error cases that no annotation team could produce. These gold-standard corrections are high-value training signal.

### 9c. Lab value extraction and comparison

The ctproc `lab/` module (`extractor.py`, `patterns.py`, `reference_ranges.py`) extracts lab values from criteria text. A trial that requires "Hemoglobin ≥ 10 g/dL" contains a structured constraint that can be compared against a patient's known lab values.

**TODO — document the lab extraction pipeline in its own section (this deserves §3e)**

Current state: extractor exists but is not integrated into the ranking pipeline. Integrating it would allow hard filtering on lab thresholds — a patient with Hgb=8 would be automatically excluded from trials requiring Hgb≥10.

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
- [ ] Get TrialGPT outputs for direct corpus-level comparison
- [ ] Document TrialGPT criterion annotations: format, coverage, alignment with our criteria parser
