# Pipeline Error Analysis — BioLinkBERT-large, TREC21+KZ Combined

**Model:** BioLinkBERT-large cross-encoder, local Drive checkpoint  
**Eval:** TREC21 + KZ, 134 topics, sim→SVM→classifier cascade (eval mode, all judged docs through)  
**Overall:** NDCG@10=0.7528, MRR=0.8253  
**Predictions file:** `eval_predictions.jsonl` (1340 records)

---

## Confusion Matrix

The predicted label here is derived from **rank position** relative to the qrel label counts for each topic — not the classifier's raw output class. This makes the matrix a ranking quality measure: a doc ranked in the top-k positions gets predicted=relevant, next band gets partial, rest get not_relevant.

| actual \ pred | not_relevant | partially_relevant | relevant |
|---|---|---|---|
| **not_relevant** (364) | 66 (4.9%) | 99 (7.4%) | **199 (14.9%)** |
| **partially_relevant** (192) | 13 (1.0%) | 73 (5.4%) | **106 (7.9%)** |
| **relevant** (784) | 4 (0.3%) | **18 (1.3%)** | 762 (56.9%) |

**Total errors: 439 / 1340 (32.8%)**

Error distribution by cell:
- `actual=not_rel, pred=relevant`: 199 (45.3% of errors) — false positives crowding top positions
- `actual=partial, pred=relevant`: 106 (24.1% of errors) — partial docs elevated to top band
- `actual=not_rel, pred=partial`: 99 (22.6% of errors) — non-relevant docs in middle band
- `actual=relevant, pred=partial`: 18 (4.1%) — relevant docs missed from top band
- `actual=partial, pred=not_rel`: 13 (3.0%) — partial docs ranked too low
- `actual=relevant, pred=not_rel`: 4 (0.9%) — worst misses: relevant docs at bottom

---

## Error Cause Breakdown

Estimated across all 439 errors after reading representative samples:

- **Diagnosis specificity failure** (model matches domain/organ, not specific condition): ~48%
- **Reasonable disagreement** (partial vs relevant label is genuinely debatable): ~22%
- **Age/demographic exclusion not recognized**: ~8%
- **Short/vague criteria over-ranked** (trial says "any lung disease", model obliges): ~7%
- **Data error / incomplete qrel** (no positive docs in judged pool, or label appears wrong): ~5%
- **Deep model error** (patient meets criteria but model misses subtle threshold): ~7%
- **Systemic / retrieval gap** (relevant trial never in judged pool): ~3%

---

## Error Analysis by Cell

### Cell: actual=not_relevant, pred=relevant (199 errors, 45.3%)

These are the highest-impact errors for NDCG: non-relevant trials occupy the top-k slots, pushing
relevant trials down or out of the top 10.

#### Individual error analysis — key clusters

**Cluster A: Diagnosis specificity failure (~120/199 errors)**

The model conflates organ-system or symptom similarity with trial eligibility. Examples:

| # | Topic summary | Trial ranked high | Why it's wrong | Cause |
|---|---|---|---|---|
| 1 | 2yo boy, itchy rash since infancy (atopic dermatitis) | Kawasaki disease trial (ranked 1–4) | Patient has eczema, NOT KD; no fever, no mucosal involvement | Model error — shallow: "child + rash" vector matches KD |
| 2 | 47yo smoker, COPD presentation | Pulmonary fibrosis trial (ranked 1–2) | Patient likely has COPD/emphysema, not IPF; smoking history is shared but diagnosis diverges | Model error — shallow |
| 3 | 47yo smoker, COPD | Asbestosis/occupational trial | No asbestos exposure mentioned; construction history ≠ asbestos exposure | Model error — shallow |
| 4 | 27yo woman, shin papules + oral ulcers (likely Behçet's) | Generic dermatology skin biopsy trials (ranked 2–5) | Trials are for skin lesion biopsy of suspicious malignancies; Behçet's is an autoimmune vasculitis | Model error — shallow |
| 5 | 41yo acromegaly patient post-surgery | Acromegaly trial (ranked 2, NOT error), but generic endocrine observation trials | Broad "endocrine disease" criteria vacuuming up a specific condition patient | Model error — shallow |
| 6 | 47yo woman, dizziness + tinnitus (Menière's?) | Hypothyroid patient trial (ranked 10) | Completely different condition; "dizziness" in topic matched thyroid trial on some feature | Model error — shallow |

**Representative topic-level breakdown:**

Topic 18 (atopic dermatitis, 2yo): All 10 judged trials are non-relevant and NDCG=0.
The pipeline retrieved Kawasaki disease trials for an eczema patient — a retrieval/classification failure.
The true relevant trials (atopic dermatitis treatment) were never in the judged pool, making this topic
a qrel completeness issue as much as a model error.

Topic 12 (COPD, 47yo smoker): The one relevant trial (COPD GOLD staging study) is ranked 8th.
Trials ranked 1–7 are for pulmonary fibrosis, asbestosis, and generic "lung disease" — related organ
system, wrong diagnosis. NDCG@10=0.393.

Topic 43 (skin rash + oral ulcers): Behçet's disease-like presentation. The correct relevant trial
("skin lesions + consent") is ranked 1st correctly. But non-relevant generic skin biopsy and skin
problem registry trials fill positions 2–5. NDCG@10=0.389, despite the top rank being correct —
demonstrates how non-relevant docs in positions 2–5 dilute NDCG.

**Cluster B: Age/demographic exclusion not recognized (~28/199 errors)**

The model ranks trials where the patient would be explicitly excluded by age criteria.

| # | Topic | Trial age criterion | Patient age | Cause |
|---|---|---|---|---|
| 1 | 31yo woman with joint pain | "Exclusion: patients less than 50 years old" | 31yo — explicit exclusion | Model error — deep (missed exclusion) |
| 2 | 15-week-old infant | "Man or woman older than 18 years" | Infant — excluded | Model error — shallow (missed age) |
| 3 | 4yo boy, wheezing | "Allergic asthma age 12+" | 4yo — excluded | Model error — shallow |
| 4 | 16yo girl, myasthenia gravis | "children < 21 years" — this one is RELEVANT, but listed as a false positive, suggesting the annotator labeled it not_relevant while the trial does admit < 21yo | Data quality question |

**Cluster C: Short/vague criteria over-ranked (~18/199 errors)**

Statistical finding: FP docs average **284 chars** vs 428 chars for correctly-predicted not_relevant docs (p<0.0001). Trials with only a sentence of eligibility text ("Inclusion: any lung disease. Exclusion: none") score high because the model sees no disqualifying criteria. These are observational registries.

Examples: "Inclusion Criteria: Patients with cough or shortness of breath" (for COPD topic), "Inclusion Criteria: Adults with pulmonary fibrosis" (similarly broad). The model can't penalize what's not written.

#### Statistical patterns (contrastive: FP errors vs correctly-predicted not_relevant)

| Pattern | FP rate | Correct not_rel rate | Enrichment | p-value |
|---|---|---|---|---|
| Short doc (< 300 chars) | 48% | 31% | 1.55x | <0.001 |
| High word overlap with topic | 0.20 mean | 0.10 mean | 2.0x | <0.0001 |
| Has age restriction in text | 14.1% | 28.8% | 0.49x | 0.009 |
| Broad criteria (generic consent + short) | 9.0% | 1.5% | 6.0x | 0.05 |

Key: FP errors have **more word overlap** (model is fooled by lexical similarity) but **fewer age restrictions** (correctly predicted not_rel docs often have explicit age criteria the model uses as signal). When age criteria are absent, the model defaults toward relevant.

#### Stage attribution

~90% are **classifier failure**: the docs reached the classifier through the sim+SVM cascade and were misranked. These are not retrieval gaps — the docs are in the judged pool, meaning they were retrieved originally.
~10% may be **ranking calibration**: partial and not_relevant docs ranked in the top slot when P(relevant) scores are similar.

---

### Cell: actual=partially_relevant, pred=relevant (106 errors, 24.1%)

These contribute partial NDCG damage: a partial doc in position k contributes gain=1 instead of gain=2,
reducing DCG by 1/log2(k+1) per slot.

#### Individual error analysis

Most of these are **reasonable disagreement** between model and annotator. Examples:

| # | Topic | Trial | Annotation | Reasoning |
|---|---|---|---|---|
| 1 | 8yo boy, muscular dystrophy | "All patients with diagnosis of muscular dystrophy" | Partial | Model is correct that this matches; annotator gave partial possibly because age range not confirmed or it's observational |
| 2 | 39yo man, acute gout | "Acute gout by ACR/EULAR criteria" | Partial | Strong match — patient has acute gout. Partial label may reflect uncertainty about whether ACR criteria explicitly met |
| 3 | 47yo COPD patient | COPD registry: "age >35, 30+ pack years, obstructive spirometry" | Partial | Patient meets 2/3 criteria explicitly; spirometry not confirmed in topic |
| 4 | 17yo elbow pain | Lateral epicondylitis trial | Partial | Patient has elbow pain consistent with LTE; annotator partial because age 17 may be borderline |
| 5 | 23yo man, syncope | "Convulsive or non-convulsive syncopes 18+ years" | Partial | Clear match — patient has syncope. Unclear why partial |

**Estimated cause distribution for this cell:**
- Reasonable disagreement (model arguably correct): ~50%
- Model over-estimates relevance (partial trial, patient excluded by unstated criterion): ~30%
- Borderline age/eligibility: ~20%

These 106 errors are the **least actionable** cluster: half appear to be annotator conservatism rather than model failures.

---

### Cell: actual=relevant, pred=partial (18 errors, 4.1%) and pred=not_relevant (4 errors, 0.9%)

**22 total false negatives** — the most NDCG-critical errors since they represent missed matches.
The good news: the model correctly ranks relevant docs first in 762/784 cases (97.2% recall in top band).

#### Individual error analysis — all 22 FN errors examined

**Pattern A: Correct match loses to wrong-diagnosis competitor (~9/22)**

The relevant trial has specific eligibility text that matches, but non-relevant trials with higher
surface similarity fill the top positions first. Examples:

- **Topic 201412** (25yo woman, hypothyroidism): Relevant trial = "Premenopausal women with overt primary hypothyroidism who did not receive thyroid hormones." An exact match. But ranked 9th (pred=partial) because 8 other thyroid-adjacent or general-medicine trials ranked above it.
- **Topic 20144** (Kawasaki disease, 2yo boy): Relevant trial = "Kawasaki Disease Presentation." Ranked 8th (pred=not_relevant). The doc text is mostly exclusion criteria ("Laboratory Any laboratory toxicity...") — key inclusion criteria text may be truncated or sparse, giving the classifier little signal.
- **Topic 201424** (33yo, abdominal trauma with splenic injury): Relevant trial = "Need for splenectomy." Only 3 words of inclusion criteria. Ranked 8th. The classifier can't score this highly when the criteria are so vague.

**Pattern B: Vague/generic relevant trial criteria (~8/22)**

The true-positive trial has minimal eligibility text, which makes it score no better than the non-relevant
trials with similarly generic criteria. Examples:

- Topic 201420 (multiple fractures, car accident): Relevant trial = "Road traffic accident victims hospitalized in surgical department less than 2 weeks." Ranked 10th. Outcompeted by more verbose trials about trauma protocols.
- Topic 201427 (21yo, colonic adenomatosis / FAP): Relevant trial = "Adult patients undergoing colonoscopy for screening or surveillance." Ranked 10th. A registry-style trial for colonoscopy surveillance, but the patient needs FAP-specific management.
- Topic 201429 (51yo woman, osteoporosis in menopause): Relevant trial = "age ≥20, able to fill questionnaire." Ranked 10th. Extremely generic.

**Pattern C: Borderline cases / annotator calls (~5/22)**

- Topic 201527 (15yo girl, fatigue + school absences): Ranked 3rd and 4th. Trials include one for "persistent yellow skin/sclera" (biliary obstruction) and one for "innocent heart murmurs" — clearly not the intended match. The correct diagnosis seems to be iron deficiency anemia or hypothyroidism. The annotated "relevant" trial may itself be questionable.

#### Stage attribution

- **Classifier reranking failure**: ~75% — trial retrieved and in pool, but outcompeted by wrong-diagnosis docs with higher P(relevant) scores
- **Sparse criteria / doc truncation**: ~20% — relevant trial has so little eligibility text that the classifier can't distinguish it from noise
- **Genuine ambiguity**: ~5%

---

## Data Quality Issues

1. **Topic 18 (atopic dermatitis)**: All 10 judged trials are Kawasaki disease trials or generic pediatric trials — none is for atopic dermatitis. The correct trial type (eczema treatment) was never retrieved in the original TREC pooling and therefore never judged. This topic contributes NDCG=0 not because the model failed but because the judged pool is incomplete. The model's error (ranking KD trials high for an eczema patient) is real, but the NDCG=0 penalty is inflated.

2. **Topic 201528 (Lyme arthritis, 8yo)**: All 10 judged docs labeled not_relevant despite several Lyme disease trials in the pool (tick-borne illness screening, PTLDS evaluation). The presentation (knee swelling + fever post-tick bite = Lyme arthritis) should match some of these. Possible explanation: trials require confirmed serological Lyme diagnosis, which the topic doesn't confirm. Annotators may have been strict about the unconfirmed diagnosis. Questionable but defensible.

3. **Topic 43, rank 1**: Trial says "Candidate with skin lesions / signed consent" with exclusion "prior surgery or radiotherapy." This is labeled **relevant** for a Behçet's-like patient with shin papules + oral ulcers — arguably reasonable since the patient has skin lesions, but the trial looks like a generic dermatology observation study. Borderline data quality.

4. **~15 partial labels** that look like they should be not_relevant (e.g., topic 34: elbow pain patient matched to generic "patients with lateral epicondylitis" trials — the patient's elbow diagnosis is not confirmed, so partial is defensible, but some look like forced partial labels for completeness).

---

## Priority Fixes

### 1. Hard-negative mining for diagnosis specificity (affects ~120 FP errors)

**Problem:** Model conflates organ-system similarity with diagnostic eligibility. A Kawasaki trial ranks high for an eczema patient; a pulmonary fibrosis trial ranks high for a COPD patient.

**Fix:** Mine contrastive pairs from the error analysis: for each FP error where trial T is ranked high for topic Q but actual label = 0, add (Q, T_correct, T_wrong) as a DPO training example. Specifically target:
- Eczema topic → Kawasaki trial as hard negative, atopic dermatitis trial as positive
- COPD topic → pulmonary fibrosis / asbestosis trials as hard negatives, COPD trial as positive
- Skin rash+ulcers → skin biopsy/lesion trials as hard negatives

This directly feeds into PLAN.md §4 (DPO). Expected impact: reduce FP errors by ~30–40%, +0.05–0.08 NDCG.

**Verify:** Re-run eval_baseline after DPO fine-tuning; check that topics 12, 18, 43 NDCG improve.

### 2. Demographic pre-filter — age extraction (affects ~28 FP errors)

**Problem:** "Exclusion: patients less than 50 years old" not recognized for a 31yo patient; infant matched to adult-only trials.

**Fix:** Extend the lab filter (PLAN.md §2) to also extract patient age from topic text and trial age range from inclusion/exclusion criteria. Hard-exclude trials where patient age is outside the stated range.

Implementation sketch:
```python
import re

def extract_patient_age(topic_text: str) -> int | None:
    m = re.search(r'\b(\d+)[\s-]*(year|yo|y\.o)', topic_text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def trial_excludes_age(doc_text: str, patient_age: int) -> bool:
    # Check "less than X years" / "older than X years" exclusion patterns
    # Check "age X to Y" inclusion range
    ...
```

Expected impact: ~28 FP errors eliminated, FPR ↓. Small NDCG gain since these are FP docs in top-10.

**Verify:** Run filter on the 28 identified topics; check precision of the regex against trial criteria text.

### 3. Down-rank short/vague criteria docs (affects ~18 FP errors)

**Problem:** Trials with <100 chars of eligibility text ("Inclusion: any lung disease") get ranked high because nothing disqualifies the patient. These are registries and observational studies with non-specific criteria.

**Fix (simple):** Add a soft penalty to classifier scores for docs with very short eligibility text. Or add `len(doc_text)` as a feature in a re-scoring step.

**Fix (better):** In fine-tuning, add these registry trials as hard negatives when a more specific matching trial exists.

**Verify:** Check what fraction of short-doc trials are genuinely relevant for any topic — don't over-penalize if some short trials are truly appropriate.

### 4. Classifier fine-tuning on partial→not_relevant boundary (affects ~90 FP errors in the not_rel→partial cell)

**Problem:** 99 not_relevant docs are ranked in the "partial" band. For NDCG this is less critical than the not_rel→relevant errors, but it still hurts.

**Fix:** These overlap with the diagnostic specificity issue. Addressed by the same hard-negative DPO fix.

---

## Summary Statistics

| Metric | Value |
|---|---|
| Total records | 1340 |
| Unique topics | 134 |
| Overall error rate | 32.8% (439/1340) |
| True relevant recall (in top band) | 97.2% (762/784) |
| FP rate (not_rel ranked relevant) | 54.7% of not_rel docs (199/364) |
| Topics with NDCG@10 ≥ 0.9 | 96/134 (72%) |
| Topics with NDCG@10 < 0.5 | 8/134 (6%) |
| Topics with no positive docs in pool | 2/134 (topics 18, 201528) |

The model is strong at finding truly relevant trials (97.2% recall in top band). The NDCG deficit comes almost entirely from non-relevant and partially-relevant trials **crowding the top positions** for the remaining 28% of topics — a precision problem, not a recall problem.
