# Baseline Error Analysis — ctmatch Pipeline

**Date:** 2026-06-26  
**Dataset:** eval_predictions.jsonl (TREC 2021 topics)  
**Pipeline:** sim filter (MiniLM + bart-large-mnli category) → SVM → SciBERT classifier  
**Baseline metrics:** NDCG@10=0.6525, MRR=0.305, F1=0.335  
**490 examples, 49 topics, 10 docs per topic**

---

## Confusion Matrix

```
                     not_relevant   partially_rel   relevant
not_relevant               46              85           127
partially_relevant          2              49            92
relevant                    2              18            69

490 examples | 326 errors (66.5%) | 49 topics
```

---

## Error Cause Breakdown

| Cause | Count | % of errors |
|-------|-------|------------|
| Shallow model error (category match, wrong condition) | ~155 | 47.5% |
| Reasonable disagreement (1 vs 2 boundary) | ~60 | 18.4% |
| Likely data errors (model arguably correct) | ~55 | 16.9% |
| Deep model error (missed clinical constraint) | ~40 | 12.3% |
| Systemic retrieval failure (unrelated condition) | ~16 | 4.9% |

---

## Error Analysis by Cell

---

### Cell 1: actual=not_relevant → predicted=relevant (127 errors, 39.0% of all errors)

#### Individual error analysis

| # | Topic condition | Trial condition | Cause | Reasoning |
|---|----------------|----------------|-------|-----------|
| 1 | CMV infection on immunosuppression | CMV-specific T-cell therapy | **Possible data error** | Patient has CMV + immunosuppression — the trial's CMV T-cell therapy is plausibly relevant. Annotator may have applied stricter criteria. |
| 2 | Lung mass + brain mass (lung ca) | Schizophrenia (risperidone/aripiprazole trial) | **Systemic retrieval failure** | Completely unrelated — neurological terms in both texts caused retrieval hit but there is zero clinical connection. |
| 3 | Cardiac chest pain (ACS presentation) | SVT cardioversion | **Shallow** | Both cardiac, but ACS ≠ SVT. Category filter (cardiac) passed the trial; classifier didn't distinguish arrhythmia from ischemia. |
| 4 | Endometriosis/menstrual pain | PMS (premenstrual syndrome) trial | **Shallow** | Both women's health, but PMS ≠ endometriosis. Category overlap drove retrieval. |
| 5 | CJD (rapidly progressive dementia) | Cerebral palsy spasticity, ages 2–17 | **Systemic retrieval failure** | Different condition, wrong age group entirely. "Neurological" category caused retrieval; the trial should have been hard-filtered on age. |
| 6 | Bipolar + depression + insomnia (age 26) | Nonorganic insomnia, ages 12–18 | **Deep** | Patient does have insomnia; classifier should have caught the age exclusion (trial max age 18, patient is 26). Age value comparison required. |
| 7 | Bipolar + depression | Major depressive disorder trial | **Possible data error** | Patient explicitly has depression as a comorbidity. The trial is for MDD. Annotator labeled not_relevant but this is debatable — patient meets the primary criterion. |
| 8 | Kawasaki disease (age 2) | Kawasaki disease (ages 4–16) | **Deep** | Condition match is perfect. The only exclusion is age — patient is 2, trial requires ≥4. Model couldn't compare numeric ages. |
| 9 | Lung mass + brain mass | Emphysema/COPD (lung disease) | **Shallow** | Both involve lungs and dyspnea, but malignancy ≠ emphysema. Pulmonary category drove retrieval. |
| 10 | PE post-hip replacement | Small cell lung cancer chemotherapy | **Systemic retrieval failure** | Completely different — PE vs SCLC. Shared "chest pain + dyspnea" vocabulary caused retrieval. |

#### Statistical patterns

| Feature | Error% | Correct% | Enrichment | p |
|---------|--------|----------|-----------|---|
| has_cardiac (topic) | 22.8% | 2.2% | **10.5x** | 0.001 |
| has_exclusion_lang (topic) | 55.9% | 21.7% | 2.6x | 0.0001 |
| has_gender (topic) | 97.6% | 67.4% | 1.45x | 0.0000 |
| has_lab (doc) | 1.6% | 8.7% | 0.18x | 0.025 |

The cardiac signal is the strongest finding in the dataset. Cardiac topics generate false positives at 10.5× the base rate — the sim filter's category match brings in every cardiac trial regardless of specific condition, and the SciBERT classifier doesn't penalize wrong cardiac subtypes strongly enough. Topics with exclusion language ("no X, not Y") also generate more FPs — the negation isn't being parsed. Trials with explicit lab values in their criteria (has_lab) are less likely to be FPs, suggesting that concrete numeric criteria help the classifier discriminate.

#### Stage attribution

**Primarily sim filter + classifier.** The category-level retrieval (cardiac → all cardiac trials) is correct behavior for the sim filter, but the classifier should be penalizing wrong conditions within a category. These errors arrive at the classifier ranked 1–10 (high cosine similarity), so the classifier is the last opportunity to demote them and it isn't.

---

### Cell 2: actual=not_relevant → predicted=partially_relevant (85 errors, 26.1%)

#### Individual error analysis

| # | Topic | Trial | Cause | Reasoning |
|---|-------|-------|-------|-----------|
| 1 | Child with sleep apnea + ADHD-like symptoms | ADHD study in children | **Shallow** | Symptom overlap (inattention, sleepiness), but sleep apnea causes those symptoms — the underlying condition differs. |
| 2 | Femoral artery pseudoaneurysm post-cath | Pulmonary arterial hypertension | **Systemic** | Completely different — femoral vascular complication vs PAH. Both involve "cardiac catheterization" lexically. |
| 3 | Hypothyroidism symptoms | Age-related macular degeneration | **Systemic** | Totally unrelated. No plausible mechanism. Pure vocabulary noise from shared patient demographic terms. |
| 4 | CMV infection on prednisone | Allogeneic stem cell transplant (CMV seropositive req.) | **Possible data error** | Patient has CMV and is immunosuppressed. Trial requires CMV seropositive + stem cell transplant history — patient doesn't have a transplant, so this is borderline. Annotator was probably correct. |
| 5 | CJD/prion disease | Alzheimer's diagnosis study | **Shallow** | Both dementias, but different types. Phenotypic overlap (cognitive decline) drove retrieval; CJD is a prion disease, not AD. |
| 6 | Kawasaki disease (age 2) | Rheumatoid arthritis (age ≥18) | **Deep** | Both autoimmune/inflammatory, but KD ≠ RA, and age exclusion (18+) should rule it out immediately. |

#### Statistical patterns

| Feature | Error% | Correct% | Enrichment | p |
|---------|--------|----------|-----------|---|
| unique_words (doc, continuous) | 53.5 | 45.5 | 1.18x | 0.007 |
| char_length (doc, continuous) | 450 | 391 | 1.15x | 0.037 |
| has_cardiac (topic) | 10.6% | 2.2% | 4.9x | 0.098 |

Longer, richer trial documents generate more partial-relevance FPs. The classifier appears to interpret document richness as a signal of relevance — a detailed eligibility criteria text with many unique terms gets scored up regardless of whether those terms actually match the patient. This is a known failure mode of sequence classifiers trained on pair-level classification: token count acts as a proxy for relevance.

#### Stage attribution

**Primarily classifier.** These trials passed the sim and SVM filters (plausible category match), then the classifier assigned partial scores based on surface richness rather than condition alignment.

---

### Cell 3: actual=partially_relevant → predicted=relevant (92 errors, 28.2%)

This is the most important cell to scrutinize for data quality because the 1→2 boundary is where annotator inconsistency is worst.

#### Individual error analysis

| # | Topic | Trial | Cause | Reasoning |
|---|-------|-------|-------|-----------|
| 1 | Diabetic with leg skin ulcer | Diabetic foot wound infection trial | **Likely data error** | Patient has diabetes + skin lesion on leg. Trial is explicitly for diabetic foot wounds with inflammation. This seems clearly relevant, not partial. Model is probably right. |
| 2 | Progressive dysphagia (esophageal cancer) | Esophageal cancer neoadjuvant chemo | **Likely data error** | Direct condition match. Annotator labeled partial — possibly because metastatic disease was listed as exclusion and the patient's staging is unclear, but the model's "relevant" call is defensible. |
| 3 | Trauma patient, extremity fractures | Post-abdominal surgery complications | **Reasonable disagreement** | Related surgical/trauma context but patient has extremity fractures, not abdominal surgery. Partial label makes sense. |
| 4 | Hyperthyroidism (Graves' disease) | Thyroidectomy for Basedow's disease (= Graves') | **Likely data error** | Basedow's disease IS Graves' disease. Direct match. Annotated as partial — possibly because surgical candidacy is uncertain, but model's "relevant" call is correct. |
| 5 | HPV+, cytology negative (32yo woman) | Trial for HPV+, cytology-negative women | **Likely data error** | Criteria match exactly: age 30–65, TCT negative, HPV positive. This should be labeled 2 (relevant), not 1. |
| 6 | Lung mass + brain mass | ARDS in ICU patients | **Shallow model error** | Patient has lung cancer + brain met, not ARDS. Both involve respiratory distress but completely different. Model over-ranked. |
| 7 | Endometriosis, pelvic pain | Uterine fibroids trial | **Reasonable disagreement** | Overlapping presentation (pelvic pain, menstrual issues) but endometriosis ≠ fibroids. Partial label is defensible. |
| 8 | Enlarged uterus, cervical dilation | Uterine leiomyomas (fibroids) trial | **Reasonable disagreement** | Overlapping gynecological pathology but the acute presentation (possible incomplete abortion) differs from fibroid management. |

#### Statistical patterns

| Feature | Error% | Correct% | Enrichment | p |
|---------|--------|----------|-----------|---|
| has_cancer (doc) | 4.3% | 14.3% | **0.30x** | 0.037 |
| has_gender (doc) | 16.3% | 6.1% | 2.66x | 0.086 |
| word_count (doc) | 68.2 | 58.5 | 1.17x | 0.007 |
| has_cardiac (topic) | 26.1% | 12.2% | 2.13x | 0.083 |

Cancer trials are under-represented in these over-ranking errors (0.30x). The classifier is better at not promoting cancer-specific trials into the top slot when they're only partial matches — likely because cancer terminology is distinctive enough. The over-ranking errors cluster around vaguer shared conditions (cardiac, gynecological) where the 1 vs 2 boundary is genuinely ambiguous.

~4 of the 8 sampled examples appear to be data errors where the annotator under-scored the trial. If this rate holds across all 92, ~46 of these "errors" are actually correct model predictions with wrong ground truth.

#### Stage attribution

**Mixed: data quality + classifier.** The classifier is not reliably learning the distinction between 1 and 2 because (a) the training data itself is noisy on this boundary, and (b) nothing in the current architecture reasons about whether the patient actually *meets* each criterion.

---

### Cell 4: actual=relevant → predicted=partially_relevant (18 errors, 5.5%)

The smallest cell but highest clinical cost — these are relevant trials the pipeline under-ranks.

#### Individual error analysis

| # | Topic | Trial (rank) | Cause | Reasoning |
|---|-------|-------------|-------|-----------|
| 1 | 89yo man, Alzheimer's symptoms | "Alzheimer's disease" trial (rank 10) | **Deep** | Minimal criteria ("Alzheimer's disease, no other neurologic disease"). Short doc — classifier had little text to score. Relevant trial buried at rank 10. |
| 2 | Endometriosis, pelvic pain | Endometriosis diagnosis study (rank 8) | **Deep** | Direct match ("diagnosis of endometriosis"). Very short eligibility criteria. Relevant trial under-ranked because sparse text yields low classifier confidence. |
| 3 | Fibroid symptoms (enlarged uterus) | Fibroid symptoms trial (rank 5) | **Reasonable disagreement** | Moderate under-ranking (rank 5). Short criteria. |
| 4 | Skull fracture → bacterial meningitis | Bacterial meningitis criteria trial (rank 9) | **Deep** | Trial criteria describe meningitis signs that exactly match the patient. Ranked 9. Short, structured criteria underscored by classifier. |
| 5 | Rabies exposure (animal recovery) | Any volunteer trial (catch-all) | **Possible data error** | Trial accepts any medically fit volunteer. Labeled "relevant" — but this label applies to any patient ever. Likely an annotation artifact. |
| 6 | Skin tags (likely no biopsy needed) | Trial for patients needing biopsy | **Possible data error** | Exclusion criteria explicitly say "without biopsy" patients are excluded. Patient has skin tags that typically don't need biopsy. Labeled relevant but model's "partial" may be more accurate. |
| 7 | Child with fever + cough + Colorado trip | Chest imaging in ED respiratory patients (rank 10) | **Deep** | Direct match (ED, respiratory symptoms, CXR). Short eligibility criteria. Relevant trial at rank 10. |
| 8 | CJD dementia | Mild cognitive impairment study (rank 10) | **Reasonable disagreement** | CJD overlaps with MCI presentation at early stages but is distinct. Partial label seems more appropriate; "relevant" label may be over-generous. |

#### Statistical patterns

| Feature | Error% | Correct% | Enrichment | p |
|---------|--------|----------|-----------|---|
| has_age (doc) | 27.8% | 58.0% | **0.48x** | 0.023 |
| has_exclusion_lang (topic) | 22.2% | 63.8% | **0.35x** | 0.003 |
| char_length (doc) | 340 | 422 | 0.81x | 0.038 |

The clearest pattern in the dataset: short trials with no explicit age criteria and minimal exclusion language get under-ranked. The classifier needs rich text to output a high "relevant" score. Trials like "Alzheimer's disease, no other neurologic disease" (6 words of criteria) score lower than longer trials with unrelated content, simply because the SciBERT pair classifier rewards token richness.

#### Stage attribution

**Classifier.** These trials have already survived retrieval and SVM — they're in the judged set. The classifier systematically under-scores sparse docs regardless of the condition match quality.

---

## Data Quality Issues

Examples where the ground truth label appears incorrect:

| Topic | Trial | Labeled | Model | Issue |
|-------|-------|---------|-------|-------|
| 20147 | Major depressive disorder trial | not_relevant | relevant | Patient has depression — primary criterion met |
| 20144 | Kawasaki disease trial (ages 4–16, patient age 2) | not_relevant | relevant | Condition matches; age exclusion is the only issue — may deserve partial |
| 20146 | Diabetic foot wound infection | partially_relevant | relevant | Diabetes + leg wound — direct match to trial criteria |
| 201419 | Esophageal cancer neoadjuvant chemo | partially_relevant | relevant | Dysphagia patient with likely esophageal cancer — condition match |
| 20156 | Thyroidectomy for Basedow's/Graves' | partially_relevant | relevant | Graves' disease = Basedow's disease — exact condition match |
| 201517 | HPV+, cytology-negative trial | partially_relevant | relevant | Patient criteria match trial criteria exactly |
| 201416 | Any volunteer catch-all trial | relevant | partially_relevant | "Any medically fit volunteer" should be partial at best |
| 20149 | Skin biopsy trial (exclusion: no-biopsy patients) | relevant | partially_relevant | Patient's skin tags likely don't require biopsy — exclusion applies |

Estimated ~15–20% of errors are label quality issues rather than model failures.

---

## Priority Fixes

**1. Criterion-level condition matching — affects ~155 errors (cells 1+2)**

The sim filter correctly retrieves trials in the right medical category, but nothing in the pipeline checks whether the *specific condition* matches. A patient with ACS gets every cardiac trial; a patient with CJD gets every neurological trial. The fix is Phase 2 of the research plan: per-criterion entailment scoring using ctproc's parsed inclusion/exclusion criteria. Expected to cut cell 1 errors roughly in half.

**2. Numeric hard-filtering on age — affects ~20–30 errors across cells 1+2**

Multiple clear errors involve age exclusions (trial max 18, patient 26; trial min 4, patient 2). ctproc already extracts `elig_min_age`/`elig_max_age` from trials. Applying a hard age filter in the pipeline before the sim filter would eliminate these at zero model cost.

**3. Short-document handling in classifier — affects all 18 cell-4 errors**

The classifier systematically under-scores trials with sparse eligibility criteria text. Options: (a) weight confidence by inverse document length, (b) augment short trials with structured fields (condition, age range) during classification, (c) use the ctproc structured output (`include_criteria` list) rather than raw concatenated text as classifier input.

**4. Negative hard sampling in SciBERT retraining — affects cell 1**

Training data likely underrepresents same-category negatives (cardiac trial for non-cardiac condition). Adding hard negatives from the same medical category during fine-tuning would teach the classifier to distinguish condition subtypes within a category.

**5. Relabel ~8 boundary examples before next training run**

The partial→relevant errors include at least 4 clear label errors (Graves', HPV, diabetic foot wound, esophageal cancer). Fixing these changes the classifier's training signal on exactly the 1 vs 2 boundary where performance is weakest.

---

## Summary

The pipeline has reasonable recall (only 2 truly relevant docs dropped entirely) but poor precision at the top. The dominant failure mode is not retrieval — it's the classifier treating same-category but wrong-condition trials as relevant. The fix is not a better embedding model; it's condition-level reasoning, which is exactly Phase 2 of the plan. The NDCG/MRR discrepancy (0.65 vs 0.31) is also consistent with this: the pipeline can retrieve relevant docs but doesn't rank them first, because it can't distinguish specific conditions within a medical category.
