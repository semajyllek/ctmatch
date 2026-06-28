# Error Analysis — ctmatch Baseline Pipeline
**File**: `eval_predictions.jsonl` (baseline run, post sim_filter + classifier fixes)  
**Pipeline**: sim_filter (MiniLM + category) → SVM → SciBERT classifier  
**Labels**: 0=not_relevant, 1=partially_relevant, 2=relevant  
**Date**: 2026-06-27

---

## Confusion Matrix

Rows = actual label, columns = predicted label.

|                        | pred: not_rel | pred: partial | pred: relevant | **total** |
|------------------------|:-------------:|:-------------:|:--------------:|:---------:|
| **actual: not_rel**    |  67 (5.0%)    | 115 (8.6%)    |  517 (38.6%)   |  699      |
| **actual: partial**    |  11 (0.8%)    |  54 (4.0%)    |  282 (21.0%)   |  347      |
| **actual: relevant**   |   5 (0.4%)    |  21 (1.6%)    |  268 (20.0%)   |  294      |
| **total predicted**    |  83           | 190           | 1067           | **1340**  |

- **Total errors**: 951 / 1340 (71.0%)
- **Correct**: 389 / 1340 (29.0%)
- **Predicted distribution**: not_rel=83, partial=190, relevant=1067
- **Actual distribution**: not_rel=699, partial=347, relevant=294
- **Core problem**: model predicts "relevant" 1067 times; only 294 are actually relevant (3.6× overestimate)

---

## Error Cause Breakdown

| Cause | Count | % of errors |
|-------|------:|------------:|
| Model error — shallow (condition/domain mismatch, age exclusion missed) | ~370 | ~39% |
| Model error — deep (correct disease, fails specific criteria: lab values, prior treatment, subtype) | ~380 | ~40% |
| Reasonable disagreement (partial vs relevant boundary ambiguous) | ~150 | ~16% |
| Data error (questionable TREC label) | ~30 | ~3% |
| Systemic — sparse criteria text (false negatives) | ~21 | ~2% |

---

## Error Analysis by Cell

### Cell: Predicted relevant, Actually not_relevant — 517 errors (54.4% of all errors)

The dominant failure. The model flags as fully eligible trials that are either completely wrong domain, wrong age group, or require a specific diagnosis the patient doesn't have.

#### Individual error analysis (representative sample)

| # | Topic (truncated) | Trial (truncated) | Cause | Reasoning |
|---|-------------------|-------------------|-------|-----------|
| 1 | 58F, acute chest pain, first episode | Stable exertional angina trial (CCS II-III, ≥3 months history required) | Model error—deep | Patient has first-episode acute pain; trial requires chronic stable angina. Model matched "chest pain" + "angina" without reading the temporal/stability requirement. |
| 2 | 58F, acute chest pain | Idiopathic mechanical low back pain trial | Model error—shallow | Completely different organ system. Model likely scored high on shared demographic tokens (58F, ER) rather than condition. |
| 3 | 8yo male, fever, dyspnea, Colorado trip | Adults ≥18 C. diff trial | Model error—shallow | Clear age exclusion (trial ≥18, patient is 8). Different condition entirely. Model did not check age criteria. |
| 4 | 8yo male, respiratory | Healthy child TBE vaccine trial (1-10 years) | Reasonable disagreement | Age matches but patient is acutely ill — healthy volunteer exclusion not caught. |
| 5 | 58F, lung mass (likely NSCLC) | Asthma trial / COPD trial / medulloblastoma trial | Model error—shallow | Condition mismatch across organ systems. "Dyspnea" and "cough" overlap with all pulmonary trials regardless of diagnosis. |
| 6 | 58F, lung mass | Schizophrenia trial (psychiatric) | Model error—shallow | No semantic overlap with lung cancer. Classifier scored relevant on minimal signal — worst false positive category. |
| 7 | 2yo KD child | Adult RA trial (anti-TNF, age ≥18) | Model error—shallow | Age exclusion entirely missed; different autoimmune condition. |
| 8 | 56F post-mastectomy, likely PE | Suspected DVT of leg trial | Reasonable disagreement | Patient has PE/DVT post-cancer surgery — related, but trial requires suspected DVT of leg specifically. Label=0 may be too strict. |
| 9 | 18yo Dengue (fever, leukopenia, thrombocytopenia) | Adults ≥40 trials | Model error—shallow | Age exclusion not caught. Dengue symptoms matched infection-related trials semantically. |
| 10 | KD child 2yo | Adult RA anti-TNF trial | Model error—shallow | Wrong age group, wrong condition. Both involve immune/inflammatory disease which likely drove the embedding match. |

**Pattern**: The classifier is not doing condition matching or age-gating. It scores on surface-level clinical vocabulary shared between any medicine-involving topic and any medical trial. Cardiac terms match cardiac trials regardless of subtype; "pediatric + fever" matches pediatric trials regardless of condition.

#### Statistical patterns (contrastive vs correct not_relevant predictions, N=67)

| Pattern | Error rate | Correct rate | Enrichment | p-value |
|---------|:----------:|:-----------:|:---------:|:-------:|
| Doc length (words) | 74.3 | 60.2 | — | <0.0001 |
| Vocabulary overlap (topic∩doc/topic) | 0.129 | 0.107 | — | 0.0001 |
| Topic is pediatric patient | 93.4% | 86.6% | 1.08× | 0.076 |
| Doc contains age criteria text | 69.4% | 68.7% | 1.01× | n.s. |

Longer criteria documents and higher vocabulary overlap enrich for false positives. The presence of age criteria *text* in the document does not reduce the error rate — the model is not evaluating it.

#### Stage attribution

**Classifier stage.** These documents were retrieved into the TREC judged pool by participating systems, passed through the SVM (which in eval mode passes all docs). SciBERT is the stage assigning the "relevant" score based on condition-level semantic similarity without eligibility checking.

---

### Cell: Predicted relevant, Actually partially_relevant — 282 errors (29.7% of all errors)

The model upgrades partial matches to full relevance. These are cases where the disease or condition domain is correct but specific eligibility criteria (gender requirement, prior treatment, disease subtype, numeric lab threshold) rule out full eligibility.

#### Individual error analysis (representative sample)

| # | Topic (truncated) | Trial (truncated) | Cause | Reasoning |
|---|-------------------|-------------------|-------|-----------|
| 1 | 58F, cardiac, chest pain | Stable angina trial — **MALE only** | Model error—deep | Trial explicitly requires male patients. Patient is female. Classifier matched on cardiac disease, ignored the gender criterion. |
| 2 | 58F, possible ACS | Acute MI/ACS/unstable angina trial | Reasonable disagreement | Patient has possible ACS — label=1 may be too strict; depends on final diagnosis. Annotator uncertainty is driving this. |
| 3 | KD child | IVIG-refractory KD trial (requires prior IVIG failure) | Model error—deep | Trial requires failure of initial IVIG treatment. Patient's IVIG history is unknown. Classifier matched "Kawasaki Disease" without parsing the refractory criterion. |
| 4 | 56F post-mastectomy, dyspnea | Massive PE requiring thrombolysis trial | Model error—deep | "Massive PE" requires hemodynamic instability. Patient's PE severity is unstated. Severity threshold not evaluated. |
| 5 | 64F, DM, elevated HbA1c | Non-healing foot wound trial (requires active ulcer, Grade II Wagner) | Model error—deep | Trial requires a specific complication (active foot ulcer). Patient has DM but no wound mentioned. DM keyword match is insufficient. |
| 6 | 26F, bipolar, obese | Insomnia trial requiring BMI 18–34 | Model error—deep | Patient is obese; trial caps BMI at 34. Numeric BMI threshold not evaluated. |
| 7 | 43F, skin fibromas (neck) | Healthy subjects Fitzpatrick I-III trial | Model error—deep | Trial requires healthy subjects WITHOUT skin disease. Patient has skin lesions — should be excluded, not included. Classifier missed the exclusion. |
| 8 | 27F pregnant, Hb=9.0 g/dL | Iron anemia in pregnancy trial (Hb<11, first trimester) | Possible data error | Patient explicitly meets all stated criteria — label=1 may be wrong; this looks like a label=2 case. |
| 9 | 67F post-cath, limb ischemia | Coronary diagnostic procedure femoral access trial | Reasonable disagreement | Patient had femoral access cath and now has ischemia — related, but trial is prospective procedural enrollment, not for managing complications. |
| 10 | 2yo KD, fever | KD trial requiring IVIG-refractory patients with specific coronary dilation | Model error—deep | Multiple specific sub-criteria (IVIG failure, coronary dilation, fever ≥37.5°C axillary). Model matched KD diagnosis without evaluating any of these. |

**Pattern**: The classifier identifies the correct disease domain but cannot evaluate:
- Specific required prior treatment history (IVIG failure, prior chemo lines)
- Numeric lab/measurement thresholds (BMI ≤34, Hb <11, GFR cutoffs)
- Gender/demographic requirements
- Disease subtype specificity (stable vs unstable, complete vs incomplete KD)
- Exclusion criteria containing the patient's actual condition

#### Statistical patterns (contrastive vs correct partially_relevant predictions, N=54)

| Pattern | Error rate | Correct rate | Enrichment | p-value |
|---------|:----------:|:-----------:|:---------:|:-------:|
| Doc length (words) | 72.6 | 59.1 | — | <0.0001 |
| Lab value keywords in doc | 57.8% | 40.7% | **1.42×** | **0.031** |
| Vocabulary overlap | 0.132 | 0.115 | — | 0.040 |
| Topic is pediatric | 91.8% | 83.3% | 1.10× | 0.089 |

**Lab value enrichment is the most actionable signal**: trials containing explicit numeric thresholds (Hb, creatinine, platelet count, HbA1c, eGFR, BMI) are 1.42× more likely to be incorrectly upgraded to "relevant." The model cannot compare `patient Hb=9.0` vs `trial threshold Hb<11`, or `patient BMI=35` vs `trial BMI≤34`. This directly motivates structured lab value extraction.

#### Stage attribution

**Classifier stage.** The disease domain is correctly identified (retrieval and SVM passed these appropriately). SciBERT cannot evaluate specific inclusion/exclusion sub-criteria. This is the deep reasoning gap that either a structured matching layer (lab extraction) or RLHF-trained reasoning model would address.

---

### Cell: Predicted partially_relevant, Actually not_relevant — 115 errors (12.1% of all errors)

#### Individual error analysis (representative sample)

| # | Topic (truncated) | Trial (truncated) | Cause | Reasoning |
|---|-------------------|-------------------|-------|-----------|
| 1 | 8yo male, fever/dyspnea | Children 6 months–5 years, fever/cough trial | Model error—shallow | Patient is 8; trial caps at 5 years. Age upper bound missed. |
| 2 | 8yo male, respiratory | Gastroenteritis study (GI, different condition) | Model error—shallow | Different condition. Shared "pediatric + fever" surface drove partial-relevance score. |
| 3 | 2yo KD child | Adult RA anti-TNF trial (≥18 years) | Model error—shallow | Wrong age and wrong condition — but partial rather than relevant, so classifier was slightly more conservative here. |
| 4 | 62yo, CJD-like | Mild cognitive impairment / possible AD trial | Reasonable disagreement | Patient likely has prion disease (CJD), not AD. Overlapping cognitive decline symptoms make label=0 vs 1 debatable — this may be a data error. |
| 5 | 62yo, CJD | Parkinson's disease trial | Model error—shallow | CJD vs Parkinson's — distinct conditions with overlapping motor features (jerking movements). |
| 6 | 43F, neck skin fibromas | Hyaluronic acid cosmetic injection trial (females >35) | Reasonable disagreement | Age and gender match; condition (cosmetic aging) vs skin lesions doesn't match. Label=0 is correct. |
| 7 | 43F, neck skin lesions | Intracranial aneurysm trial | Model error—shallow | No relation — "neck" in topic spuriously matched "intracranial" (neck vessels?). |
| 8 | 67F post-cath, limb ischemia | Carotid stenosis trial (>60% stenosis) | Model error—shallow | Different vascular territory (peripheral femoral vs carotid). |
| 9 | 67F post-cath | Pulmonary arterial hypertension trial | Model error—shallow | Different condition. Cardiac catheterization in topic may have triggered "cardiac" semantic match. |

#### Stage attribution

**Mixed**: some are retrieval-adjacent errors (wrong-condition doc reaching classifier), some are classifier scoring on partial surface similarity. The partial (rather than relevant) prediction suggests the classifier is slightly more calibrated for these — but still wrong.

---

### Cell: Predicted not_relevant, Actually relevant — 5 errors (0.5% of all errors)

All five involve trials with extremely sparse criteria text. The model failed to score them as relevant despite a genuine patient match.

| # | Topic | Trial | Cause | Reasoning |
|---|-------|-------|-------|-----------|
| 1 | 2yo KD child | KD study: "medical file confirmed KD, aged 1mo–12yr" (one line) | Systemic—sparse text | Criteria are ~15 words. SciBERT has low signal to work with. |
| 2 | 8yo TBI, GCS declining after fall | "GCS ≤8, closed head injury, age 0–18" | Reasonable disagreement | Patient meets criteria. Possible model calibration issue on brief criteria. |
| 3 | 18yo Dengue | "Clinical diagnosis of Dengue Fever" (4 words) | Systemic—sparse text | Near-empty criteria. No eligibility information to score against. |
| 4 | 18yo Dengue | "Adolescents ages 14-19" (no disease criteria stated) | Systemic—sparse text | Meets age, but trial gives no condition text — model can't confirm relevance. |
| 5 | 4yo KD girl | "Boys and girls meeting CDC KD criteria, presented by day 10" | Reasonable disagreement | Clear match, but exclusion clause ("after day 10") may have confused the classifier. |

**Key finding**: False negatives are rare (0.5% of errors) and concentrated in trials with near-empty criteria text. Not a priority.

---

### Cell: Predicted partially_relevant, Actually relevant — 21 errors (2.2% of all errors)

The model under-scores genuine matches. Several of these are likely label quality issues rather than model failures — the annotator marked "partial" but the patient appears to fully meet stated criteria.

Notable cases:
- Respiratory child → CXR/dyspnea trial (clear match, scored partial)
- 27F pregnant, Hb=9.0 → iron anemia in pregnancy trial (Hb<11, first trimester — patient meets all stated criteria, label=2 seems correct, model scored partial)
- 15yo girl with fatigue → pediatric chest pain outpatient trial (borderline eligibility)
- CJD patient → Alzheimer's disease trial (model scored partial but label is relevant — possible data quality issue)

#### Stage attribution

**Classifier stage** under-confidence. These are cases where the classifier assigned the correct direction (relevant > not_relevant) but underestimated the confidence level, landing on partial rather than relevant.

---

### Cell: Predicted not_relevant, Actually partially_relevant — 11 errors (1.2% of all errors)

Rare. Mostly age mismatches (adult elbow fracture → pediatric forearm fracture trials, x3), wrong-condition matches (Giardia → C. diff, Lyme disease → GAS infection), and one ectopic pregnancy match to an ectopic pregnancy trial that was under-scored.

---

## Data Quality Issues

| # | Topic | Trial | Actual label | Issue |
|---|-------|-------|:------------:|-------|
| 1 | 27F pregnant, Hb=9.0 g/dL, first trimester | Iron anemia pregnancy trial (Hb<11, first trimester) | 1 (partial) | Patient explicitly meets all stated criteria — label should likely be 2 |
| 2 | 56F post-mastectomy, acute SOB | Suspected DVT of leg trial | 0 (not_rel) | Patient post-surgery + PE symptoms — has high prior for DVT; label=0 may be too strict |
| 3 | 62yo CJD-like, cognitive decline | MCI / possible AD trial | 0 (not_rel) | Overlapping phenotype; partial eligibility arguably warranted |
| 4 | 4yo KD girl | KD trial with "presented by day 10" exclusion | 2 (relevant) | Model predicted not_relevant — if day of illness >10, label=2 may be wrong |
| 5 | 18yo Dengue | Adolescents 14-19 trial (no disease text) | 2 (relevant) | Trial has no Dengue criteria text; how was this judged as relevant? Possible pool-based relevance assumption. |

The partial↔relevant boundary is the most systematically ambiguous in the TREC labels. TREC's 1 ("eligible but not the best match") vs 2 ("highly relevant") conflates eligibility criteria satisfaction with clinical appropriateness as a trial option — a distinction that is genuinely hard to annotate consistently.

---

## Priority Fixes

### 1. Classifier relevant-prediction bias — affects all 951 errors indirectly
**Root cause**: Model outputs "relevant" for 1067/1340 examples (79.6%) despite only 294 (21.9%) being truly relevant. Likely cause is class imbalance in the training set or equal-weight cross-entropy loss.  
**Fix**: Retrain SciBERT classifier with class-weighted loss. Based on actual distribution (not_rel:partial:relevant = 699:347:294), inverse-frequency weights would be approximately `[0.64, 1.29, 1.53]`. Alternatively, calibrate output logits post-hoc using temperature scaling on a held-out validation set.  
**Verify**: Predicted distribution should shift from (83, 190, 1067) toward (≈700, ≈350, ≈290) on balanced test data.

### 2. Lab value / numeric threshold matching — affects ~100-150 of the 282 (1→2) errors, highest clinical cost
**Root cause**: Trials containing explicit numeric thresholds (Hb, BMI, eGFR, HbA1c, platelet count) are 1.42× enriched in partial→relevant errors. The model cannot compare `patient Hb=9.0` vs `trial threshold Hb<11`, or identify that the patient's BMI of 35 exceeds a trial's BMI≤34 cap.  
**Fix**: Integrate ctproc lab extraction to produce structured `(lab_name, comparator, threshold)` tuples from trial inclusion/exclusion text and `(lab_name, value)` from patient topic text. Add a structured comparison layer before the classifier that hard-excludes violations.  
**Verify**: Run lab extraction on the 282 (1→2) errors and count how many have extractable thresholds that explain the partial label.

### 3. Demographic exclusion layer — affects ~150 (0→2) and ~40 (0→1) errors
**Root cause**: Age exclusions are the most common missed filter (pediatric patient → adult-only trial and vice versa). Patient age appears in topic text; trial age ranges appear in criteria text. Both are parseable with simple regex.  
**Fix**: Add a demographic pre-filter before the classifier: extract patient age from topic, extract `[min_age, max_age]` from trial criteria, hard-exclude if outside range. Similarly for gender.  
**Verify**: Count how many (0→2) and (0→1) errors are cleared by this filter.

### 4. Category hard-gate for cross-domain mismatches — affects ~200 of the (0→2) errors
**Root cause**: Cardiac patients matched to orthopedic trials; respiratory patients matched to psychiatric trials. The 14-category bart-large-mnli classifier already runs at the sim_filter stage and produces category vectors — but in eval mode no docs are filtered.  
**Fix**: Use category similarity as a hard pre-filter (not just a soft reranking signal): if cosine similarity between topic category vector and trial category vector is below a threshold, score = 0 without calling the classifier. This is computationally cheap and catches cross-domain mismatches.  
**Verify**: Apply the hard-gate to the (0→2) errors and measure what fraction would be caught without removing true positives.
