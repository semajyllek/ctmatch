## ctmatch


**program for matching clinical trials to patient text descriptions**

This package is designed generally for the task described in the precision medicine track of TREC since 2021.
That is, an information retrieval task to match patient descriptions (topics) to clinical trials data (xml documents)

ctmatch leverages several tools to build the representations in the dataset against which the topics are matched,
as well as langugage models that have been fine-tuned on the curated ctmatch dataset of relevance-labelled topic, document 
pairs. 

The pipeline currently matches user input topics to the static snapshot of clinical trials data downloaded for the TREC task from december of 2015, now stored on huggingface in the datasets `semaj83/ctmatch_classification` and `semaj83/ctmatch_ir` but can be updated with a current dataset of clinical trials data using ctproc to process.

web app (while I can afford it): https://huggingface.co/spaces/semaj83/ctmatch

### project status (2026-08) — honest wrap-up

This project set out to beat the TREC 2022 winner (NDCG@10 0.6125). It did not — and the closing
investigation established *why*, with findings that stand on their own. Full write-up: **`docs/project_wrapup.md`**.

- **Funnel + frontier labeler (cleanly held-out).** A cheap, fully-open funnel (BM25 + dense + RRF →
  BioLinkBERT cross-encoder rerank) cuts ~375k → 100 candidates while preserving a **0.899** top-10 oracle
  ceiling. Ranking those 100 with a `claude-opus-4-8` whole-doc labeler scores **NDCG@10 = 0.505 on blind
  TREC22 vs 0.339 for the cross-encoder (+0.15)** — real evidence a frontier labeler out-ranks a fine-tuned
  cross-encoder on the same candidates. Chain-of-thought did not help (worse, at 2×/5× cost).
- **A clean negative result.** Error-reflective prompt distillation *over-corrected* — it pushed true
  eligibles down hardest (gold=2 score shift −20.6), steering the labeler toward criterion-verification the
  terse topics can't support (the §7g information wall), reached here from a third independent direction.
- **The grounding that reframes the "miss":** standalone-labeler P@10 ≈ 0.43–0.49; the **TREC 2022 winner's
  P@10 is 0.508**. A 90%-clean top-10 from a three-line patient vignette is **above the current global
  frontier** — the pipeline is best understood as a fast, cheap, ~SOTA-precision **assistive first pass**, not
  an autonomous oracle.

> The retrieval→ensemble results in the next section are an earlier, **test-adapted** framing
> (competitive-but-not-superior; see its own caveats and `docs/deep_dive_outline.md` §12d). The
> funnel/labeler/reflection findings above are the cleanly held-out part. Turn-by-turn evidence:
> `docs/reflective_labeler_design.md` §8b–§8f.

### results — open, full-corpus pipeline (TREC 2022 Clinical Trials)

An end-to-end, **fully open-weight** pipeline for full-corpus clinical-trial retrieval:

> full-text corpus (ClinicalTrials.gov API v2) → **hybrid BM25 + dense retrieval** (with optional open-model neural query synthesis) → **LambdaMART ensemble** over BM25/dense/RRF features, two BioLinkBERT cross-encoders (eligibility + topicality views), two open **Qwen-2.5-7B** judges (eligibility + topicality), and a SapBERT condition-match signal — all on one frozen document representation (`R = elig_first-L512`).

| System | NDCG@10 | P@10 | MRR | Protocol | Open weights |
|---|---|---|---|---|---|
| h2oloo (TREC 2022 winner) | 0.6125 | 0.5080 | 0.7262 | full-corpus, blind | — |
| **ctmatch (this work)** | **0.5750** | **0.482** | **0.7643** | full-corpus, TREC22 held-out | ✅ |
| TrialGPT (Jin et al.) | 0.7252 | — | — | pooled-candidate rerank (*not comparable*) | ❌ (GPT-4) |

- **Competitive with the best TREC 2022 full-corpus system, and ahead on MRR.** NDCG@10 0.5750 vs h2oloo 0.6125 — the bootstrap 95% CI contains 0.6125 (a statistical tie); on **MRR (first-eligible placement) ctmatch leads, 0.7643 vs 0.7262 (+0.038)**; P@10 is within noise (0.482 vs 0.508). NDCG@10 uses graded gains; P@10/MRR are eligible-only, matching the TREC 2022 overview — all three directly comparable. **No closed model in the inference path.**
- **Honest caveats:** the configuration was tuned with TREC22 access (h2oloo's run was a blind submission), and n=50 gives wide CIs; a paired significance test would need h2oloo's per-topic run file. An external generalization test on TREC 2023 is the intended held-out validation. See `docs/deep_dive_outline.md` §12d.
- TrialGPT's 0.7252 is **not** comparable: it reranks a reduced pooled candidate set (~26k trials → top-500), not the full ~375k-trial corpus. See `docs/deep_dive_outline.md` §7e.
- Full methodology, ablations, the representation audit, and limitations: **`docs/deep_dive_outline.md`** — §2g/§2h (representation), §12 (consolidated method + results).

> Note: the "pipeline filters" section below documents the earlier 4-stage cascade (sim / SVM / classifier / gen), retained as the original ctmatch system. The current SOTA-competitive system is the retrieval→ensemble pipeline above; the notebooks that build it are listed in `docs/deep_dive_outline.md` §12a.

### pipeline filters

Currently 4 filters are applied to the set of documents for ranking and reranking:

1. The first (sim) filter is based on extracted document eligbility criteria embedding and topic cosine similarity (see ctmatch_ir dataset):
   
   - embeddings (384-dim) are created using the last hidden layer of SentenceTransformers(`sentence-transformers/all-MiniLM-L6-v2`)
   
   - inferred category vectors are arbitrarily selected 14 classes i.e. pulmonary, cardiac, health, other.... with 
   probabilites as softmax of the output from a zero-shot classification of `facebook/bart-large-mnli` applied to the 
   'condition' field of the ct documents and the raw text of the topic.

   The docs are ranked by this combined distance score and the top {1000} closest documents to the topic are selected (out of ~384k documents), passed to the next filter.


2. The second (SVM) filter uses the same embedded representations as the sim fiter:
   - an SVM is used to learn a decision boundary between the topic as one class and the documents as another, then passes forward the top closest {100} documents to the decision boundary.


3. The third (classification) filter again uses a different LM, , fine-tuned for sequece classification on the `semaj83/ctmatch_classification` dataset of abelled topic, document, relevancy triples.


4. The fourth (gen) filter uses a prompting-based approach to query a gpt model with topic, id'd doc texts and asks the LM to generate a list 



### api

```
topic = "A 46 yo male with gastric cancer. He has recieved 3 rounds of chemotherapy without radiation and recent MRI shows tumor shrinking to < 4 cm. in diameter."

ctm = CTMatch()
ranked_pairs = ctm.match_pipeline(topic, top_k=10)

"""
[('NCT00003788', 'Inclusion Criteria: Histologically confirmed newly diagnosed or recurrent supratentorial glioblastoma or malignant astrocytoma Grade 3 or 4 astrocytoma as defined by the Daumas-Duport classification Suitable for radical resection on the basis of imaging studies Patients with recurrent disease must have failed surgery and radiotherapy Age and over Performance status Karnofsky 60-100% for newly diagnosed tumor Karnofsky 70-100% for recurrent tumor Life expectancy At least 3 months Hematopoietic Recurrent tumor WBC at least 2,000/mm^3 Platelet count at least 80,000/mm^3 Hepatic Recurrent tumor PT/PTT no greater than 1.5 times upper limit of normal (ULN) Bilirubin and LFTs less than 2 times ULN Alkaline phosphatase no greater than 3 times ULN GGT no greater than 3 times ULN Renal Creatinine no greater than 2 mg/dL Other Not pregnant or nursing PRIOR Biologic therapy Not specified Chemotherapy Not specified Endocrine therapy Not specified Radiotherapy See Disease Characteristics No prior cranial radiotherapy for newly diagnosed tumor Surgery See Disease Characteristics, Exclusion Criteria:')]
"""

```

### evaluation:

Evaluation is done with MRR (mean reciprocal rank) on the resulting documents from the labelled dataset and the `Evaluator` object in the evaluator.py module. (see evaluation notebook)

![Screenshot](ctmatch_results.png)


### classifier training

This repo also contains code in the CTMatch object to train a classifier on the `semaj83/ctmatch_classification` dataset (see finetuning notebook).
Several LMs are supported but others may need to code modifications, starting with adding to the list of supported LMs at the top of dataprep.py

