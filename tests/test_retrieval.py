"""
Tests for retrieval modules using synthetic data shaped like the real
ctmatch_ir dataset: 384-dim embeddings, 14-dim category vectors, doc texts.
"""
import unittest
import numpy as np
import pandas as pd


def _make_embeddings(n_docs: int, dim: int = 384, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.randn(n_docs, dim))


def _make_categories(n_docs: int, n_cats: int = 14, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cats = rng.dirichlet(np.ones(n_cats), size=n_docs)
    return pd.DataFrame(cats)


# Synthetic doc texts that mimic real eligibility criteria
DOC_TEXTS = [
    "Patients with histologically confirmed pancreatic cancer all stages",
    "Adults over 18 with hypertension and diabetes mellitus type 2",
    "Hemoglobin at least 9 g/dL Platelet count at least 100000",
    "History of coronary artery disease and prior myocardial infarction",
    "Diagnosed with relapsing remitting multiple sclerosis RRMS",
    "Severe asthma exacerbation FEV1 less than 60 percent predicted",
    "Recurrent urinary tract infections with Klebsiella Pseudomonas",
    "Anaplastic astrocytoma recurrent glioblastoma multiforme",
    "Sickle cell disease with history of vaso-occlusive crises",
    "Type 1 diabetes insulin dependent for at least 5 years",
    "Acute pancreatitis elevated amylase lipase leukocytosis",
    "Cervical cancer stage IB HPV positive radical hysterectomy",
    "Chronic hepatitis C HCV RNA positive interferon ribavirin",
    "Major depressive disorder MDD HAM-D score above 17",
    "Parkinson disease tremor bradykinesia levodopa therapy",
    "Neonatal jaundice phototherapy bilirubin elevated newborn",
    "Breast cancer HER2 positive ER PR negative lymph node involvement",
    "COPD chronic obstructive pulmonary disease oxygen therapy home",
    "Cholecystitis gallstones right upper quadrant pain cholecystectomy",
    "Acromegaly pituitary adenoma IGF-1 elevated transsphenoidal surgery",
]


class TestEmbeddingFilter(unittest.TestCase):

    def setUp(self):
        self.n_docs = 20
        self.emb_df = _make_embeddings(self.n_docs)
        self.cat_df = _make_categories(self.n_docs)
        rng = np.random.RandomState(99)
        self.topic_emb = rng.randn(384)
        self.topic_cat = rng.dirichlet(np.ones(14))

    def test_sim_filter_returns_correct_count(self):
        from ctmatch.matching.retrieval import sim_filter
        doc_set = list(range(self.n_docs))
        result = sim_filter(self.topic_emb, self.topic_cat, self.emb_df, self.cat_df, doc_set, top_n=5)
        self.assertEqual(len(result), 5)

    def test_sim_filter_returns_subset(self):
        from ctmatch.matching.retrieval import sim_filter
        doc_set = list(range(self.n_docs))
        result = sim_filter(self.topic_emb, self.topic_cat, self.emb_df, self.cat_df, doc_set, top_n=5)
        for idx in result:
            self.assertIn(idx, doc_set)

    def test_embedding_only_filter(self):
        from ctmatch.matching.retrieval import embedding_only_filter
        doc_set = list(range(self.n_docs))
        result = embedding_only_filter(self.topic_emb, self.emb_df, doc_set, top_n=5)
        self.assertEqual(len(result), 5)

    def test_embedding_only_ordered_by_similarity(self):
        from ctmatch.matching.retrieval import embedding_only_filter
        from numpy.linalg import norm
        doc_set = list(range(self.n_docs))
        result = embedding_only_filter(self.topic_emb, self.emb_df, doc_set, top_n=self.n_docs)
        # verify descending similarity
        sims = []
        for idx in result:
            doc_emb = self.emb_df.iloc[idx].values
            sim = np.dot(self.topic_emb, doc_emb) / (norm(self.topic_emb) * norm(doc_emb))
            sims.append(sim)
        for i in range(len(sims) - 1):
            self.assertGreaterEqual(sims[i], sims[i + 1])

    def test_top_n_larger_than_set(self):
        from ctmatch.matching.retrieval import embedding_only_filter
        doc_set = list(range(5))
        result = embedding_only_filter(self.topic_emb, self.emb_df, doc_set, top_n=100)
        self.assertEqual(len(result), 5)


class TestSVMFilter(unittest.TestCase):

    def setUp(self):
        self.n_docs = 20
        self.emb_df = _make_embeddings(self.n_docs)
        rng = np.random.RandomState(99)
        self.topic_emb = rng.randn(384)

    def test_svm_filter_returns_correct_count(self):
        from ctmatch.matching.retrieval import svm_filter
        doc_set = list(range(self.n_docs))
        result = svm_filter(self.topic_emb, self.emb_df, doc_set, top_n=5)
        self.assertEqual(len(result), 5)

    def test_svm_filter_excludes_topic(self):
        from ctmatch.matching.retrieval import svm_filter
        doc_set = list(range(self.n_docs))
        result = svm_filter(self.topic_emb, self.emb_df, doc_set, top_n=self.n_docs)
        # topic was index 0 in the training set, should not appear in results
        # results should all be valid doc indices
        for idx in result:
            self.assertIn(idx, doc_set)

    def test_svm_filter_on_subset(self):
        from ctmatch.matching.retrieval import svm_filter
        doc_set = [2, 5, 8, 11, 14, 17]
        result = svm_filter(self.topic_emb, self.emb_df, doc_set, top_n=3)
        self.assertEqual(len(result), 3)
        for idx in result:
            self.assertIn(idx, doc_set)


class TestBM25Filter(unittest.TestCase):

    def setUp(self):
        self.texts = DOC_TEXTS
        self.n_docs = len(self.texts)

    def test_build_index(self):
        from ctmatch.matching.retrieval import BM25Index
        index = BM25Index(self.texts)
        self.assertEqual(index._size, self.n_docs)

    def test_retrieve_cancer_query(self):
        from ctmatch.matching.retrieval import BM25Index
        index = BM25Index(self.texts)
        results = index.retrieve("pancreatic cancer diagnosis", top_n=3)
        self.assertEqual(len(results), 3)
        # doc 0 mentions "pancreatic cancer" — should rank high
        top_ids = [idx for idx, _score in results]
        self.assertIn(0, top_ids)

    def test_retrieve_diabetes_query(self):
        from ctmatch.matching.retrieval import BM25Index
        index = BM25Index(self.texts)
        results = index.retrieve("diabetes insulin type 1", top_n=3)
        top_ids = [idx for idx, _score in results]
        # doc 9 is "Type 1 diabetes insulin dependent"
        self.assertIn(9, top_ids)

    def test_retrieve_with_doc_set(self):
        from ctmatch.matching.retrieval import BM25Index
        index = BM25Index(self.texts)
        doc_set = [0, 5, 10, 15]
        results = index.retrieve("cancer", top_n=2, doc_set=doc_set)
        self.assertLessEqual(len(results), 2)
        for idx, _score in results:
            self.assertIn(idx, doc_set)

    def test_bm25_filter_function(self):
        from ctmatch.matching.retrieval import BM25Index, bm25_filter
        index = BM25Index(self.texts)
        doc_set = list(range(self.n_docs))
        result = bm25_filter(index, "asthma exacerbation", doc_set, top_n=3)
        self.assertEqual(len(result), 3)
        # doc 5 mentions "severe asthma exacerbation"
        self.assertIn(5, result)


class TestHybridFilter(unittest.TestCase):

    def setUp(self):
        self.n_docs = len(DOC_TEXTS)
        self.emb_df = _make_embeddings(self.n_docs)
        rng = np.random.RandomState(99)
        self.topic_emb = rng.randn(384)

    def test_rrf_basic(self):
        from ctmatch.matching.retrieval import reciprocal_rank_fusion
        list1 = [(0, 10.0), (1, 8.0), (2, 6.0)]
        list2 = [(2, 10.0), (0, 8.0), (3, 6.0)]
        fused = reciprocal_rank_fusion(list1, list2, k=60)
        fused_ids = [idx for idx, _score in fused]
        # doc 0 appears in both lists (rank 1 and rank 2) — should rank high
        self.assertEqual(fused_ids[0], 0)

    def test_rrf_single_list(self):
        from ctmatch.matching.retrieval import reciprocal_rank_fusion
        ranked = [(5, 1.0), (3, 0.8), (1, 0.5)]
        fused = reciprocal_rank_fusion(ranked, k=60)
        fused_ids = [idx for idx, _score in fused]
        self.assertEqual(fused_ids, [5, 3, 1])

    def test_hybrid_filter_returns_correct_count(self):
        from ctmatch.matching.retrieval import BM25Index
        from ctmatch.matching.retrieval.hybrid import hybrid_filter
        index = BM25Index(DOC_TEXTS)
        doc_set = list(range(self.n_docs))
        result = hybrid_filter(
            index, "pancreatic cancer",
            self.topic_emb, self.emb_df, doc_set, top_n=5,
        )
        self.assertEqual(len(result), 5)

    def test_hybrid_filter_returns_valid_indices(self):
        from ctmatch.matching.retrieval import BM25Index
        from ctmatch.matching.retrieval.hybrid import hybrid_filter
        index = BM25Index(DOC_TEXTS)
        doc_set = [0, 3, 7, 11, 16]
        result = hybrid_filter(
            index, "cancer treatment",
            self.topic_emb, self.emb_df, doc_set, top_n=3,
        )
        for idx in result:
            self.assertIn(idx, doc_set)


class TestBuildBM25Index(unittest.TestCase):

    def test_from_dataframe(self):
        from ctmatch.matching.retrieval import build_bm25_index
        df = pd.DataFrame({"text": DOC_TEXTS})
        index = build_bm25_index(df)
        results = index.retrieve("diabetes", top_n=2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
