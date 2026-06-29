"""
Tests for classifier_filter ranking logic and batch_inference output contract.

Run locally:  pytest tests/test_classifier_filter.py -v
Run in Colab: !pytest /content/drive/.../ctmatch/tests/test_classifier_filter.py -v

Torch is optional for the pure-logic tests.
The hub-model tests (TestRelevantColIndex, TestHubModelColumnOrdering) require
torch and network access — they load the real checkpoint and verify that
relevant_col_index() returns a column where truly-relevant docs score higher
than truly-irrelevant docs. These are the ground-truth contract tests; all
ranking logic must agree with them.
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─── Helpers that replicate the two ranking strategies exactly ────────────────

def rank_by_ascending_p_not_relevant(probs, doc_set, top_n):
    """OLD pipeline logic: ascending P(not_relevant) = probs[:,0]. Returns ranked doc_set."""
    neg_scores = np.asarray(probs)[:, 0]
    sorted_indices = list(np.argsort(neg_scores)[:min(len(doc_set), top_n)])
    return [doc_set[i] for i in sorted_indices]


def rank_by_descending_p_relevant(probs, doc_set, top_n):
    """NEW pipeline logic: descending P(relevant) = probs[:,2]. Returns ranked doc_set."""
    rel_scores = np.asarray(probs)[:, 2]
    sorted_indices = list(np.argsort(-rel_scores)[:min(len(doc_set), top_n)])
    return [doc_set[i] for i in sorted_indices]


def ndcg_at_k(ranked_ids, doc2rel, k=10):
    dcg = sum(
        (2 ** doc2rel.get(doc_id, 0) - 1) / np.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
    )
    ideal_rels = sorted(doc2rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Argsort direction ────────────────────────────────────────────────────────

class TestArgsortDirection:
    """np.argsort semantics — the foundation the ranking strategy is built on."""

    def test_argsort_no_neg_gives_smallest_first(self):
        scores = np.array([0.8, 0.1, 0.5])
        assert list(np.argsort(scores)) == [1, 2, 0], (
            "np.argsort(x) is ASCENDING: index of smallest value comes first"
        )

    def test_argsort_neg_gives_largest_first(self):
        scores = np.array([0.8, 0.1, 0.5])
        assert list(np.argsort(-scores)) == [0, 2, 1], (
            "np.argsort(-x) is equivalent to DESCENDING: index of largest value comes first"
        )

    def test_ascending_p_not_rel_puts_lowest_p_not_rel_first(self):
        # OLD strategy: rank by p[:,0] ascending
        # p[:,0] = P(not_relevant) for each doc
        probs = np.array([
            [0.50, 0.30, 0.20],  # doc A: P(not_rel)=0.50
            [0.10, 0.80, 0.10],  # doc B: P(not_rel)=0.10  ← LOWEST, ranks first
            [0.80, 0.15, 0.05],  # doc C: P(not_rel)=0.80
        ])
        scores = probs[:, 0]
        order = list(np.argsort(scores))
        assert order[0] == 1, f"Doc B (lowest P(not_rel)) should rank first; got index {order[0]}"

    def test_descending_p_relevant_puts_highest_p_rel_first(self):
        # NEW strategy: rank by p[:,2] descending
        probs = np.array([
            [0.50, 0.30, 0.20],  # doc A: P(relevant)=0.20
            [0.10, 0.80, 0.10],  # doc B: P(relevant)=0.10
            [0.05, 0.15, 0.80],  # doc C: P(relevant)=0.80  ← HIGHEST, ranks first
        ])
        scores = probs[:, 2]
        order = list(np.argsort(-scores))
        assert order[0] == 2, f"Doc C (highest P(relevant)) should rank first; got index {order[0]}"


# ─── Core ranking correctness ─────────────────────────────────────────────────

class TestRankingCorrectness:
    """
    Given calibrated 3-class predictions, confirm which strategy correctly surfaces
    relevant docs when there is significant partial-class probability mass.

    This is the scenario that breaks with SciBERT (which barely predicts partial)
    but is common with calibrated models like BioLinkBERT-large.
    """

    @pytest.fixture
    def calibrated_probs_and_labels(self):
        """
        6 docs. True labels: [relevant, partial, not, not, partial, relevant]
        The partial docs (docs 1 and 4) have LOWER P(not_relevant) than one of the
        relevant docs (doc 5) — this is the adversarial case for the old strategy.
        """
        probs = np.array([
            [0.05, 0.20, 0.75],  # doc idx 0: relevant    P(not)=0.05, P(rel)=0.75
            [0.08, 0.82, 0.10],  # doc idx 1: partial     P(not)=0.08, P(rel)=0.10  ← low P(not)!
            [0.80, 0.15, 0.05],  # doc idx 2: not_rel     P(not)=0.80
            [0.75, 0.20, 0.05],  # doc idx 3: not_rel     P(not)=0.75
            [0.09, 0.81, 0.10],  # doc idx 4: partial     P(not)=0.09, P(rel)=0.10  ← low P(not)!
            [0.12, 0.13, 0.75],  # doc idx 5: relevant    P(not)=0.12, P(rel)=0.75
        ])
        true_labels = [2, 1, 0, 0, 1, 2]  # ground truth
        doc_set = [10, 20, 30, 40, 50, 60]  # arbitrary pipeline indices
        return probs, true_labels, doc_set

    def test_new_code_puts_relevant_docs_first(self, calibrated_probs_and_labels):
        probs, true_labels, doc_set = calibrated_probs_and_labels
        ranked = rank_by_descending_p_relevant(probs, doc_set, top_n=6)
        # Docs 10 and 60 (indices 0 and 5) are the two relevant ones
        top2 = set(ranked[:2])
        assert top2 == {10, 60}, (
            f"NEW code: top-2 must be the relevant docs [10,60], got {ranked[:2]}\n"
            f"Full ranking: {ranked}"
        )

    def test_old_code_promotes_partial_docs_above_relevant(self, calibrated_probs_and_labels):
        """
        Demonstrates the OLD bug: partial docs (idx 1, 4) have P(not_relevant) in
        [0.08, 0.09] which is LOWER than relevant doc idx 5's P(not_relevant)=0.12.
        So the old ascending-P(not_relevant) strategy ranks partials above relevant doc 60.
        """
        probs, true_labels, doc_set = calibrated_probs_and_labels
        ranked = rank_by_ascending_p_not_relevant(probs, doc_set, top_n=6)
        # Doc 20 (partial, P(not)=0.08) and doc 50 (partial, P(not)=0.09) should
        # both appear before doc 60 (relevant, P(not)=0.12)
        pos_partial_20 = ranked.index(20)
        pos_partial_50 = ranked.index(50)
        pos_relevant_60 = ranked.index(60)
        assert pos_partial_20 < pos_relevant_60 and pos_partial_50 < pos_relevant_60, (
            f"OLD bug NOT triggered: expected partial docs to rank above relevant doc 60.\n"
            f"Positions — doc20={pos_partial_20}, doc50={pos_partial_50}, doc60={pos_relevant_60}\n"
            f"Full ranking: {ranked}"
        )

    def test_new_code_ranks_not_relevant_last(self, calibrated_probs_and_labels):
        probs, true_labels, doc_set = calibrated_probs_and_labels
        ranked = rank_by_descending_p_relevant(probs, doc_set, top_n=6)
        bottom2 = set(ranked[-2:])
        assert bottom2 == {30, 40}, (
            f"NEW code: bottom-2 must be not_relevant docs [30,40], got {ranked[-2:]}"
        )

    def test_both_codes_equivalent_when_partial_mass_is_near_zero(self):
        """When P(partial) ≈ 0 for all docs (SciBERT behaviour), both strategies agree."""
        probs = np.array([
            [0.05, 0.02, 0.93],  # relevant
            [0.12, 0.02, 0.86],  # relevant
            [0.85, 0.10, 0.05],  # not relevant
        ])
        doc_set = [1, 2, 3]
        old_rank = rank_by_ascending_p_not_relevant(probs, doc_set, top_n=3)
        new_rank = rank_by_descending_p_relevant(probs, doc_set, top_n=3)
        assert old_rank == new_rank, (
            f"When P(partial) ≈ 0, both strategies must agree.\nOLD={old_rank}, NEW={new_rank}"
        )

    def test_top_n_limits_output_length(self, calibrated_probs_and_labels):
        probs, _, doc_set = calibrated_probs_and_labels
        assert len(rank_by_descending_p_relevant(probs, doc_set, top_n=3)) == 3
        assert len(rank_by_descending_p_relevant(probs, doc_set, top_n=6)) == 6
        assert len(rank_by_descending_p_relevant(probs, doc_set, top_n=100)) == 6  # capped at len(doc_set)


# ─── NDCG impact ─────────────────────────────────────────────────────────────

class TestNDCGImpact:
    """Quantify how much NDCG@10 the P(not_relevant) strategy costs vs P(relevant)."""

    def test_new_ranking_beats_old_when_partial_dominant(self):
        """
        Topic with 2 relevant, 3 partial, 5 not_relevant docs.
        Partial docs have lower P(not_relevant) than the relevant docs.
        Old strategy demotes relevant docs; new strategy promotes them.
        """
        doc_ids = [f"NCT{i:04d}" for i in range(10)]
        doc2rel = {d: 0 for d in doc_ids}
        doc2rel["NCT0000"] = 2  # relevant
        doc2rel["NCT0001"] = 2  # relevant
        doc2rel["NCT0002"] = 1  # partial
        doc2rel["NCT0003"] = 1  # partial
        doc2rel["NCT0004"] = 1  # partial

        probs = np.array([
            [0.15, 0.10, 0.75],  # NCT0000: relevant  — P(not)=0.15
            [0.18, 0.08, 0.74],  # NCT0001: relevant  — P(not)=0.18
            [0.08, 0.82, 0.10],  # NCT0002: partial   — P(not)=0.08  ← lower than relevant!
            [0.09, 0.81, 0.10],  # NCT0003: partial   — P(not)=0.09  ← lower than relevant!
            [0.07, 0.83, 0.10],  # NCT0004: partial   — P(not)=0.07  ← lower than relevant!
            [0.80, 0.15, 0.05],  # NCT0005: not_rel
            [0.82, 0.13, 0.05],  # NCT0006: not_rel
            [0.78, 0.17, 0.05],  # NCT0007: not_rel
            [0.81, 0.14, 0.05],  # NCT0008: not_rel
            [0.79, 0.16, 0.05],  # NCT0009: not_rel
        ])
        doc_set = list(range(10))

        old_indices = rank_by_ascending_p_not_relevant(probs, doc_set, top_n=10)
        new_indices = rank_by_descending_p_relevant(probs, doc_set, top_n=10)

        old_ids = [doc_ids[i] for i in old_indices]
        new_ids = [doc_ids[i] for i in new_indices]

        ndcg_old = ndcg_at_k(old_ids, doc2rel, k=10)
        ndcg_new = ndcg_at_k(new_ids, doc2rel, k=10)

        print(f"\n  OLD top-5: {old_ids[:5]}  NDCG@10={ndcg_old:.4f}")
        print(f"  NEW top-5: {new_ids[:5]}  NDCG@10={ndcg_new:.4f}")

        assert ndcg_new > ndcg_old, (
            f"NEW strategy must beat OLD when partial-dominant docs exist.\n"
            f"OLD NDCG@10={ndcg_old:.4f}, NEW NDCG@10={ndcg_new:.4f}"
        )
        assert set(new_ids[:2]) == {"NCT0000", "NCT0001"}, (
            f"Relevant docs must occupy top-2 under new strategy, got {new_ids[:2]}"
        )
        assert old_ids[0] in {"NCT0002", "NCT0003", "NCT0004"}, (
            f"OLD: a partial-dominant doc must be in position 1 (demonstrating the bug), "
            f"got {old_ids[0]}"
        )

    def test_ndcg_calculation_itself(self):
        """Sanity check for the ndcg helper used in this test file."""
        doc2rel = {"A": 2, "B": 1, "C": 0}
        # Perfect ranking
        assert ndcg_at_k(["A", "B", "C"], doc2rel, k=3) == pytest.approx(1.0)
        # Worst ranking (not_rel first, relevant last)
        worst = ndcg_at_k(["C", "B", "A"], doc2rel, k=3)
        assert worst < 1.0


# ─── Torch-dependent tests (skipped locally if torch unavailable) ─────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBatchInferenceOutputContract:
    """
    batch_inference(return_preds=True) returns a torch.Tensor of shape (N, 3).
    Verify that column indexing via list-comp and direct slice agree.
    """

    def test_listcomp_and_slice_agree_for_col2(self):
        import torch
        probs = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.3, 0.5, 0.2]])
        via_listcomp = np.asarray([p[2].item() for p in probs])
        via_slice = probs[:, 2].numpy()
        np.testing.assert_allclose(via_listcomp, via_slice, rtol=1e-6)

    def test_sort_order_agrees_between_listcomp_and_slice(self):
        import torch
        probs = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.3, 0.5, 0.2]])
        via_listcomp = np.argsort(-np.asarray([p[2].item() for p in probs]))
        via_slice = np.argsort(-probs[:, 2].numpy())
        np.testing.assert_array_equal(via_listcomp, via_slice)


# ─── relevant_col_index() contract ────────────────────────────────────────────

class TestRelevantColIndex:
    """
    relevant_col_index() reads model.config.id2label and returns the column
    index labelled 'relevant'. These tests verify the logic with mock configs.
    """

    class _MockConfig:
        def __init__(self, id2label):
            self.id2label = id2label

    class _MockModel:
        def __init__(self, id2label):
            self.config = TestRelevantColIndex._MockConfig(id2label)

    class _MockClassifier:
        """Minimal stand-in for ClassifierModel so we can call relevant_col_index()."""
        def __init__(self, id2label):
            self.model = TestRelevantColIndex._MockModel(id2label)

        # paste the real implementation here so the test is self-contained
        def relevant_col_index(self):
            id2label = self.model.config.id2label
            matches = [int(k) for k, v in id2label.items() if v == 'relevant']
            assert len(matches) == 1, (
                f"Expected exactly one 'relevant' entry in id2label, got {id2label}"
            )
            return matches[0]

    def test_standard_ordering_returns_2(self):
        clf = self._MockClassifier({0: 'not_relevant', 1: 'partially_relevant', 2: 'relevant'})
        assert clf.relevant_col_index() == 2

    def test_reversed_ordering_returns_0(self):
        """If the model was saved with reversed labels, we get 0 — not 2."""
        clf = self._MockClassifier({0: 'relevant', 1: 'partially_relevant', 2: 'not_relevant'})
        assert clf.relevant_col_index() == 0

    def test_string_keys_work(self):
        """id2label loaded from JSON may have string keys."""
        clf = self._MockClassifier({'0': 'not_relevant', '1': 'partially_relevant', '2': 'relevant'})
        assert clf.relevant_col_index() == 2

    def test_missing_relevant_raises(self):
        clf = self._MockClassifier({0: 'not_relevant', 1: 'partially_relevant'})
        with pytest.raises(AssertionError, match="Expected exactly one"):
            clf.relevant_col_index()

    def test_duplicate_relevant_raises(self):
        clf = self._MockClassifier({0: 'relevant', 1: 'partially_relevant', 2: 'relevant'})
        with pytest.raises(AssertionError, match="Expected exactly one"):
            clf.relevant_col_index()


# ─── Hub model column ordering (requires network + torch) ─────────────────────

HUB_CHECKPOINT = 'semaj83/ctmatch-clf-biolinkbert-large'

# A clearly relevant pair: diabetic patient, diabetes trial
RELEVANT_TOPIC = "58-year-old male with type 2 diabetes, HbA1c 8.2%, on metformin."
RELEVANT_DOC = (
    "Eligibility: Adults aged 18+ with confirmed type 2 diabetes mellitus "
    "and HbA1c between 7.5% and 11%. Exclusion: type 1 diabetes, pregnancy."
)
# A clearly irrelevant pair: same patient, ophthalmology trial
IRRELEVANT_DOC = (
    "Eligibility: Adults with primary open-angle glaucoma and intraocular pressure "
    "above 21 mmHg. Exclusion: prior ocular surgery, diabetic retinopathy."
)

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.hub
class TestHubModelColumnOrdering:
    """
    Load the real checkpoint and verify that the column returned by
    relevant_col_index() actually scores the relevant doc higher than the
    irrelevant doc. This is the ground-truth contract test — if it fails,
    the column ordering in the hub model is wrong and the ranking code must
    use a different column (or the model must be retrained).

    Mark: pytest -m hub  (slow, requires network)
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HUB_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(HUB_CHECKPOINT)
        model.eval()
        return model, tokenizer

    def _get_probs(self, model, tokenizer, topic, doc):
        import torch
        inputs = tokenizer(topic, doc, return_tensors='pt',
                           truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=1).squeeze(0)

    def test_id2label_contains_relevant(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert 'relevant' in model.config.id2label.values(), (
            f"Model id2label has no 'relevant' entry: {model.config.id2label}"
        )

    def test_relevant_col_scores_higher_on_matching_pair(self, model_and_tokenizer):
        """The column labelled 'relevant' in id2label must score HIGHER for the
        clearly-relevant doc than for the clearly-irrelevant doc."""
        model, tokenizer = model_and_tokenizer

        id2label = model.config.id2label
        rel_col = next(int(k) for k, v in id2label.items() if v == 'relevant')

        probs_rel = self._get_probs(model, tokenizer, RELEVANT_TOPIC, RELEVANT_DOC)
        probs_irrel = self._get_probs(model, tokenizer, RELEVANT_TOPIC, IRRELEVANT_DOC)

        print(f"\n  id2label: {id2label}")
        print(f"  rel_col={rel_col}")
        print(f"  P(col{rel_col}) for RELEVANT doc:    {probs_rel[rel_col]:.4f}")
        print(f"  P(col{rel_col}) for IRRELEVANT doc:  {probs_irrel[rel_col]:.4f}")
        print(f"  All probs (relevant doc):   {[f'{p:.3f}' for p in probs_rel.tolist()]}")
        print(f"  All probs (irrelevant doc): {[f'{p:.3f}' for p in probs_irrel.tolist()]}")

        assert probs_rel[rel_col] > probs_irrel[rel_col], (
            f"Column {rel_col} (labelled 'relevant' in id2label) must score higher "
            f"for the clearly-relevant doc.\n"
            f"  relevant doc   P(col{rel_col})={probs_rel[rel_col]:.4f}\n"
            f"  irrelevant doc P(col{rel_col})={probs_irrel[rel_col]:.4f}\n"
            f"  id2label: {id2label}\n"
            f"  If this fails, the hub model's column ordering does not match its id2label."
        )
