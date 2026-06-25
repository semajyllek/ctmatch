import os
import unittest
from ctmatch.data.data_paths import (
    _data_root, TREC_REL_PATH, KZ_REL_PATH,
    get_data_tuples, get_trec_doc_data_tuples,
    get_trec_topic_data_tuples, get_kz_doc_data_tuples,
    get_kz_topic_data_tuples,
)


class TestDataRoot(unittest.TestCase):

    def test_default_resolves_to_data_dir(self):
        root = _data_root()
        self.assertEqual(root.name, "data")
        self.assertTrue(str(root).endswith("ctmatch/data"))

    def test_env_override(self):
        os.environ["CTMATCH_DATA_ROOT"] = "/tmp/test_data"
        try:
            root = _data_root()
            self.assertEqual(str(root), "/tmp/test_data")
        finally:
            del os.environ["CTMATCH_DATA_ROOT"]


class TestPathConstants(unittest.TestCase):

    def test_trec_rel_path_points_to_judgments(self):
        self.assertTrue(TREC_REL_PATH.endswith("trec_21_judgments.txt"))

    def test_kz_rel_path_points_to_qrels(self):
        self.assertTrue(KZ_REL_PATH.endswith("qrels-clinical_trials.txt"))

    def test_trec_judgments_exist(self):
        """Verify the actual TREC 2021 judgments file is present."""
        if os.path.exists(_data_root()):
            self.assertTrue(os.path.exists(TREC_REL_PATH), f"Missing: {TREC_REL_PATH}")

    def test_kz_qrels_exist(self):
        """Verify the Koontz qrels file is present."""
        if os.path.exists(_data_root()):
            self.assertTrue(os.path.exists(KZ_REL_PATH), f"Missing: {KZ_REL_PATH}")


class TestDataTuples(unittest.TestCase):

    def test_trec_doc_tuples_5_parts(self):
        """TREC 2021 has 5 zip parts."""
        tuples = get_trec_doc_data_tuples()
        self.assertEqual(len(tuples), 5)

    def test_trec_doc_sources_are_zips(self):
        for src, tgt in get_trec_doc_data_tuples():
            self.assertTrue(src.endswith(".zip"), f"Expected .zip: {src}")
            self.assertTrue(tgt.endswith(".jsonl"), f"Expected .jsonl: {tgt}")

    def test_trec_doc_parts_numbered(self):
        """Each part should have a unique part number."""
        sources = [src for src, _ in get_trec_doc_data_tuples()]
        for i in range(1, 6):
            self.assertTrue(
                any(f"part{i}.zip" in s for s in sources),
                f"Missing part{i}",
            )

    def test_trec_topic_tuples_both_years(self):
        """Should have both 2021 and 2022 topic files."""
        tuples = get_trec_topic_data_tuples()
        self.assertEqual(len(tuples), 2)
        sources = [src for src, _ in tuples]
        self.assertTrue(any("trec_21" in s for s in sources))
        self.assertTrue(any("trec_22" in s for s in sources))

    def test_trec_topics_are_xml(self):
        for src, tgt in get_trec_topic_data_tuples():
            self.assertTrue(src.endswith(".xml"))
            self.assertTrue(tgt.endswith(".jsonl"))

    def test_trec_topic_xml_exists(self):
        """Verify the actual TREC topic XML is present."""
        if os.path.exists(_data_root()):
            src, _ = get_trec_topic_data_tuples()[0]
            self.assertTrue(os.path.exists(src), f"Missing: {src}")

    def test_kz_doc_tuples(self):
        tuples = get_kz_doc_data_tuples()
        self.assertEqual(len(tuples), 1)
        self.assertTrue(tuples[0][0].endswith(".zip"))

    def test_kz_topic_tuples(self):
        tuples = get_kz_topic_data_tuples()
        self.assertEqual(len(tuples), 1)
        self.assertTrue(tuples[0][0].endswith(".topics"))

    def test_get_data_tuples_dispatch(self):
        doc_t, topic_t = get_data_tuples("trec")
        self.assertEqual(len(doc_t), 5)
        self.assertEqual(len(topic_t), 2)

        doc_k, topic_k = get_data_tuples("kz")
        self.assertEqual(len(doc_k), 1)
        self.assertEqual(len(topic_k), 1)

    def test_all_paths_under_data_root(self):
        root = str(_data_root())
        for src, tgt in get_trec_doc_data_tuples():
            self.assertTrue(src.startswith(root))
            self.assertTrue(tgt.startswith(root))


class TestRealDataPresence(unittest.TestCase):
    """Verify key local data files exist (skipped if data dir is missing)."""

    def setUp(self):
        self.data_root = _data_root()
        if not self.data_root.exists():
            self.skipTest("Data directory not present")

    def test_trec_21_topics_xml(self):
        path = self.data_root / "trec_data" / "trec_21_topics.xml"
        self.assertTrue(path.exists())

    def test_trec_22_topics_xml(self):
        path = self.data_root / "trec_data" / "trec_22_topics.xml"
        self.assertTrue(path.exists())

    def test_kz_topics(self):
        path = self.data_root / "kz_data" / "topics-2014_2015-description.topics"
        self.assertTrue(path.exists())

    def test_kz_qrels(self):
        path = self.data_root / "kz_data" / "qrels-clinical_trials.txt"
        self.assertTrue(path.exists())

    def test_combined_classifier_data(self):
        path = self.data_root / "combined_classifier_data.jsonl"
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
