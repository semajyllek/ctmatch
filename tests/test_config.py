import unittest
from ctmatch.config import PipeConfig


class TestPipeConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = PipeConfig()
        self.assertEqual(cfg.name, "scibert_finetuned_ctmatch")
        self.assertEqual(cfg.num_classes, 3)
        self.assertEqual(cfg.max_length, 512)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.seed, 42)

    def test_custom_values(self):
        cfg = PipeConfig(name="test_run", batch_size=32, learning_rate=1e-4)
        self.assertEqual(cfg.name, "test_run")
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.learning_rate, 1e-4)

    def test_filters(self):
        cfg = PipeConfig(filters=["sim", "svm", "classifier"])
        self.assertEqual(cfg.filters, ["sim", "svm", "classifier"])

    def test_default_filters_none(self):
        cfg = PipeConfig()
        self.assertIsNone(cfg.filters)

    def test_default_model_checkpoints(self):
        cfg = PipeConfig()
        self.assertIn("scibert", cfg.classifier_model_checkpoint)
        self.assertIn("MiniLM", cfg.embedding_model_checkpoint)
        self.assertIn("bart-large-mnli", cfg.category_model_checkpoint)

    def test_ir_setup_flag(self):
        cfg = PipeConfig(ir_setup=True)
        self.assertTrue(cfg.ir_setup)

    def test_is_namedtuple(self):
        cfg = PipeConfig()
        self.assertIsInstance(cfg, tuple)
        self.assertTrue(hasattr(cfg, '_fields'))

    def test_splits_default(self):
        cfg = PipeConfig()
        self.assertAlmostEqual(cfg.splits["train"], 0.8)
        self.assertAlmostEqual(cfg.splits["val"], 0.1)


if __name__ == "__main__":
    unittest.main()
