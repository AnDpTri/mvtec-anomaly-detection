"""Kiểm thử metric: AUROC, AUPRO, ngưỡng từ ảnh good validation."""
import unittest

import numpy as np

from helpers import fake_data
from src.tn.metrics import aupro, compute_metrics, image_scores, metrics_full, per_defect_auroc, thresholds_from_val


class Metrics(unittest.TestCase):
    def test_perfect_maps(self):
        _, _, labels, masks = fake_data()
        maps = masks.astype(np.float32)
        met = compute_metrics(image_scores(maps), maps, labels, masks)
        for k in ("i_auroc", "i_ap", "p_auroc", "p_ap", "aupro"):
            self.assertAlmostEqual(met[k], 1.0, places=4, msg=k)

    def test_inverted_maps(self):
        _, _, labels, masks = fake_data()
        maps = 1.0 - masks.astype(np.float32)
        met = compute_metrics(image_scores(maps), maps, labels, masks)
        self.assertLess(met["p_auroc"], 0.1)
        self.assertLess(met["aupro"], 0.1)

    def test_aupro_random_is_low_and_no_defect_is_nan(self):
        _, _, _, masks = fake_data()
        rng = np.random.default_rng(0)
        self.assertLess(aupro(rng.random(masks.shape), masks), 0.2)
        self.assertTrue(np.isnan(aupro(rng.random(masks.shape), np.zeros_like(masks))))

    def test_image_scores_is_max(self):
        maps = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        np.testing.assert_array_equal(image_scores(maps), [8, 17])

    def test_thresholds_and_metrics_at_threshold(self):
        _, _, labels, masks = fake_data()
        val = np.random.default_rng(0).normal(1, 0.1, (10, 8, 8)).astype(np.float32)
        thr = thresholds_from_val(val)
        s = val.reshape(10, -1).max(1)
        self.assertAlmostEqual(thr["thr_img"], s.mean() + 3 * s.std(), places=5)
        maps = masks.astype(np.float32) * 5
        met = metrics_full(image_scores(maps), maps, labels, masks, thr)
        self.assertEqual(met["img_tpr_thr"], 1.0)
        self.assertEqual(met["img_fpr_thr"], 0.0)
        self.assertAlmostEqual(met["pix_f1_thr"], 1.0)


class PerDefect(unittest.TestCase):
    def test_per_defect_auroc_separates_defect_types(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.3])
        labels = np.array([0, 0, 1, 1, 1, 1])
        dtypes = ["good", "good", "a", "a", "b", "b"]
        out = per_defect_auroc(scores, labels, dtypes)
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(out["a"], 1.0)
        self.assertLess(out["b"], 1.0)


if __name__ == "__main__":
    unittest.main()
