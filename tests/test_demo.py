"""Kiểm thử demo. Phần dùng mô hình thật chỉ chạy khi có results/checkpoints_tn và datasets/."""
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.tn.config import METHODS


class ToInput(unittest.TestCase):
    """Ảnh người dùng tải lên được đưa về đúng kích thước mô hình (không cần mô hình thật)."""

    def test_resizes_pil_and_numpy_and_gray(self):
        from src.demo import tn_models
        rgb = Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8))
        self.assertEqual(tn_models.to_input(rgb, (32, 48)).shape, (1, 32, 48, 3))
        self.assertEqual(tn_models.to_input(np.zeros((20, 20), np.uint8), (16, 16)).shape, (1, 16, 16, 3))
        self.assertEqual(tn_models.to_input(np.zeros((20, 20, 4), np.uint8), (16, 16)).shape, (1, 16, 16, 3))

    def test_missing_model_message(self):
        from src.demo import tn_models
        with self.assertRaises(FileNotFoundError):
            tn_models.load_meta("loai_khong_ton_tai")


class DemoWithSavedModels(unittest.TestCase):
    """Demo nạp mô hình đã fit sẵn. Bỏ qua nếu thiếu results/checkpoints_tn hoặc datasets/."""

    @classmethod
    def setUpClass(cls):
        from src.demo import tn_models
        cls.T = tn_models
        cats = [c for _, c in tn_models.categories()]
        if not cats:
            raise unittest.SkipTest("chưa có mô hình trong results/checkpoints_tn")
        cls.cat = "grid" if "grid" in cats else cats[0]
        df = pd.read_csv(tn_models.MODEL_ROOT / cls.cat / "test_scores.csv")
        df = df[[(tn_models.DATA_ROOT / p).exists() for p in df.path]]
        if df.empty:
            raise unittest.SkipTest("chưa có ảnh test trong datasets/")
        cls.row = df.iloc[0]

    def test_meta(self):
        meta = self.T.load_meta(self.cat)
        self.assertEqual(meta["category"], self.cat)
        self.assertEqual(set(meta["thr"]), set(METHODS))

    def test_predict_matches_saved_scores(self):
        img = Image.open(self.T.DATA_ROOT / self.row.path)
        for m in METHODS:
            _, amap, score, meta = self.T.predict(img, self.cat, m)
            self.assertEqual(amap.shape, tuple(meta["hw"]))
            self.assertAlmostEqual(score, float(self.row[f"score_{m}"]), delta=0.02 * abs(score) + 1e-3, msg=m)

    def test_app_predict_returns_overlay_and_verdict(self):
        from src.demo import app
        overlay, _, text = app.predict(Image.open(self.T.DATA_ROOT / self.row.path), self.cat, "patchcore")
        self.assertEqual(overlay.size[::-1], tuple(self.T.load_meta(self.cat)["hw"]))
        self.assertTrue(text.startswith("CÓ LỖI") or text.startswith("KHÔNG LỖI"))
        with self.assertRaises(ValueError):
            app.predict(None, self.cat, "ae")


if __name__ == "__main__":
    unittest.main()
