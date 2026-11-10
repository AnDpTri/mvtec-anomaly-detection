"""Kiểm thử bộ mã src/tn và demo. Chạy trên CPU, dữ liệu giả nhỏ; không cần datasets/ hay mô hình thật.

Backbone WideResNet được thay bằng mạng nhỏ (cùng số kênh, cùng độ phân giải đầu ra) để không phải tải
trọng số; phần demo nạp mô hình thật chỉ chạy khi results/checkpoints_tn có mặt.

Chạy:  python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.tn import methods
from src.tn.config import METHODS, SQUARE
from src.tn.results import Results
from src.tn.metrics import aupro, compute_metrics, image_scores, metrics_full, thresholds_from_val
from src.tn.shifts import apply_shift
from src.tn.splits import contamination_pool, n_contam, split_fit_val, tn1_ratios_for, tn1_split, tn2_split

torch.set_num_threads(2)


class FakeBackbone(nn.Module):
    """Thay cho WideResNet-50-2: trả về đặc trưng 256/512/1024 kênh ở lưới H/4, H/8, H/16."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c1 = nn.Conv2d(3, 256, 4, 4)
        self.c2 = nn.Conv2d(256, 512, 2, 2)
        self.c3 = nn.Conv2d(512, 1024, 2, 2)

    @torch.no_grad()
    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        return f1, f2, self.c3(f2)


def fake_data(n_test=6, n_good=8, size=64, seed=0):
    """Ảnh nền nhiễu nhẹ; ảnh lỗi có một ô sáng, mask đúng ô đó."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 110, (n_good, size, size, 3), dtype=np.uint8)
    test = rng.integers(90, 110, (n_test, size, size, 3), dtype=np.uint8)
    masks = np.zeros((n_test, size, size), bool)
    labels = np.array([0, 1] * (n_test // 2))
    for i in np.where(labels == 1)[0]:
        test[i, 20:36, 20:36] = 250
        masks[i, 20:36, 20:36] = True
    return base, test, labels, masks


class Splits(unittest.TestCase):
    def setUp(self):
        self.data = dict(train=[f"t{i}" for i in range(20)],
                         test=[dict(label=int(i >= 10), dtype="a" if i % 2 else "b") for i in range(40)])

    def test_n_contam(self):
        self.assertEqual(n_contam(0, 100), 0)
        self.assertEqual(n_contam(0.2, 80), 20)             # 20 lỗi / (80 + 20) = 20%
        self.assertEqual(n_contam(0.05, 95), 5)

    def test_pool_is_defects_balanced_and_deterministic(self):
        pool = contamination_pool(self.data["test"], 10, seed=1)
        self.assertEqual(pool, contamination_pool(self.data["test"], 10, seed=1))
        self.assertTrue(all(self.data["test"][i]["label"] == 1 for i in pool))
        kinds = [self.data["test"][i]["dtype"] for i in pool]
        self.assertLessEqual(abs(kinds.count("a") - kinds.count("b")), 1)

    def test_tn1_split_nested_and_test_fixed(self):
        _, pool_size = tn1_ratios_for(self.data, [0.05, 0.1, 0.2], 1)
        splits = {r: tn1_split(self.data, r, seed=0, pool_size=pool_size) for r in (0.05, 0.1, 0.2)}
        tests = [s[1] for s in splits.values()]
        self.assertEqual(tests[0], tests[1])                 # cùng seed thì test giống hệt ở mọi tỷ lệ
        self.assertEqual(tests[1], tests[2])
        extra = {r: {i for s, i in tr if s == "test"} for r, (tr, _) in splits.items()}
        self.assertLessEqual(extra[0.05], extra[0.1])
        self.assertLessEqual(extra[0.1], extra[0.2])
        for r, (_, te) in splits.items():
            self.assertFalse(extra[r] & set(te))              # ảnh lỗi đã vào train không còn trong test

    def test_tn1_ratios_drop_when_test_runs_out(self):
        ok, _ = tn1_ratios_for(self.data, [0.05, 0.5, 0.9], min_test_defects=5)
        self.assertIn(0.05, ok)
        self.assertNotIn(0.9, ok)

    def test_tn2_split(self):
        tr, te = tn2_split(self.data, 4, seed=3)
        self.assertEqual(len(tr), 4)
        self.assertEqual(tr, tn2_split(self.data, 4, seed=3)[0])
        self.assertEqual(len(te), 40)
        self.assertEqual(len(tn2_split(self.data, "all", seed=0)[0]), 20)

    def test_split_fit_val_disjoint_and_complete(self):
        paths = [f"p{i}" for i in range(40)]
        fit, val = split_fit_val(paths)
        self.assertEqual(len(val), 6)                         # 15% của 40
        self.assertEqual(sorted(fit + val), sorted(paths))
        self.assertEqual((fit, val), split_fit_val(paths))


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


class ResultsStore(unittest.TestCase):
    def test_load_dedups_per_method_and_key(self):
        """TN1/TN2 dùng chung key giữa các phương pháp: load không được làm mất dòng của phương pháp khác."""
        with tempfile.TemporaryDirectory() as d:
            res = Results(Path(d))
            self.assertTrue(res.load("tn2").empty)
            res.append("tn2", "padim", dict(key="a", method="padim", v=1))
            res.append("tn2", "padim", dict(key="a", method="padim", v=2))
            res.append("tn2", "ae", dict(key="a", method="ae", v=3))
            df = res.load("tn2")
            self.assertEqual(sorted(df.method), ["ae", "padim"])
            self.assertEqual(int(df[df.method == "padim"].v.iloc[0]), 2)     # giữ lần chạy cuối
            self.assertEqual(len(res.load("tn2", dedup=False)), 3)


class Shifts(unittest.TestCase):
    def test_geometric_shift_keeps_mask_area_and_rejects_unknown(self):
        _, test, _, masks = fake_data()
        img, mk = apply_shift(test, masks, "translate", 4)
        self.assertEqual(img.shape, test.shape)
        self.assertEqual(mk[1].sum(), masks[1].sum())
        same, _ = apply_shift(test, masks, "none", 0)
        self.assertIs(same, test)
        with self.assertRaises(ValueError):
            apply_shift(test, masks, "khong_co", 1)


class MethodsInterface(unittest.TestCase):
    """fit/predict của ba phương pháp trên ảnh giả: map cùng kích thước ảnh, ảnh lỗi điểm cao hơn ảnh tốt."""

    def setUp(self):
        self.imgs_fit, self.imgs_te, self.labels, self.masks = fake_data()
        for p in (patch.object(methods, "get_backbone", lambda: FakeBackbone()),
                  patch.object(methods, "AE_STEPS", 3),
                  patch.object(methods, "AE_BATCH", 4),
                  patch.object(methods, "DEVICE", "cpu")):          # máy có GPU vẫn chạy kiểm thử trên CPU
            p.start()
            self.addCleanup(p.stop)

    def _maps(self, model):
        maps = methods.predict_method(model, self.imgs_te)
        self.assertEqual(maps.shape, self.imgs_te.shape[:3])
        self.assertEqual(maps.dtype, np.float32)
        self.assertTrue(np.isfinite(maps).all())
        return maps

    def test_padim(self):
        model = methods.fit_method("padim", self.imgs_fit)
        self.assertEqual(model[0], "padim")
        s = image_scores(self._maps(model))
        self.assertGreater(s[self.labels == 1].min(), s[self.labels == 0].max())

    def test_patchcore(self):
        model = methods.fit_method("patchcore", self.imgs_fit)
        self.assertEqual(model[0], "patchcore")
        s = image_scores(self._maps(model))
        self.assertGreater(s[self.labels == 1].min(), s[self.labels == 0].max())

    def test_autoencoder_resizes_input_internally(self):
        model = methods.fit_method("ae", self.imgs_fit)
        self.assertEqual(model[0], "ae")
        self._maps(model)                                     # ảnh 64x64 vào AE ở 256x256, map trả về 64x64

    def test_fit_is_deterministic(self):
        a = methods.predict_method(methods.fit_method("patchcore", self.imgs_fit, seed=1), self.imgs_te)
        b = methods.predict_method(methods.fit_method("patchcore", self.imgs_fit, seed=1), self.imgs_te)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_save_load_roundtrip(self):
        self.assertEqual(set(methods.MODEL_FILES), set(METHODS))
        for method in METHODS:
            model = methods.fit_method(method, self.imgs_fit)
            with tempfile.TemporaryDirectory() as d:
                methods.save_model(model, d)
                self.assertTrue((Path(d) / methods.MODEL_FILES[method]).exists())
                back = methods.load_model(d, method)
            self.assertEqual(back[0], method)
            tol = 5e-2 if method == "patchcore" else 1e-4      # bank PatchCore lưu fp16
            np.testing.assert_allclose(methods.predict_method(model, self.imgs_te),
                                       methods.predict_method(back, self.imgs_te), atol=tol, rtol=1e-3)

    def test_greedy_coreset_picks_distinct_points(self):
        idx = methods.greedy_coreset(torch.randn(500, 1024).half(), 50)
        self.assertEqual(len(set(idx.tolist())), 50)

    def test_default_size_constant(self):
        self.assertEqual(SQUARE, (256, 256))


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
        from PIL import Image
        img = Image.open(self.T.DATA_ROOT / self.row.path)
        for m in METHODS:
            _, amap, score, meta = self.T.predict(img, self.cat, m)
            self.assertEqual(amap.shape, tuple(meta["hw"]))
            self.assertAlmostEqual(score, float(self.row[f"score_{m}"]), delta=0.02 * abs(score) + 1e-3, msg=m)

    def test_app_predict_returns_overlay_and_verdict(self):
        from PIL import Image
        from src.demo import app
        overlay, _, text = app.predict(Image.open(self.T.DATA_ROOT / self.row.path), self.cat, "patchcore")
        self.assertEqual(overlay.size[::-1], tuple(self.T.load_meta(self.cat)["hw"]))
        self.assertTrue(text.startswith("CÓ LỖI") or text.startswith("KHÔNG LỖI"))
        with self.assertRaises(ValueError):
            app.predict(None, self.cat, "ae")


if __name__ == "__main__":
    unittest.main()
