"""Kiểm thử ba phương pháp: fit/predict, lưu/nạp, tương thích định dạng file mô hình đã lưu từ trước."""
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch

from helpers import fake_data, use_fake_backbone_on_cpu
from src.tn.config import METHODS, SQUARE
from src.tn.methods import (DETECTORS, AEDetector, PaDiMDetector, PatchCoreDetector, extract_features, fit_detector,
                            load_detector, run_on_features)
from src.tn.methods.ae import ConvAE
from src.tn.methods.patchcore import greedy_coreset
from src.tn.metrics import image_scores


class Registry(unittest.TestCase):
    def test_detectors_match_config_methods(self):
        self.assertEqual(set(DETECTORS), set(METHODS))
        self.assertEqual(len({d.filename for d in DETECTORS.values()}), len(METHODS))
        for name, det in DETECTORS.items():
            self.assertEqual(det.name, name)

    def test_unknown_method(self):
        with self.assertRaises(KeyError):
            fit_detector("khong_co", np.zeros((1, 8, 8, 3), np.uint8))

    def test_default_size_constant(self):
        self.assertEqual(SQUARE, (256, 256))


class DetectorInterface(unittest.TestCase):
    """fit/predict trên ảnh giả: map cùng kích thước ảnh, ảnh lỗi điểm cao hơn ảnh tốt."""

    def setUp(self):
        self.imgs_fit, self.imgs_te, self.labels, self.masks = fake_data()
        use_fake_backbone_on_cpu(self)

    def _maps(self, model):
        maps = model.predict(self.imgs_te)
        self.assertEqual(maps.shape, self.imgs_te.shape[:3])
        self.assertEqual(maps.dtype, np.float32)
        self.assertTrue(np.isfinite(maps).all())
        return maps

    def test_padim(self):
        model = fit_detector("padim", self.imgs_fit)
        self.assertIsInstance(model, PaDiMDetector)
        s = image_scores(self._maps(model))
        self.assertGreater(s[self.labels == 1].min(), s[self.labels == 0].max())

    def test_patchcore(self):
        model = fit_detector("patchcore", self.imgs_fit)
        self.assertIsInstance(model, PatchCoreDetector)
        s = image_scores(self._maps(model))
        self.assertGreater(s[self.labels == 1].min(), s[self.labels == 0].max())

    def test_autoencoder_resizes_input_internally(self):
        model = fit_detector("ae", self.imgs_fit)
        self.assertIsInstance(model, AEDetector)
        self._maps(model)                                     # ảnh 64x64 vào AE ở 256x256, map trả về 64x64

    def test_fit_is_deterministic(self):
        a = fit_detector("patchcore", self.imgs_fit, seed=1).predict(self.imgs_te)
        b = fit_detector("patchcore", self.imgs_fit, seed=1).predict(self.imgs_te)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_features_path_equals_image_path(self):
        """Fit/predict trên đặc trưng trích sẵn (TN1/TN2) cho đúng kết quả của fit/predict trên ảnh."""
        pad_tr, pc_tr = extract_features(self.imgs_fit)
        pad_te, pc_te = extract_features(self.imgs_te)
        feats = dict(padim=(pad_tr, pad_te), patchcore=(pc_tr, pc_te))
        for method in ("padim", "patchcore"):
            via_img = fit_detector(method, self.imgs_fit).predict(self.imgs_te)
            tr, te = feats[method]
            via_feat = DETECTORS[method].fit_features(tr).predict_features(te, hw=self.imgs_te.shape[1:3])
            np.testing.assert_allclose(via_img, via_feat, atol=1e-5)

    def test_run_on_features_returns_default_size_maps(self):
        pad_tr, _ = extract_features(self.imgs_fit, which=("padim",))
        pad_te, _ = extract_features(self.imgs_te, which=("padim",))
        maps = run_on_features("padim", pad_tr, pad_te)
        self.assertEqual(maps.shape, (len(self.imgs_te), *SQUARE))

    def test_extract_features_shapes_and_which(self):
        pad, pc = extract_features(self.imgs_te, which=("padim",))
        self.assertEqual(tuple(pad.shape), (6, 100, 16, 16))      # ảnh 64 -> lưới H/4
        self.assertIsNone(pc)
        pad, pc = extract_features(self.imgs_te, which=("patchcore",))
        self.assertIsNone(pad)
        self.assertEqual(tuple(pc.shape), (6, 8, 8, 1024))        # lưới H/8

    def test_save_load_roundtrip(self):
        for method in METHODS:
            model = fit_detector(method, self.imgs_fit)
            with tempfile.TemporaryDirectory() as d:
                path = model.save(d)
                self.assertEqual(path, Path(d) / DETECTORS[method].filename)
                self.assertTrue(path.exists())
                back = load_detector(d, method)
            self.assertIsInstance(back, DETECTORS[method])
            tol = 5e-2 if method == "patchcore" else 1e-4      # bank PatchCore lưu fp16
            np.testing.assert_allclose(model.predict(self.imgs_te), back.predict(self.imgs_te), atol=tol, rtol=1e-3)

    def test_greedy_coreset_picks_distinct_points(self):
        idx = greedy_coreset(torch.randn(500, 1024).half(), 50)
        self.assertEqual(len(set(idx.tolist())), 50)

    def test_patchcore_keeps_small_bank_and_subsamples_large_one(self):
        feat = torch.randn(2, 8, 8, 1024).half()                    # 128 patch < PC_CORESET_MIN_BANK
        self.assertEqual(len(PatchCoreDetector.fit_features(feat).bank), 128)
        with unittest.mock.patch("src.tn.config.PC_CORESET_MIN_BANK", 100):
            self.assertEqual(len(PatchCoreDetector.fit_features(feat, coreset=0.5).bank), 64)


class SavedFormatIsStable(unittest.TestCase):
    """File .pt ghi bằng bản mã cũ (state_dict ConvAE, dict mu/inv/H/W, tensor fp16) phải nạp được bằng bản mới.

    Các file này được dựng tay đúng định dạng của results/checkpoints_tn, không đi qua Detector.save.
    """

    def setUp(self):
        self.imgs_fit, self.imgs_te, _, _ = fake_data()
        use_fake_backbone_on_cpu(self)

    def test_legacy_autoencoder_state_dict(self):
        torch.manual_seed(0)
        sd = ConvAE().state_dict()
        for key in ("enc.0.weight", "enc.1.running_mean", "fc1.weight", "fc2.bias", "dec.0.weight", "dec.12.bias"):
            self.assertIn(key, sd, "tên khóa state_dict đổi là hỏng mô hình đã lưu")
        with tempfile.TemporaryDirectory() as d:
            torch.save(sd, Path(d) / "ae.pt")
            model = load_detector(d, "ae")
        maps = model.predict(self.imgs_te)
        self.assertEqual(maps.shape, self.imgs_te.shape[:3])

    def test_legacy_padim_dict(self):
        feat = extract_features(self.imgs_fit, which=("padim",))[0]
        ref = PaDiMDetector.fit_features(feat)
        with tempfile.TemporaryDirectory() as d:
            torch.save(dict(mu=ref.mu.cpu(), inv=ref.inv.cpu(), H=ref.H, W=ref.W), Path(d) / "padim.pt")
            model = load_detector(d, "padim")
        np.testing.assert_allclose(model.predict(self.imgs_te), ref.predict(self.imgs_te), atol=1e-5)

    def test_legacy_patchcore_fp16_bank(self):
        feat = extract_features(self.imgs_fit, which=("patchcore",))[1]
        bank = feat.reshape(-1, 1024)
        with tempfile.TemporaryDirectory() as d:
            torch.save(bank, Path(d) / "patchcore_bank.pt")
            model = load_detector(d, "patchcore")
        self.assertEqual(model.bank.dtype, torch.float32)
        self.assertEqual(tuple(model.bank.shape), tuple(bank.shape))
        self.assertTrue(np.isfinite(model.predict(self.imgs_te)).all())


if __name__ == "__main__":
    unittest.main()
