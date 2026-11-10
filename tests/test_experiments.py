"""Kiểm thử vòng chạy thí nghiệm trên dữ liệu MVTec AD giả (ảnh PNG nhỏ, backbone giả, CPU)."""
import tempfile
import unittest
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from helpers import fake_data, silence_stdout, use_fake_backbone_on_cpu, write_fake_mvtec
from src.tn import analysis, data as D
from src.tn.config import METHODS
from src.tn.experiments import (CategoryCache, Results, prepare_rotation, run_adapt, run_matrix, run_tn1, run_tn2,
                                run_tn4b)
from src.tn.experiments.common import fit_with_threshold, gather
from src.tn.experiments.screw import align_screw, circ_dist, screw_pose
from src.tn.experiments.shifts import apply_shift, geo_shift_fullres, shift_list, suppress_invalid


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

    def test_brightness_and_noise_are_deterministic(self):
        _, test, _, masks = fake_data()
        b, _ = apply_shift(test, masks, "brightness", 1.2)
        self.assertGreater(b.mean(), test.mean())
        n1, _ = apply_shift(test, masks, "noise", 10)
        n2, _ = apply_shift(test, masks, "noise", 10)
        np.testing.assert_array_equal(n1, n2)

    def test_shift_list_uses_overrides(self):
        shifts = {"rotate": [5, 10], "blur": [1]}
        self.assertEqual(shift_list("grid", shifts, {}), [("rotate", 5), ("rotate", 10), ("blur", 1)])
        self.assertEqual(shift_list("transistor", shifts, {"transistor": {"rotate": [2]}}),
                         [("rotate", 2), ("blur", 1)])
        self.assertEqual(shift_list("grid", shifts, {}, kinds=("rotate",)), [("rotate", 5), ("rotate", 10)])

    def test_geo_shift_fullres_marks_filled_border_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            write_fake_mvtec(d, "c", size=128)
            items = D.load_mvtec(d, "c")["test"][-2:]              # hai ảnh lỗi cuối
            imgs, masks, valid = geo_shift_fullres(items, "translate", 16, (64, 64), erode=2)
            self.assertEqual(imgs.shape, (2, 64, 64, 3))
            self.assertEqual(masks.shape, valid.shape)
            self.assertFalse(valid.all())                           # vùng viền được lấp bị loại
            maps = np.random.default_rng(0).random((2, 64, 64)).astype(np.float32)
            out = suppress_invalid(maps, valid)
            self.assertLessEqual(out.max(), maps.max())
            for i in range(2):
                self.assertEqual(out[i][~valid[i]].max(), out[i][valid[i]].min())


class Screw(unittest.TestCase):
    @staticmethod
    def _screw(angle=0.0, size=160):
        """Vít tối trên nền sáng: thân dài + đầu rộng ở bên phải, xoay `angle` độ quanh tâm."""
        img = np.full((size, size, 3), 220, np.uint8)
        cv2.rectangle(img, (30, 75), (120, 85), (40, 40, 40), -1)         # thân
        cv2.rectangle(img, (120, 65), (135, 95), (40, 40, 40), -1)        # đầu
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=(220, 220, 220))

    def test_pose_follows_rotation(self):
        a0 = screw_pose(self._screw(0))[0]
        a1 = screw_pose(self._screw(30))[0]
        self.assertAlmostEqual(float(circ_dist(a1, a0)), 30, delta=3)

    def test_align_makes_pose_canonical(self):
        out, mask, ang = align_screw(self._screw(50), mask=np.zeros((160, 160), bool))
        self.assertEqual(out.shape, (160, 160, 3))
        self.assertEqual(mask.shape, (160, 160))
        self.assertLess(float(circ_dist(screw_pose(out)[0], 180.0)), 5)

    def test_circ_dist_wraps_around(self):
        self.assertEqual(float(circ_dist(359, 1)), 2.0)
        self.assertEqual(float(circ_dist(10, 350)), 20.0)


class ResultsStore(unittest.TestCase):
    def test_append_done_load_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            res = Results(Path(d) / "kq")
            self.assertEqual(res.done("tn2", "padim"), set())
            self.assertTrue(res.load("tn2").empty)
            res.append("tn2", "padim", dict(key="a", v=1))
            res.append("tn2", "padim", dict(key="a", v=2))
            res.append("tn2", "ae", dict(key="b", v=3))
            self.assertEqual(res.done("tn2", "padim"), {"a"})
            df = res.load("tn2")
            self.assertEqual(sorted(df.key), ["a", "b"])
            self.assertEqual(int(df[df.key == "a"].v.iloc[0]), 2)         # giữ lần chạy cuối
            self.assertEqual(len(res.load("tn2", dedup=False)), 3)
            res.append_rows("chi_tiet", pd.DataFrame(dict(x=[1, 2])))
            res.append_rows("chi_tiet", pd.DataFrame(dict(x=[3])))
            self.assertEqual(len(pd.read_csv(Path(d) / "kq" / "chi_tiet.csv")), 3)


class RunsOnFakeMvtec(unittest.TestCase):
    """Chạy thật các vòng TN1, TN2, TN3, TN4b trên 2 category giả; chạy lại thì không thêm dòng."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "mvtec"
        for i, cat in enumerate(("ca", "cb")):
            write_fake_mvtec(cls.root, cat, seed=i)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        use_fake_backbone_on_cpu(self)
        silence_stdout(self)
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.res = Results(Path(self.out.name) / "results")

    def test_category_cache_and_gather(self):
        cache = CategoryCache(self.root, ["padim", "patchcore"])
        C = cache.get("ca")
        self.assertEqual(C["imgs_tr"].shape[0], 8)
        self.assertEqual(len(C["data"]["test"]), 8)
        self.assertEqual(set(C["feats"]), {"padim", "patchcore"})
        self.assertIs(cache.get("ca"), C)                                   # lần sau lấy từ cache
        items = [("train", 0), ("train", 1), ("test", 3)]
        self.assertEqual(gather(C, items, "img").shape[0], 3)
        self.assertEqual(len(gather(C, items, "padim")), 3)

    def test_fit_with_threshold(self):
        imgs_fit, _, _, _ = fake_data()
        model, thr = fit_with_threshold("patchcore", imgs_fit[:6], imgs_fit[6:])
        self.assertEqual(set(thr), {"thr_img", "thr_pix"})
        self.assertGreaterEqual(thr["thr_img"], 0)
        self.assertEqual(model.name, "patchcore")

    def test_tn1_tn2_write_rows_and_resume(self):
        maps_dir = Path(self.out.name) / "maps"
        maps_dir.mkdir()
        cache = CategoryCache(self.root, list(METHODS))
        run_tn1(self.res, cache, ["ca"], ["patchcore"], [0.0, 0.2], [0], min_test_defects=2)
        run_tn2(self.res, cache, ["ca"], list(METHODS), [1, "all"], [0], maps_dir)
        tn1, tn2 = self.res.load("tn1"), self.res.load("tn2")
        self.assertEqual(sorted(tn1.ratio), [0.0, 0.2])
        self.assertEqual(int(tn1[tn1.ratio == 0.2].n_contam.iloc[0]), 2)
        self.assertEqual(len(tn2), 6)                                       # 3 phương pháp × (k=1, all)
        for col in ("i_auroc", "p_auroc", "aupro", "n_train", "time_s"):
            self.assertIn(col, tn2.columns)
        self.assertEqual({p.name for p in maps_dir.glob("*.npz")}, {f"ca_{m}.npz" for m in METHODS})
        run_tn2(self.res, cache, ["ca"], list(METHODS), [1, "all"], [0], maps_dir)
        self.assertEqual(len(self.res.load("tn2", dedup=False)), 6)         # key đã có thì bỏ qua

    def test_tn3_matrix_and_adapt(self):
        loader = partial(D.load_mvtec, self.root)
        hw = (64, 64)
        run_matrix(self.res, "tn3_x", loader, ["ca", "cb"], list(METHODS), hw=hw)
        mx = self.res.load("tn3_x")
        self.assertEqual(len(mx), 2 * 2 * len(METHODS))
        self.assertEqual(sorted(mx.same.unique()), [0, 1])
        run_adapt(self.res, loader, "ca", ["cb"], [1], [0], ["patchcore"], hw)
        ad = self.res.load("tn3_adapt")
        self.assertEqual(sorted(ad.regime), ["chi_k", "nguon+k"])
        n = ad.set_index("regime").n_train
        self.assertGreater(n["nguon+k"], n["chi_k"])

    def test_tn4b_synthetic_shift(self):
        run_tn4b(self.res, self.root, ["ca"], ["patchcore"], {"brightness": [1.2]}, {})
        sy = self.res.load("tn4_syn")
        self.assertEqual(sorted(sy.kind), ["brightness", "none"])
        self.assertIn("img_fpr_thr", sy.columns)

    def test_analysis_on_results(self):
        loader = partial(D.load_mvtec, self.root)
        run_matrix(self.res, "tn3_x", loader, ["ca", "cb"], ["padim", "patchcore"], hw=(64, 64))
        mx = self.res.load("tn3_x")
        summ = analysis.diagonal_summary(mx)
        self.assertEqual(set(summ.index.get_level_values(1)), {"khác category", "cùng category"})
        with tempfile.TemporaryDirectory() as d:
            analysis.plot_matrices(mx, ["ca", "cb"], "thử", Path(d) / "m.png")
            self.assertTrue((Path(d) / "m.png").exists())

    def test_rotation_prepare_needs_screw(self):
        cache = CategoryCache(self.root, ["padim"])
        with self.assertRaises(Exception):
            prepare_rotation(cache)                                          # không có category 'screw'


class AnalysisTables(unittest.TestCase):
    def setUp(self):
        silence_stdout(self)

    def test_shots_to_x_and_failure_cases(self):
        self.assertEqual(analysis.shots_to_x(["1", "4", "all"]), [0.0, 2.0, 6.0])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.95, 0.05])
        labels = np.array([0, 0, 1, 1, 1, 0])
        fn, fp, thr = analysis.failure_cases(scores, labels, n=1)
        self.assertEqual(fn[0][0], 2)                                       # ảnh lỗi điểm thấp nhất
        self.assertEqual(fp[0][0], 1)                                       # ảnh good điểm cao nhất
        self.assertTrue(np.isfinite(thr))

    def test_worst_case_and_merge_geo(self):
        row = lambda kind, level, auroc: dict(category="c", method="padim", kind=kind, level=level, i_auroc=auroc,
                                              aupro=auroc, img_fpr_thr=0.1)
        sy = pd.DataFrame([row("none", 0, 1.0), row("rotate", 5, 0.9), row("brightness", 0.6, 0.8),
                           row("brightness", 1.5, 0.7)])
        wc = analysis.worst_case(sy).set_index("kind")
        self.assertAlmostEqual(wc.loc["brightness", "d_i_auroc"], -0.3)     # mức xa 1 nhất là 1.5
        geo = pd.DataFrame([row("rotate", 10, 0.5)])
        merged = analysis.merge_geo(sy, geo)
        self.assertEqual(list(merged[merged.kind == "rotate"].level), [10])
        self.assertIs(analysis.merge_geo(sy, geo.iloc[0:0]), sy)


if __name__ == "__main__":
    unittest.main()
