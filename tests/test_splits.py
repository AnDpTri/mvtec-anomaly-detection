"""Kiểm thử chia tập: TN1 (trộn ảnh lỗi), TN2 (few-shot), tách fit/val."""
import unittest

from src.tn.splits import contamination_pool, n_contam, split_fit_val, tn1_ratios_for, tn1_split, tn2_split


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


if __name__ == "__main__":
    unittest.main()
