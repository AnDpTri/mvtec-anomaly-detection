"""Chia tập cho TN1 (train lẫn ảnh lỗi), TN2 (few-shot) và tách fit/val để đặt ngưỡng.

Phần tử train là cặp ('train', i) hoặc ('test', i): ảnh thứ i của tập train hoặc test gốc.
"""
import numpy as np

from src.tn.config import VAL_FRAC


def n_contam(ratio, n_good):
    """Số ảnh lỗi để ảnh lỗi chiếm `ratio` của tập train (good + lỗi)."""
    return int(round(ratio * n_good / (1 - ratio))) if ratio > 0 else 0


def contamination_pool(test, pool_size, seed):
    """Trích ảnh lỗi chia đều theo loại lỗi (round-robin), thứ tự cố định theo seed."""
    rng = np.random.default_rng(seed)
    by_type = {}
    for i, t in enumerate(test):
        if t["label"] == 1:
            by_type.setdefault(t["dtype"], []).append(i)
    for k in by_type:
        rng.shuffle(by_type[k])
    order = []
    while any(by_type.values()):
        for k in sorted(by_type):
            if by_type[k]:
                order.append(by_type[k].pop())
    return order[:pool_size]


def tn1_ratios_for(data, ratios, min_test_defects):
    """Các tỷ lệ chạy được (test còn đủ ảnh lỗi) và kích thước kho ảnh lỗi cho tỷ lệ lớn nhất."""
    n_good, n_def = len(data["train"]), sum(t["label"] for t in data["test"])
    ok = [r for r in ratios if n_def - n_contam(r, n_good) >= min_test_defects]
    return ok, max(n_contam(r, n_good) for r in ok)


def tn1_split(data, ratio, seed, pool_size):
    """Kho ảnh lỗi bị bỏ khỏi test ở MỌI tỷ lệ, nên trong cùng seed tập test giống hệt nhau và 5% ⊂ 10% ⊂ 20%."""
    pool = contamination_pool(data["test"], pool_size, seed)
    k = n_contam(ratio, len(data["train"]))
    train = [("train", i) for i in range(len(data["train"]))] + [("test", i) for i in pool[:k]]
    pool_set = set(pool)
    test = [i for i in range(len(data["test"])) if i not in pool_set]
    return train, test


def tn2_split(data, shots, seed):
    n = len(data["train"])
    if shots == "all":
        idx = list(range(n))
    else:
        idx = sorted(np.random.default_rng(seed).choice(n, size=shots, replace=False).tolist())
    return [("train", i) for i in idx], list(range(len(data["test"])))


def split_fit_val(paths, seed=0, frac=VAL_FRAC):
    """-> (đường dẫn để fit, đường dẫn validation). MVTec AD không có validation nên tách từ train/good."""
    perm = np.random.default_rng(seed).permutation(len(paths))
    n_val = int(round(frac * len(perm)))
    return [paths[i] for i in perm[n_val:]], [paths[i] for i in perm[:n_val]]
