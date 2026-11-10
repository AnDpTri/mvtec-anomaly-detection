"""TN1 (train lẫn ảnh lỗi) và TN2 (few-shot) trên MVTec AD. Đặc trưng của mỗi category trích một lần."""
import time

import numpy as np

from src.tn import data as D
from src.tn.config import free_gpu
from src.tn.experiments.common import gather
from src.tn.methods import fit_detector, run_on_features
from src.tn.metrics import compute_metrics, image_scores
from src.tn.splits import n_contam, tn1_ratios_for, tn1_split, tn2_split


def run_method(method, C, train_items, test_idx, seed):
    """Fit trên các phần tử train, trả anomaly map của các ảnh test `test_idx` (ảnh của category `C`)."""
    test_items = [("test", i) for i in test_idx]
    if method == "ae":
        return fit_detector("ae", gather(C, train_items, "img"), seed=seed).predict(gather(C, test_items, "img"))
    return run_on_features(method, gather(C, train_items, method), gather(C, test_items, method), seed=seed)


def run_once(res, cache, exp, method, cat, key, train_items, test_idx, seed, extra, maps_dir=None):
    C = cache.get(cat)
    t0 = time.time()
    maps = run_method(method, C, train_items, test_idx, seed)
    elapsed = time.time() - t0
    labels = [C["data"]["test"][i]["label"] for i in test_idx]
    scores = image_scores(maps)
    met = compute_metrics(scores, maps, labels, C["masks"][test_idx])
    res.append(exp, method, dict(key=key, category=cat, method=method, seed=seed, n_train=len(train_items),
                                 n_test=len(test_idx), n_test_defect=int(sum(labels)), time_s=round(elapsed, 1),
                                 **extra, **met))
    if maps_dir is not None:
        np.savez_compressed(maps_dir / f"{cat}_{method}.npz", maps=maps.astype(np.float16), scores=scores,
                            test_idx=np.asarray(test_idx))
    free_gpu()
    print(f"  {key:40s} I-AUROC={met['i_auroc']:.3f} P-AUROC={met['p_auroc']:.3f} "
          f"AUPRO={met['aupro']:.3f} ({elapsed:.0f}s)")


def run_tn1(res, cache, cats, methods, ratios, seeds, min_test_defects):
    """Train lẫn ảnh lỗi. Tỷ lệ làm test còn < min_test_defects ảnh lỗi bị bỏ qua (grid không có 20%)."""
    for cat in cats:
        data = D.load_mvtec(cache.data_root, cat)
        ok, pool_size = tn1_ratios_for(data, ratios, min_test_defects)
        print(f"== TN1 [{cat}] tỷ lệ chạy: {ok} | kho ảnh lỗi = {pool_size} ảnh")
        for method in methods:
            done = res.done("tn1", method)
            for seed in seeds:
                for r in ok:
                    key = f"{cat}|r={r}|s={seed}"
                    if key in done:
                        continue
                    train_items, test_idx = tn1_split(data, r, seed, pool_size)
                    run_once(res, cache, "tn1", method, cat, key, train_items, test_idx, seed,
                              extra=dict(ratio=r, n_contam=n_contam(r, len(data["train"]))))


def run_tn2(res, cache, cats, methods, shots, seeds, maps_dir):
    """Few-shot. Mức 'all' chỉ chạy seed 0 và lưu anomaly map cho TN5."""
    for cat in cats:
        data = D.load_mvtec(cache.data_root, cat)
        print(f"== TN2 [{cat}]")
        for method in methods:
            done = res.done("tn2", method)
            for k in shots:
                for seed in (seeds if k != "all" else [0]):
                    key = f"{cat}|k={k}|s={seed}"
                    if key in done:
                        continue
                    train_items, test_idx = tn2_split(data, k, seed)
                    run_once(res, cache, "tn2", method, cat, key, train_items, test_idx, seed,
                              extra=dict(shots=str(k)), maps_dir=maps_dir if k == "all" else None)

