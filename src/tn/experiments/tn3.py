"""TN3: đổi loại sản phẩm (ma trận nguồn -> đích) và thích nghi sang bo mới bằng vài ảnh good của bo đích."""
import time

import numpy as np

from src.tn import data as D
from src.tn.config import SQUARE, free_gpu
from src.tn.experiments.common import fit_with_threshold
from src.tn.metrics import image_scores, metrics_full
from src.tn.splits import split_fit_val


class TargetCache:
    """Ảnh test + mask + nhãn của các category đích, giữ trong RAM để dùng lại cho nhiều nguồn."""

    def __init__(self, loader, hw):
        self.loader, self.hw, self.cache = loader, hw, {}

    def get(self, cat):
        if cat not in self.cache:
            d = self.loader(cat)
            self.cache[cat] = dict(imgs=D.load_imgs([t["path"] for t in d["test"]], self.hw),
                                   masks=D.load_masks(d["test"], self.hw, thr=0),   # 0/1 (VisA) lẫn 0/255 (MVTec)
                                   labels=np.array([t["label"] for t in d["test"]]),
                                   dtypes=[t["dtype"] for t in d["test"]])
        return self.cache[cat]


def run_matrix(res, exp, loader, cats, methods, hw=SQUARE, coreset=None):
    """Train trên từng category NGUỒN, test trên test set của MỌI category ĐÍCH; ngưỡng lấy từ val của NGUỒN."""
    tests = TargetCache(loader, hw)
    for src in cats:
        for method in methods:
            done = res.done(exp, method)
            todo = [tgt for tgt in cats if f"{src}->{tgt}|{method}" not in done]
            if not todo:
                continue
            t0 = time.time()
            fit_p, val_p = split_fit_val(loader(src)["train"])
            model, thr = fit_with_threshold(method, D.load_imgs(fit_p, hw), D.load_imgs(val_p, hw), coreset=coreset)
            for tgt in todo:
                T = tests.get(tgt)
                maps = model.predict(T["imgs"])
                scores = image_scores(maps)
                met = metrics_full(scores, maps, T["labels"], T["masks"], thr)
                res.append(exp, method, dict(key=f"{src}->{tgt}|{method}", source=src, target=tgt, method=method,
                                             same=int(src == tgt), n_fit=len(fit_p), **thr, **met))
                print(f"  {method:9s} {src:>10s} -> {tgt:<10s} I-AUROC={met['i_auroc']:.3f} "
                      f"AUPRO={met['aupro']:.3f} FPR@ngưỡng nguồn={met['img_fpr_thr']:.2f}")
            print(f"  -> {method} (nguồn {src}) xong trong {time.time() - t0:.0f}s")
            del model
            free_gpu()


def run_adapt(res, loader, source, targets, ks, seeds, methods, hw, coreset=None, exp="tn3_adapt"):
    """Thêm k ảnh good của bo đích: 'nguon+k' (cùng ảnh nguồn) so với 'chi_k'. Ngưỡng từ val của ĐÍCH."""
    tests = TargetCache(loader, hw)
    src_fit, _ = split_fit_val(loader(source)["train"])
    imgs_src = None
    for tgt in targets:
        T = tests.get(tgt)
        tgt_fit, tgt_val = split_fit_val(loader(tgt)["train"])
        imgs_tval = D.load_imgs(tgt_val, hw)
        for seed in seeds:
            for k in ks:
                pick = [tgt_fit[i] for i in np.random.default_rng(seed).choice(len(tgt_fit), k, replace=False)]
                imgs_k = D.load_imgs(pick, hw)
                for regime in ["nguon+k", "chi_k"]:
                    for method in methods:
                        key = f"{source}->{tgt}|{regime}|k={k}|s={seed}|{method}"
                        if key in res.done(exp, method):
                            continue
                        t0 = time.time()
                        if regime == "nguon+k":
                            if imgs_src is None:
                                imgs_src = D.load_imgs(src_fit, hw)
                            train = np.concatenate([imgs_src, imgs_k])
                        else:
                            train = imgs_k
                        model, thr = fit_with_threshold(method, train, imgs_tval, seed=seed, coreset=coreset)
                        maps = model.predict(T["imgs"])
                        scores = image_scores(maps)
                        met = metrics_full(scores, maps, T["labels"], T["masks"], thr)
                        res.append(exp, method, dict(key=key, source=source, target=tgt, regime=regime, k=k,
                                                     seed=seed, method=method, n_train=len(train), **thr, **met))
                        print(f"  {method:9s} {source}->{tgt} {regime:8s} k={k:<3d} I-AUROC={met['i_auroc']:.3f} "
                              f"AUPRO={met['aupro']:.3f} ({time.time() - t0:.0f}s)")
                        del model, train
                        free_gpu()

