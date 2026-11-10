"""Vòng chạy TN1, TN2, TN2b, TN3, TN4. Mỗi run ghi một dòng CSV qua `Results`; run đã có key thì bỏ qua.

Key, tên file và cột CSV giữ đúng như bản notebook cũ để đọc tiếp được kết quả đã chạy.
"""
import time

import numpy as np
import pandas as pd
import torch

from src.tn import data as D
from src.tn.config import SQUARE, require_gpu
from src.tn.methods import (extract_features, fit_method, free_gpu, predict_method, run_on_features, ae_fit,
                            ae_predict)
from src.tn.metrics import compute_metrics, image_scores, metrics_full, thresholds_from_val
from src.tn.screw import align_screw, circ_dist, screw_pose
from src.tn.shifts import apply_shift, geo_shift_fullres, shift_list, suppress_invalid
from src.tn.splits import n_contam, split_fit_val, tn1_ratios_for, tn1_split, tn2_split


# ----------------------------------------------------------------- TN1, TN2: đặc trưng trích một lần
class CategoryCache:
    """Ảnh, mask và đặc trưng của MỘT category MVTec AD trong RAM (đổi category thì xóa cái cũ)."""

    def __init__(self, data_root, methods):
        self.data_root, self.methods, self.cur = data_root, methods, {}

    def get(self, cat, feats=True):
        need = feats and any(m in self.methods for m in ("padim", "patchcore"))
        C = self.cur.get(cat)
        if C is not None and (C["feats"] is not None or not need):
            return C
        t0 = time.time()
        data = D.load_mvtec(self.data_root, cat)
        imgs_tr = D.load_imgs(data["train"])
        imgs_te = D.load_imgs([t["path"] for t in data["test"]])
        C = dict(data=data, imgs_tr=imgs_tr, imgs_te=imgs_te, masks=D.load_masks(data["test"]), feats=None)
        if need:
            pad_tr, pc_tr = extract_features(imgs_tr)
            pad_te, pc_te = extract_features(imgs_te)
            C["feats"] = dict(padim=(pad_tr, pad_te), patchcore=(pc_tr, pc_te))
        self.cur = {cat: C}
        print(f"[{cat}] đã nạp {len(imgs_tr)} train / {len(imgs_te)} test trong {time.time() - t0:.0f}s")
        return C


def _gather(C, items, kind):
    """items: list ('train', i) | ('test', i) -> mảng ảnh (kind='img') hoặc đặc trưng của phương pháp `kind`."""
    if kind == "img":
        return np.stack([C["imgs_tr"][i] if s == "train" else C["imgs_te"][i] for s, i in items])
    tr, te = C["feats"][kind]
    idx_tr = [i for s, i in items if s == "train"]
    idx_te = [i for s, i in items if s == "test"]
    return torch.cat([tr[idx_tr]] + ([te[idx_te]] if idx_te else []))


def _run_method(method, C, train_items, test_idx, seed):
    test_items = [("test", i) for i in test_idx]
    if method == "ae":
        return ae_predict(ae_fit(_gather(C, train_items, "img"), seed=seed), _gather(C, test_items, "img"))
    return run_on_features(method, _gather(C, train_items, method), _gather(C, test_items, method), seed=seed)


def _run_once(res, cache, exp, method, cat, key, train_items, test_idx, seed, extra, maps_dir=None):
    C = cache.get(cat)
    t0 = time.time()
    maps = _run_method(method, C, train_items, test_idx, seed)
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
                    _run_once(res, cache, "tn1", method, cat, key, train_items, test_idx, seed,
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
                    _run_once(res, cache, "tn2", method, cat, key, train_items, test_idx, seed,
                              extra=dict(shots=str(k)), maps_dir=maps_dir if k == "all" else None)


# ----------------------------------------------------------------- TN2b: góc xoay của screw
def prepare_rotation(cache):
    """Góc từng con vít, ảnh đã căn góc và đặc trưng của cả hai biến thể ('goc' và 'can_goc')."""
    C = cache.get("screw", feats=False)
    ang_tr = np.array([screw_pose(im)[0] for im in C["imgs_tr"]])
    ang_te = np.array([screw_pose(im)[0] for im in C["imgs_te"]])
    al_tr = np.stack([align_screw(im)[0] for im in C["imgs_tr"]])
    al = [align_screw(im, m) for im, m in zip(C["imgs_te"], C["masks"])]
    al_te, al_masks = np.stack([a[0] for a in al]), np.stack([a[1] for a in al])
    if C["feats"] is not None:
        orig = dict(C["feats"])
    else:
        pt, ct = extract_features(C["imgs_tr"])
        pe, ce = extract_features(C["imgs_te"])
        orig = dict(padim=(pt, pe), patchcore=(ct, ce))
    pt, ct = extract_features(al_tr)
    pe, ce = extract_features(al_te)
    return dict(C=C, ang_tr=ang_tr, ang_te=ang_te, al_tr=al_tr, al_te=al_te,
                variants={"goc": (orig, C["masks"]),
                          "can_goc": (dict(padim=(pt, pe), patchcore=(ct, ce)), al_masks)})


def run_tn2b(res, R, shots, seeds, methods=("padim", "patchcore")):
    """Chạy lại few-shot với đúng các ảnh train của TN2, trên ảnh gốc và ảnh đã căn góc."""
    data = R["C"]["data"]
    labels = np.array([t["label"] for t in data["test"]])
    done = res.done("tn2rot", "all")
    for variant, (feats, masks_v) in R["variants"].items():
        for method in methods:
            tr_f, te_f = feats[method]
            for k in shots:
                for seed in (seeds if k != "all" else [0]):
                    key = f"{variant}|{method}|k={k}|s={seed}"
                    if key in done:
                        continue
                    train_items, test_idx = tn2_split(data, k, seed)
                    tr_idx = [i for _, i in train_items]
                    t0 = time.time()
                    maps = run_on_features(method, tr_f[tr_idx], te_f[test_idx], seed=seed)
                    scores = image_scores(maps)
                    met = compute_metrics(scores, maps, labels[test_idx], masks_v[test_idx])
                    dmin = circ_dist(R["ang_te"][test_idx][:, None], R["ang_tr"][tr_idx][None, :]).min(1)
                    res.append_rows("tn2rot_scores", pd.DataFrame(dict(
                        key=key, variant=variant, method=method, shots=str(k), seed=seed, test_idx=test_idx,
                        label=labels[test_idx], dtype=[data["test"][i]["dtype"] for i in test_idx],
                        angle=R["ang_te"][test_idx].round(1), angle_dist=dmin.round(1), score=scores)))
                    res.append("tn2rot", "all", dict(key=key, variant=variant, method=method, shots=str(k), seed=seed,
                                                     time_s=round(time.time() - t0, 1), **met))
                    free_gpu()
                    print(f"  {key:32s} I-AUROC={met['i_auroc']:.3f} AUPRO={met['aupro']:.3f}")


# ----------------------------------------------------------------- TN3: đổi loại sản phẩm
class TestSet:
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
    tests = TestSet(loader, hw)
    for src in cats:
        for method in methods:
            done = res.done(exp, method)
            todo = [tgt for tgt in cats if f"{src}->{tgt}|{method}" not in done]
            if not todo:
                continue
            t0 = time.time()
            fit_p, val_p = split_fit_val(loader(src)["train"])
            model = fit_method(method, D.load_imgs(fit_p, hw), coreset=coreset)
            thr = thresholds_from_val(predict_method(model, D.load_imgs(val_p, hw)))
            for tgt in todo:
                T = tests.get(tgt)
                maps = predict_method(model, T["imgs"])
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
    tests = TestSet(loader, hw)
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
                        model = fit_method(method, train, seed=seed, coreset=coreset)
                        thr = thresholds_from_val(predict_method(model, imgs_tval))
                        maps = predict_method(model, T["imgs"])
                        scores = image_scores(maps)
                        met = metrics_full(scores, maps, T["labels"], T["masks"], thr)
                        res.append(exp, method, dict(key=key, source=source, target=tgt, regime=regime, k=k,
                                                     seed=seed, method=method, n_train=len(train), **thr, **met))
                        print(f"  {method:9s} {source}->{tgt} {regime:8s} k={k:<3d} I-AUROC={met['i_auroc']:.3f} "
                              f"AUPRO={met['aupro']:.3f} ({time.time() - t0:.0f}s)")
                        del model, train
                        free_gpu()


# ----------------------------------------------------------------- TN4: domain shift
def run_tn4a(res, ad2_root, cats_hw, methods, coreset, maps_dir):
    """MVTec AD 2: train ở ánh sáng thường, ngưỡng từ validation/good, kết quả tách theo điều kiện chụp."""
    for cat, hw in cats_hw.items():
        todo = [m for m in methods if f"{cat}|{m}|all" not in res.done("tn4_ad2", m)]
        if not todo:
            print(f"[{cat}] đã xong")
            continue
        d = D.load_ad2(ad2_root, cat)
        t0 = time.time()
        imgs_tr, imgs_val = D.load_imgs(d["train"], hw), D.load_imgs(d["val"], hw)
        imgs_te = D.load_imgs([t["path"] for t in d["test"]], hw)
        masks = D.load_masks(d["test"], hw)
        labels = np.array([t["label"] for t in d["test"]])
        conds = np.array([t["cond"] for t in d["test"]])
        print(f"[{cat}] {hw[0]}x{hw[1]} | train={len(imgs_tr)} val={len(imgs_val)} test_public={len(imgs_te)} "
              f"| điều kiện: {sorted(set(conds.tolist()))} ({time.time() - t0:.0f}s)")
        for method in todo:
            t0 = time.time()
            model = fit_method(method, imgs_tr, coreset=coreset)
            thr = thresholds_from_val(predict_method(model, imgs_val))
            maps = predict_method(model, imgs_te)
            scores = image_scores(maps)
            np.savez_compressed(maps_dir / f"ad2_{cat}_{method}.npz", maps=maps.astype(np.float16), scores=scores)
            res.append_rows("tn4_ad2_scores", pd.DataFrame(dict(
                category=cat, method=method, cond=conds, scene=[t["scene"] for t in d["test"]],
                label=labels, score=scores, thr_img=thr["thr_img"])))
            for c in sorted(set(conds.tolist())) + ["all"]:
                idx = np.arange(len(labels)) if c == "all" else np.where(conds == c)[0]
                met = metrics_full(scores[idx], maps[idx], labels[idx], masks[idx], thr)
                res.append("tn4_ad2", method, dict(key=f"{cat}|{method}|{c}", category=cat, method=method, cond=c,
                                                   n_good=int((labels[idx] == 0).sum()), n_bad=int(labels[idx].sum()),
                                                   **thr, **met))
            r = metrics_full(scores, maps, labels, masks, thr)
            print(f"  {method:9s} I-AUROC(all)={r['i_auroc']:.3f} AUPRO={r['aupro']:.3f} "
                  f"FPR@ngưỡng={r['img_fpr_thr']:.2f} TPR@ngưỡng={r['img_tpr_thr']:.2f} ({time.time() - t0:.0f}s)")
            del model, maps
            free_gpu()


def _fit_val_test(data_root, cat):
    data = D.load_mvtec(data_root, cat)
    fit_p, val_p = split_fit_val(data["train"])
    return data, D.load_imgs(fit_p), D.load_imgs(val_p)


def run_tn4b(res, data_root, cats, methods, shifts, overrides):
    """Shift nhân tạo trên ảnh test 256 px (bản cũ cho xoay/dịch; xoay/dịch đúng cách ở run_tn4b_geo)."""
    for cat in cats:
        todo_shifts = [("none", 0)] + shift_list(cat, shifts, overrides)
        todo_m = {m: [(k, l) for k, l in todo_shifts if f"{cat}|{m}|{k}|{l}" not in res.done("tn4_syn", m)]
                  for m in methods}
        if not any(todo_m.values()):
            print(f"[{cat}] đã xong")
            continue
        data, imgs_fit, imgs_val = _fit_val_test(data_root, cat)
        imgs_te, masks = D.load_imgs([t["path"] for t in data["test"]]), D.load_masks(data["test"])
        labels = np.array([t["label"] for t in data["test"]])
        print(f"[{cat}] fit={len(imgs_fit)} val={len(imgs_val)} test={len(imgs_te)} | {len(todo_shifts)} điều kiện")
        for method, todo in todo_m.items():
            if not todo:
                continue
            t0 = time.time()
            model = fit_method(method, imgs_fit)
            thr = thresholds_from_val(predict_method(model, imgs_val))
            for kind, level in todo:
                te_s, m_s = apply_shift(imgs_te, masks, kind, level)
                maps = predict_method(model, te_s)
                scores = image_scores(maps)
                met = metrics_full(scores, maps, labels, m_s, thr)
                res.append("tn4_syn", method, dict(key=f"{cat}|{method}|{kind}|{level}", category=cat, method=method,
                                                   kind=kind, level=level, **thr, **met))
                print(f"  {method:9s} {kind:10s} {level:>4}  I-AUROC={met['i_auroc']:.3f} AUPRO={met['aupro']:.3f} "
                      f"FPR@ngưỡng={met['img_fpr_thr']:.2f}")
            print(f"  -> {method} xong trong {time.time() - t0:.0f}s")
            del model
            free_gpu()


def run_tn4b_geo(res, data_root, cats, methods, shifts, overrides, erode=32, min_visible=0.5,
                 exp="tn4_syn_geo2"):
    """Xoay/dịch ở độ phân giải gốc, loại vùng viền khỏi chấm điểm, bỏ ảnh lỗi mà vùng lỗi bị đẩy ra ngoài."""
    require_gpu()
    for cat in cats:
        geo = shift_list(cat, shifts, overrides, kinds=("rotate", "translate"))
        done = {m: res.done(exp, m) for m in methods}
        todo = [(k, l) for k, l in geo if any(f"{cat}|{m}|{k}|{l}" not in done[m] for m in methods)]
        if not todo:
            print(f"[{cat}] đã xong")
            continue
        t0 = time.time()
        data, imgs_fit, imgs_val = _fit_val_test(data_root, cat)
        labels = np.array([t["label"] for t in data["test"]])
        mask_area0 = D.load_masks(data["test"]).reshape(len(labels), -1).sum(1)
        models, thrs = {}, {}
        for method in methods:                                   # train MỘT lần, cùng cách chia fit/val như TN4b
            models[method] = fit_method(method, imgs_fit)
            thrs[method] = thresholds_from_val(predict_method(models[method], imgs_val))
        print(f"[{cat}] train xong {list(models)} trong {time.time() - t0:.0f}s")
        for kind, level in todo:
            te_s, m_s, valid = geo_shift_fullres(data["test"], kind, level, SQUARE, erode)
            vis = np.where(mask_area0 > 0, (m_s & valid).reshape(len(labels), -1).sum(1) / np.maximum(mask_area0, 1), 1.0)
            keep = (labels == 0) | (vis >= min_visible)
            for method in methods:
                key = f"{cat}|{method}|{kind}|{level}"
                if key in done[method]:
                    continue
                maps = suppress_invalid(predict_method(models[method], te_s), valid)[keep]
                scores = image_scores(maps)
                met = metrics_full(scores, maps, labels[keep], (m_s & valid)[keep], thrs[method])
                res.append(exp, method, dict(key=key, category=cat, method=method, kind=kind, level=level,
                                             valid_frac=round(float(valid.mean()), 3), n_excluded=int((~keep).sum()),
                                             **thrs[method], **met))
                print(f"  {method:9s} {kind:10s} {level:>4}  I-AUROC={met['i_auroc']:.3f} AUPRO={met['aupro']:.3f} "
                      f"FPR@ngưỡng={met['img_fpr_thr']:.2f} (hợp lệ {valid.mean():.0%}, loại {(~keep).sum()} ảnh lỗi)")
        del models
        free_gpu()
        print(f"[{cat}] xong trong {time.time() - t0:.0f}s")
