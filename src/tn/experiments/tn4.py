"""TN4: domain shift. 4a = MVTec AD 2 (ánh sáng), 4b = shift nhân tạo, 4b_geo = xoay/dịch ở độ phân giải gốc."""
import time

import numpy as np
import pandas as pd

from src.tn import data as D
from src.tn.config import SQUARE, free_gpu, require_gpu
from src.tn.experiments.common import fit_with_threshold
from src.tn.experiments.shifts import apply_shift, geo_shift_fullres, shift_list, suppress_invalid
from src.tn.metrics import image_scores, metrics_full
from src.tn.splits import split_fit_val


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
            model, thr = fit_with_threshold(method, imgs_tr, imgs_val, coreset=coreset)
            maps = model.predict(imgs_te)
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
            model, thr = fit_with_threshold(method, imgs_fit, imgs_val)
            for kind, level in todo:
                te_s, m_s = apply_shift(imgs_te, masks, kind, level)
                maps = model.predict(te_s)
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
            models[method], thrs[method] = fit_with_threshold(method, imgs_fit, imgs_val)
        print(f"[{cat}] train xong {list(models)} trong {time.time() - t0:.0f}s")
        for kind, level in todo:
            te_s, m_s, valid = geo_shift_fullres(data["test"], kind, level, SQUARE, erode)
            vis = np.where(mask_area0 > 0, (m_s & valid).reshape(len(labels), -1).sum(1) / np.maximum(mask_area0, 1), 1.0)
            keep = (labels == 0) | (vis >= min_visible)
            for method in methods:
                key = f"{cat}|{method}|{kind}|{level}"
                if key in done[method]:
                    continue
                maps = suppress_invalid(models[method].predict(te_s), valid)[keep]
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
