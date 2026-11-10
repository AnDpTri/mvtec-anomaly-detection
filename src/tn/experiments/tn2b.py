"""TN2b: few-shot trên screw với ảnh gốc và ảnh đã căn góc (góc xoay chi phối điểm bất thường của screw)."""
import time

import numpy as np
import pandas as pd

from src.tn.config import free_gpu
from src.tn.experiments.screw import align_screw, circ_dist, screw_pose
from src.tn.methods import extract_features, run_on_features
from src.tn.metrics import compute_metrics, image_scores
from src.tn.splits import tn2_split


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

