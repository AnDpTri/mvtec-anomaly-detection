"""Đo và căn góc con vít (TN2b): kiểm chứng giả thuyết điểm bất thường của screw bị góc xoay chi phối."""
import cv2
import numpy as np


def screw_pose(img_u8):
    """-> (góc 0–360° theo hướng mũi -> đầu vít, cx, cy). Đầu vít = đầu rộng hơn của trục chính (PCA)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    d = pts - c
    w, v = np.linalg.eigh(np.cov(d.T))
    axis = v[:, np.argmax(w)]
    t, perp = d @ axis, d @ np.array([-axis[1], axis[0]])
    L = t.max() - t.min()
    if np.abs(perp[t < t.min() + 0.15 * L]).max() > np.abs(perp[t > t.max() - 0.15 * L]).max():
        axis = -axis
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 360), c[0], c[1]


def align_screw(img_u8, mask=None, target=180.0):
    """Xoay + dịch: vít nằm ngang, đầu vít bên trái, tâm vít ở giữa ảnh. Mask đi theo cùng phép biến đổi."""
    ang, cx, cy = screw_pose(img_u8)
    H, W = img_u8.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), ang - target, 0.9)                 # thu nhỏ 10% để mũi vít không bị cắt
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    bg = int(np.median(img_u8[:10, :10]))
    out = cv2.warpAffine(img_u8, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(bg, bg, bg))
    mo = None if mask is None else cv2.warpAffine(mask.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    return out, mo, ang


def circ_dist(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
