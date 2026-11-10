"""Shift nhân tạo cho TN4b: độ sáng, nhiễu, làm mờ (trên ảnh 256) và xoay/dịch (ở độ phân giải gốc)."""
import cv2
import numpy as np


def apply_shift(imgs: np.ndarray, masks: np.ndarray, kind: str, level: float, seed: int = 0):
    """-> (ảnh, mask) đã biến đổi. Mask chỉ đổi theo các biến đổi hình học."""
    if kind == "none":
        return imgs, masks
    out, mout = np.empty_like(imgs), masks.copy()
    H, W = imgs.shape[1:3]
    for i in range(len(imgs)):
        im = imgs[i]
        if kind == "brightness":
            out[i] = np.clip(im.astype(np.float32) * level, 0, 255).astype(np.uint8)
        elif kind == "noise":
            r = np.random.default_rng(seed * 100003 + i)
            out[i] = np.clip(im.astype(np.float32) + r.normal(0, level, im.shape), 0, 255).astype(np.uint8)
        elif kind == "blur":
            out[i] = cv2.GaussianBlur(im, (0, 0), level)
        elif kind in ("rotate", "translate"):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), level, 1.0) if kind == "rotate" \
                else np.float32([[1, 0, level], [0, 1, level]])
            # lặp pixel mép (phản chiếu có thể nhân đôi lỗi sát mép mà mask không có)
            out[i] = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mout[i] = cv2.warpAffine(masks[i].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        else:
            raise ValueError(kind)
    return out, mout


def shift_list(cat: str, shifts: dict, overrides: dict, kinds=None) -> list[tuple]:
    """Danh sách (kiểu, mức) cho một category; `overrides` thay mức riêng (transistor chỉ xoay/dịch nhỏ)."""
    sh = {k: list(v) for k, v in shifts.items() if kinds is None or k in kinds}
    sh.update({k: v for k, v in overrides.get(cat, {}).items() if kinds is None or k in kinds})
    return [(k, l) for k, ls in sh.items() for l in ls]


def geo_shift_fullres(items: list, kind: str, level: float, hw, erode: int = 32):
    """Xoay/dịch ở độ phân giải gốc rồi mới thu về `hw` (không mờ do nội suy).

    -> (ảnh, mask, vùng hợp lệ). Vùng được lấp viền bị loại, co thêm `erode` px vì trường tiếp nhận của CNN;
    `level` của phép dịch tính theo pixel ảnh đã thu nhỏ.
    """
    H, W = hw
    imgs, masks, valids = [], [], []
    for t in items:
        im = cv2.cvtColor(cv2.imread(t["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]
        mk = cv2.imread(t["mask"], cv2.IMREAD_GRAYSCALE) if t["mask"] else np.zeros((h0, w0), np.uint8)
        M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), level, 1.0) if kind == "rotate" \
            else np.float32([[1, 0, level * w0 / W], [0, 1, level * h0 / H]])
        im = cv2.warpAffine(im, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mk = cv2.warpAffine(mk, M, (w0, h0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        va = cv2.warpAffine(np.full((h0, w0), 255, np.uint8), M, (w0, h0), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        imgs.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 127)
        va = cv2.resize(va, (W, H), interpolation=cv2.INTER_NEAREST)
        valids.append(cv2.erode(va, np.ones((2 * erode + 1,) * 2, np.uint8)) > 127)   # mép ảnh thật không bị co
    return np.stack(imgs), np.stack(masks), np.stack(valids)


def suppress_invalid(maps: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Vùng không hợp lệ nhận giá trị thấp nhất của vùng hợp lệ trong cùng ảnh -> không ảnh hưởng điểm max."""
    out = maps.copy()
    for i in range(len(out)):
        if not valid[i].all():
            out[i][~valid[i]] = out[i][valid[i]].min()
    return out
