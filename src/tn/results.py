"""Ghi kết quả theo từng run vào results/<exp>_<method>.csv; chạy lại thì bỏ qua các key đã có."""
from pathlib import Path

import pandas as pd

from src.tn.config import METHODS


class Results:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, exp, method):
        return self.root / f"{exp}_{method}.csv"

    def done(self, exp, method):
        p = self.path(exp, method)
        return set(pd.read_csv(p)["key"]) if p.exists() else set()

    def append(self, exp, method, row):
        p = self.path(exp, method)
        pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)

    def append_rows(self, name, df):
        """Ghi nối một bảng chi tiết (vd. điểm từng ảnh) vào results/<name>.csv."""
        p = self.root / f"{name}.csv"
        df.to_csv(p, mode="a", header=not p.exists(), index=False)

    def load(self, exp, dedup=True):
        """Gộp CSV của cả 3 phương pháp; dedup giữ lần chạy cuối của mỗi key."""
        dfs = [pd.read_csv(self.path(exp, m)) for m in METHODS if self.path(exp, m).exists()]
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        return df.drop_duplicates("key", keep="last") if dedup else df
