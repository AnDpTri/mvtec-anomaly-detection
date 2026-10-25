"""[Dưỡng] In bảng kết quả từ results/results.csv: mỗi loại một dòng, mỗi phương pháp một cột.

Cách dùng (từ thư mục gốc dự án):
    python -m src.evaluation.summary
    python -m src.evaluation.summary --metric pixel_auroc
"""
import argparse

import pandas as pd

from src.common.config import ROOT, all_categories, load_config


def summary_table(csv_path, metric="image_auroc"):
    cfg = load_config()
    df = pd.read_csv(ROOT / csv_path)
    df = df.drop_duplicates(subset=["category", "method", "note"], keep="last")
    table = df.pivot_table(index="category", columns="method", values=metric, aggfunc="last")
    order = [c for c in all_categories(cfg) if c in table.index]
    table = table.reindex(order)
    tex = [c for c in cfg["categories"]["texture"] if c in table.index]
    obj = [c for c in cfg["categories"]["object"] if c in table.index]
    rows = {"TB texture": table.loc[tex].mean(), "TB object": table.loc[obj].mean(), "TB 15 loại": table.mean()}
    return pd.concat([table, pd.DataFrame(rows).T])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--metric", default=None, choices=["image_auroc", "pixel_auroc"])
    args = parser.parse_args()
    csv_path = args.csv or load_config()["results_csv"]
    for metric in [args.metric] if args.metric else ["image_auroc", "pixel_auroc"]:
        print(f"\n== {metric}")
        print(summary_table(csv_path, metric).round(4).to_string())


if __name__ == "__main__":
    main()
