"""Sinh bảng LaTeX trực tiếp từ CSV thực nghiệm, không sửa số liệu gốc."""

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source = ROOT / "results/results.csv"
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {(r["method"], r["category"]) for r in rows}
    if len(rows) != 45 or len(keys) != 45:
        raise ValueError("Cần đúng 45 dòng riêng: 3 phương pháp × 15 loại.")
    methods = ["ae", "padim", "patchcore"]
    if {r["method"] for r in rows} != set(methods):
        raise ValueError("Tên phương pháp không khớp.")
    for r in rows:
        for metric in ["image_auroc", "pixel_auroc"]:
            if not 0 <= float(r[metric]) <= 1:
                raise ValueError("AUROC nằm ngoài [0,1].")
    tables = []
    aggregates = {}
    for metric, title in [
        ("image_auroc", "Image-AUROC"),
        ("pixel_auroc", "Pixel-AUROC"),
    ]:
        tables += [
            r"\begin{table}[htbp]\centering",
            r"\caption{" + title + r" theo từng loại (kết quả Colab gốc).}",
            r"\begin{tabular}{@{}lrrr@{}}\toprule Loại & AE & PaDiM & PatchCore \\ \midrule",
        ]
        cats = [r["category"] for r in rows if r["method"] == "ae"]
        for category in cats:
            values = [
                float(
                    next(
                        r
                        for r in rows
                        if r["category"] == category and r["method"] == m
                    )[metric]
                )
                for m in methods
            ]
            cells = [
                (r"\textbf{" + f"{v:.3f}" + "}") if v == max(values) else f"{v:.3f}"
                for v in values
            ]
            tables.append(
                category.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
            )
        aggregates[metric] = {}
        for group, types in [
            ("Texture", {"texture"}),
            ("Object", {"object"}),
            ("Tất cả", {"texture", "object"}),
        ]:
            vals = [
                statistics.mean(
                    float(r[metric])
                    for r in rows
                    if r["method"] == m and r["type"] in types
                )
                for m in methods
            ]
            aggregates[metric][group] = dict(zip(methods, vals))
            tables.append(
                (r"\midrule " if group == "Texture" else "")
                + group
                + " & "
                + " & ".join(f"{v:.3f}" for v in vals)
                + r" \\"
            )
        tables += [r"\bottomrule\end{tabular}\end{table}"]
    destination = ROOT / "bao_cao/generated"
    destination.mkdir(exist_ok=True)
    (destination / "results_tables.tex").write_text(
        "\n".join(tables) + "\n", encoding="utf-8"
    )
    summary = {
        "source": "results/results.csv",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(rows),
        "source_run": "2026-10-25 Colab Tesla T4; preserved imported CSV",
        "aggregates": aggregates,
        "timing_limitation": "AE includes data loading; PaDiM/PatchCore model-only; not directly comparable",
    }
    (ROOT / "docs/results_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
