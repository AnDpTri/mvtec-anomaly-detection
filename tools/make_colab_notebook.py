"""Sinh goi_colab/build_tn_models_colab.ipynb: notebook Colab fit mô hình demo cho 9 loại.

Code (src/tn, tools/build_tn_models.py, tools/download_mvtec.py) được nén và nhúng thẳng vào notebook,
nên chỉ cần tải một file .ipynb lên Colab. Sửa code thì chạy lại script này:  python tools/make_colab_notebook.py
"""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "goi_colab" / "build_tn_models_colab.ipynb"
FILES = [ROOT / "src" / "__init__.py", ROOT / "tools" / "build_tn_models.py", ROOT / "tools" / "download_mvtec.py",
         *sorted((ROOT / "src" / "tn").glob("*.py"))]


def code_b64():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(f, f.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(keepends=True))


def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=s.strip("\n").splitlines(keepends=True))


CELLS = [
    md("""
# Fit mô hình cho demo: 9 loại × 3 phương pháp (Google Colab)

Fit Autoencoder, PaDiM, PatchCore cho 9 loại của thí nghiệm TN1–TN5, cùng cấu hình với `src/tn`:

| Bộ | Loại | Kích thước | Coreset | Ngưỡng |
|---|---|---|---|---|
| MVTec AD | grid, screw, transistor | 256×256 | 10% | 15% train/good tách riêng |
| MVTec AD 2 | vial, sheet_metal | 256×192, 256×1024 | 1% | validation/good |
| VisA | pcb1–pcb4 | 256×320 | 1% | 15% train tách riêng |

Ngưỡng = trung bình + 3 độ lệch chuẩn trên ảnh good validation.

Code đã nhúng sẵn trong notebook (sinh bằng `tools/make_colab_notebook.py`). Chọn *Runtime → Change runtime type → T4 GPU*, rồi *Run all*.
Kết quả ghi thẳng lên `MyDrive/goi_colab/checkpoints_tn/<loại>/`. Colab ngắt giữa chừng thì chạy lại: loại đã xong được bỏ qua.
"""),
    md("## 1. Drive, code, GPU"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import base64, io, sys, zipfile
from pathlib import Path
PKG = Path('/content/drive/MyDrive/goi_colab')
PKG.mkdir(parents=True, exist_ok=True)
CODE_ZIP = '@CODE@'
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall('/content/repo')   # src/tn + tools/
sys.path.insert(0, '/content/repo')

from src.tn.config import require_gpu, check_gpu
check_gpu(); require_gpu()
"""),
    md("""
## 2. Tải dữ liệu (~10 phút)
Tải về ổ local của Colab: MVTec AD 3 loại từ bản sao Hugging Face (~760 MB; link tar.xz cũ của MVTec đã chết), MVTec AD 2 vial + sheet_metal từ trang MVTec (~2,5 GB), VisA từ AWS (~1,9 GB).
"""),
    code("""
from src.tn import data as D
DATA = Path('/content/datasets')
!python /content/repo/tools/download_mvtec.py {DATA}/MVTecAD grid screw transistor
D.prepare_ad2(DATA / 'MVTecAD2', ['vial', 'sheet_metal'])
D.prepare_visa(DATA / 'VisA', ['pcb1', 'pcb2', 'pcb3', 'pcb4'])
"""),
    md("## 3. Fit 9 loại (~25 phút trên T4)"),
    code("""
OUT = PKG / 'checkpoints_tn'
!python /content/repo/tools/build_tn_models.py --data {DATA} --out {OUT}
"""),
    md("## 4. Kiểm tra kết quả"),
    code("""
import json, pandas as pd
rows = []
for d in sorted(OUT.iterdir()):
    if (d / 'meta.json').exists():
        m = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        for meth, met in m['metrics'].items():
            rows.append(dict(loai=d.name, pp=meth, **met))
    else:
        print('CHƯA XONG:', d.name)
pd.DataFrame(rows).pivot_table(index='loai', columns='pp', values=['i_auroc', 'aupro', 'img_fpr_thr']).round(3)
"""),
    code("""
!du -sh {OUT}/*
"""),
]

if __name__ == "__main__":
    b64 = code_b64()
    cells = json.loads(json.dumps(CELLS).replace("@CODE@", b64))
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
                            kernelspec=dict(name="python3", display_name="Python 3"),
                            language_info=dict(name="python")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"đã ghi {OUT.relative_to(ROOT)} ({len(FILES)} file code nhúng)")
