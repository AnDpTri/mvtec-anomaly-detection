"""Tải MVTec AD từ bản sao Hugging Face (foersben/mvtec-ad), chỉ dùng thư viện chuẩn của Python.

Cách dùng:
    python tools/download_mvtec.py <thư mục đích> [loại1 loại2 ...]
Không ghi loại nào thì tải cả 15 loại. File đã tải đủ dung lượng sẽ được bỏ qua,
nên chạy lại được khi bị ngắt giữa chừng.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "foersben/mvtec-ad"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def list_files(path):
    """Liệt kê mọi file trong một thư mục của repo, đi theo phân trang của API."""
    files, url = [], API + path + "?recursive=true"
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            files += [f for f in json.load(r) if f["type"] == "file"]
            link = r.headers.get("Link", "")
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return files


def fetch(dest, f):
    out = os.path.join(dest, *f["path"].split("/"))
    if os.path.exists(out) and os.path.getsize(out) == f["size"]:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(RAW + urllib.parse.quote(f["path"]), out + ".part")
            os.replace(out + ".part", out)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"{f['path']}: {e}")
    if os.path.getsize(out) != f["size"]:
        raise RuntimeError(f"Sai dung lượng: {f['path']}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    categories = sys.argv[2:] or ALL_CATEGORIES

    with urllib.request.urlopen(API, timeout=60) as r:
        top = [f for f in json.load(r) if f["path"] in ("license.txt", "readme.txt")]
    for f in top:
        fetch(dest, f)

    for cat in categories:
        files = list_files(cat)
        print(f"{cat}: {len(files)} file, {sum(f['size'] for f in files) / 1e6:.1f} MB", flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda f: fetch(dest, f), files))
    print("Xong.")


if __name__ == "__main__":
    main()
