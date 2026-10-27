"""Biến trích dẫn trong bản Word thành trường Zotero, để sửa và cập nhật nguồn bằng plugin Zotero trong Word.

Mỗi trích dẫn [n] là một trường "ADDIN ZOTERO_ITEM CSL_CITATION" chứa dữ liệu nguồn (CSL-JSON từ references.bib);
danh mục tài liệu tham khảo là trường "ADDIN ZOTERO_BIBL". Nếu Zotero trên máy đã có các nguồn này (nhập từ
references.bib), mỗi trích dẫn còn mang liên kết tới đúng mục trong thư viện; nếu chưa, Zotero vẫn đọc được nhờ dữ liệu
nhúng sẵn. Kiểu trích dẫn đặt sẵn là IEEE.
"""
import json
import random
import re
import shutil
import sqlite3
import string
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

TOKEN = "ZZCITE{}ZZ"
BIB_TOKEN = "ZZBIBLIOGRAPHYZZ"
SCHEMA = "https://github.com/citation-style-language/schema/raw/master/csl-citation.json"


def mark_citations(tex):
    """Thay \\cite{a,b} bằng mã giữ chỗ; trả về (tex, danh sách các nhóm khóa theo thứ tự xuất hiện)."""
    groups = []

    def repl(m):
        groups.append([k.strip() for k in m.group(1).split(",")])
        return TOKEN.format(len(groups) - 1)
    tex = re.sub(r"\\cite\{([^}]+)\}", repl, tex)
    tex = tex.replace("\\printbibliography", BIB_TOKEN)
    return tex, groups


def numbering(groups):
    """Số thứ tự kiểu IEEE: theo lần xuất hiện đầu tiên."""
    order = []
    for g in groups:
        for k in g:
            if k not in order:
                order.append(k)
    return order, {k: i + 1 for i, k in enumerate(order)}


def library_links(titles):
    """Tìm các nguồn trong thư viện Zotero cục bộ theo tiêu đề; trả về {tiêu đề thường: (itemID, uri)}."""
    db = Path.home() / "Zotero" / "zotero.sqlite"
    if not db.exists():
        return {}
    with tempfile.TemporaryDirectory() as tmp:  # Zotero đang mở thì tệp bị khóa: đọc bản sao
        copy = Path(tmp) / "z.sqlite"
        shutil.copy(db, copy)
        wal = db.with_name(db.name + "-wal")  # thay đổi mới nhất của Zotero nằm ở tệp -wal
        if wal.exists():
            shutil.copy(wal, copy.with_name(copy.name + "-wal"))
        con = sqlite3.connect(copy)
        user = con.execute("SELECT value FROM settings WHERE setting='account' AND key='localUserKey'").fetchone()
        rows = con.execute(
            "SELECT i.itemID, i.key, v.value FROM items i JOIN itemData d ON d.itemID = i.itemID "
            "JOIN fields f ON f.fieldID = d.fieldID JOIN itemDataValues v ON v.valueID = d.valueID "
            "WHERE f.fieldName = 'title' AND i.itemID NOT IN (SELECT itemID FROM deletedItems) "
            "AND i.libraryID = 1").fetchall()
        con.close()
    if not user:
        return {}
    wanted = {norm(t) for t in titles}
    return {norm(t): (iid, f"http://zotero.org/users/local/{user[0]}/items/{key}")
            for iid, key, t in rows if norm(t) in wanted}


def norm(title):
    return re.sub(r"[^0-9a-z]+", " ", re.sub(r"[{}]", "", title).lower()).strip()


def csl_items(pandoc, bib):
    out = subprocess.run([pandoc, str(bib), "-f", "biblatex", "-t", "csljson"], capture_output=True, check=True)
    return {it["id"]: it for it in json.loads(out.stdout.decode("utf-8"))}


def bibliography_xml(pandoc, bib, csl, order):
    """Danh mục tài liệu tham khảo định dạng IEEE (các đoạn XML Word), đúng thứ tự trích dẫn trong báo cáo."""
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "b.md"
        md.write_text(" ".join(f"[@{k}]" for k in order) + "\n\n::: {#refs}\n:::\n", encoding="utf-8")
        out = Path(tmp) / "b.docx"
        subprocess.run([pandoc, str(md), "-o", str(out), "--citeproc", "--bibliography", str(bib), "--csl", str(csl)],
                       check=True)
        doc = zipfile.ZipFile(out).read("word/document.xml").decode("utf-8")
    paras = re.findall(r'<w:p>(?:(?!</w:p>).)*?w:val="Bibliography".*?</w:p>', doc, re.S)
    # Liên kết URL/DOI trỏ tới r:id của tệp tạm, không có trong báo cáo: giữ chữ, bỏ liên kết
    return [re.sub(r"<w:hyperlink [^>]*>(.*?)</w:hyperlink>", r"\1", p, flags=re.S) for p in paras]


def field_runs(instr, result_runs):
    return ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> {escape(instr)} </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>' + result_runs +
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>')


def rand_id(n=8):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def apply(doc, groups, pandoc, bib, csl):
    """Thay mã giữ chỗ trong document.xml bằng trường Zotero. Trả về (doc, số nguồn đã gắn thư viện)."""
    order, num = numbering(groups)
    items = csl_items(pandoc, bib)
    links = library_links([items[k].get("title", "") for k in order])
    linked = 0
    cite_items = {}
    for k in order:
        data = dict(items[k])
        link = links.get(norm(data.get("title", "")))
        data["id"] = link[0] if link else 900000 + num[k]
        cite_items[k] = {"id": data["id"], "uris": [link[1]] if link else [], "itemData": data}
        linked += bool(link)

    def citation(i):
        keys = groups[i]
        text = ", ".join(f"[{num[k]}]" for k in keys)
        payload = {"citationID": rand_id(), "properties": {"formattedCitation": text, "plainCitation": text,
                                                           "noteIndex": 0},
                   "citationItems": [cite_items[k] for k in keys], "schema": SCHEMA}
        instr = "ADDIN ZOTERO_ITEM CSL_CITATION " + json.dumps(payload, ensure_ascii=False)
        return instr, text

    # Mã giữ chỗ nằm trong một <w:t>; tách run thành: phần trước + trường + phần sau, giữ định dạng run
    def run_repl(m):
        rpr, attrs, text = m.group(1) or "", m.group(2), m.group(3)
        parts = re.split(r"ZZCITE(\d+)ZZ", text)
        out = []
        for j, part in enumerate(parts):
            if j % 2 == 0:
                if part:
                    out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{part}</w:t></w:r>')
            else:
                instr, shown = citation(int(part))
                out.append(field_runs(instr, f'<w:r>{rpr}<w:t>{shown}</w:t></w:r>'))
        return "".join(out)
    doc = re.sub(r"<w:r>(<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t( [^>]*)?>([^<]*ZZCITE\d+ZZ[^<]*)</w:t></w:r>",
                 run_repl, doc, flags=re.S)

    # Danh mục tài liệu tham khảo: một trường ZOTERO_BIBL trải qua các đoạn
    paras = bibliography_xml(pandoc, bib, csl, order)
    instr = 'ADDIN ZOTERO_BIBL {"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY'
    first = paras[0].replace("</w:pPr>", "</w:pPr>" + field_runs(instr, "")[:-len('<w:r><w:fldChar w:fldCharType="end"/></w:r>')], 1)
    last_end = '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    paras = [first] + paras[1:]
    paras[-1] = paras[-1][:-len("</w:p>")] + last_end
    bib_par = re.search(r"<w:p>(?:(?!<w:p>).)*?" + BIB_TOKEN + r".*?</w:p>", doc, re.S)
    doc = doc.replace(bib_par.group(0), "".join(paras))
    return doc, linked, len(order)


def doc_prefs(files):
    """Thuộc tính tài liệu Zotero: kiểu IEEE, dùng trường Word (đọc khi bấm Refresh trong tab Zotero)."""
    pref = ('<data data-version="3" zotero-version="7.0"><session id="' + rand_id() + '"/>'
            '<style id="http://www.zotero.org/styles/ieee" locale="vi-VN" hasBibliography="1" bibliographyStyleHasBeenSet="1"/>'
            '<prefs><pref name="fieldType" value="Field"/><pref name="automaticJournalAbbreviations" value="true"/>'
            '</prefs></data>')
    chunks = [pref[i:i + 240] for i in range(0, len(pref), 240)]
    props = "".join(f'<property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="{i + 2}" name="ZOTERO_PREF_{i + 1}">'
                    f"<vt:lpwstr>{escape(c)}</vt:lpwstr></property>" for i, c in enumerate(chunks))
    files["docProps/custom.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties '
        'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">' + props + "</Properties>"
    ).encode("utf-8")
    rels = files["_rels/.rels"].decode("utf-8")
    if "custom-properties" not in rels:
        rels = rels.replace("</Relationships>",
                            '<Relationship Id="rIdZoteroPrefs" Type="http://schemas.openxmlformats.org/officeDocument/'
                            '2006/relationships/custom-properties" Target="docProps/custom.xml"/></Relationships>')
        files["_rels/.rels"] = rels.encode("utf-8")
    types = files["[Content_Types].xml"].decode("utf-8")
    if "/docProps/custom.xml" not in types:
        types = types.replace("</Types>", '<Override PartName="/docProps/custom.xml" ContentType="application/'
                              'vnd.openxmlformats-officedocument.custom-properties+xml"/></Types>')
        files["[Content_Types].xml"] = types.encode("utf-8")
