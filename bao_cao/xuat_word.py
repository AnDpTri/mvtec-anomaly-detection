"""Xuất báo cáo LaTeX sang Word (bao_cao.docx), giữ số chương/hình/bảng/công thức như bản PDF.

Cần: pandoc, và đã biên dịch LaTeX một lần để có build/main.aux (số thứ tự lấy từ đó).
Chạy từ thư mục bao_cao/:
    latexmk -xelatex -outdir=build main.tex
    python xuat_word.py

Các bước:
1. Gộp main.tex và các chương thành một tệp; tự đánh số chương/mục, chú thích hình/bảng, công thức,
   thay \\ref bằng số (pandoc không tự đánh số các thứ này khi xuất Word).
2. Tạo tệp mẫu định dạng (Times New Roman 12, giãn dòng 1,5, lề như bản PDF, số trang ở chân trang).
3. pandoc chuyển sang Word, có mục lục; trích dẫn và tài liệu tham khảo thành trường Zotero kiểu IEEE
   (zotero_word.py), gắn với thư viện Zotero trên máy nếu đã nhập references.bib.
4. Ghép trang bìa (viền xanh, logo, chữ như chapters/bia.tex) vào đầu, thêm ngắt trang.
5. Mở bằng Word (nếu có) để co giãn bảng, cập nhật mục lục và xuất bản xem thử build/bao_cao_word.pdf.
"""
import argparse
import io
import re
import subprocess
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import zotero_word

HERE = Path(__file__).resolve().parent
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------- 1. Chuẩn bị LaTeX ----------

def read_labels(aux):
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8")))


def flatten(main):
    text = main.read_text(encoding="utf-8")
    body = text[text.index(r"\begin{document}") + len(r"\begin{document}"):text.index(r"\end{document}")]

    def inline(m):
        name = m.group(1)
        if name.endswith("bia"):
            return ""  # bìa dựng riêng ở bước 4
        return (HERE / f"{name}.tex").read_text(encoding="utf-8")
    return re.sub(r"\\input\{([^}]+)\}", inline, body)


def prepare(tex, labels):
    # Bỏ lệnh chỉ có nghĩa với LaTeX
    tex = re.sub(r"\\(tableofcontents|listoffigures|listoftables|clearpage|pagenumbering\{\w+\}|small|addlinespace)\b", "", tex)
    tex = re.sub(r"\\cmidrule\{[^}]*\}", "", tex)
    # pandoc chỉ nhận \midrule ngay dưới dòng tiêu đề; đường kẻ giữa thân bảng làm hỏng cả bảng
    tex = tex.replace("@{}", "")  # pandoc không hiểu @{} trong khai báo cột
    def one_midrule(m):
        t, i = m.group(0), m.group(0).find("\\midrule") + len("\\midrule")
        return t[:i] + t[i:].replace("\\midrule", "") if i > len("\\midrule") - 1 else t
    tex = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", one_midrule, tex, flags=re.S)
    # pandoc làm mất số đứng đầu ô bảng ("3.13", "42, ...", "10\%" bị hiểu như số thứ tự danh sách); bọc trong {}
    tex = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}",
                 lambda m: re.sub(r"(&[ \t]*)(\d[0-9.,]*(?:\\%)?)", r"\1{\2}", m.group(0)), tex, flags=re.S)
    # pandoc bỏ nhãn tự đặt của mục danh sách (\item[(G1)]): chuyển nhãn thành chữ đầu mục
    tex = re.sub(r"\\item\[([^\]]*)\]", r"\\item \1 ", tex)
    # pandoc đọc hỏng ô gộp cột: tách \multicolumn{n}{..}{X} thành X và n-1 ô trống
    tex = re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{([^}]*)\}", lambda m: m.group(2) + " &" * (int(m.group(1)) - 1), tex)
    tex = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", tex)
    tex = re.sub(r"\\printbibliography\[[^\]]*\]", r"\\chapter*{TÀI LIỆU THAM KHẢO}\n\\printbibliography", tex)
    # Mở đầu và Phân công không vào mục lục: đổi thành đoạn in đậm thay vì tiêu đề
    for title in ["MỞ ĐẦU", "PHÂN CÔNG NHIỆM VỤ"]:
        tex = tex.replace(f"\\chapter*{{{title}}}", f"\\begin{{center}}\\textbf{{{title}}}\\end{{center}}")

    # Chú thích hình/bảng: thêm "Hình 3.1." theo nhãn trong cùng môi trường
    def caption(m):
        env, block = m.group(1), m.group(0)
        label = re.search(r"\\label\{([^}]+)\}", block)
        if label and label.group(1) in labels:
            word = "Hình" if env == "figure" else "Bảng"
            block = re.sub(r"\\caption\{", lambda _: f"\\caption{{{word} {labels[label.group(1)]}. ", block, count=1)
        return block
    tex = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", caption, tex, flags=re.S)

    # Công thức: thay \label bằng số ở cuối dòng; \eqref, \ref thành số
    tex = re.sub(r"\\label\{(eq:[^}]+)\}", lambda m: rf"\qquad ({labels.get(m.group(1), '?')})", tex)
    tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels.get(m.group(1), '?')})", tex)
    tex = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), tex)

    # Đánh số chương/mục theo kiểu bản PDF
    out, ch, sec, sub = [], 0, 0, 0
    for line in tex.splitlines():
        m = re.match(r"\s*\\(chapter|section|subsection)\{(.*)\}\s*$", line)
        if m:
            kind, title = m.groups()
            if kind == "chapter":
                ch, sec, sub = ch + 1, 0, 0
                line = f"\\chapter*{{CHƯƠNG {ch}. {title}}}"
            elif kind == "section":
                sec, sub = sec + 1, 0
                line = f"\\section*{{{ch}.{sec}. {title}}}"
            else:
                sub += 1
                line = f"\\subsection*{{{ch}.{sec}.{sub}. {title}}}"
        out.append(line)
    return "\\documentclass{report}\n\\begin{document}\n" + "\n".join(out) + "\n\\end{document}\n"


# ---------- 2. Tệp mẫu định dạng ----------

def rpr(size=24, bold=False, italic=False, color="000000"):
    return (f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" '
            f'w:eastAsia="Times New Roman"/>{"<w:b/><w:bCs/>" if bold else ""}{"<w:i/><w:iCs/>" if italic else ""}'
            f'<w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>')


STYLES = {  # styleId: (pPr, rPr)
    "Normal": ('<w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>', rpr()),
    "BodyText": ('<w:pPr><w:spacing w:before="0" w:after="120"/><w:ind w:firstLine="567"/><w:jc w:val="both"/></w:pPr>', rpr()),
    "FirstParagraph": ('<w:pPr><w:spacing w:before="0" w:after="120"/><w:ind w:firstLine="567"/><w:jc w:val="both"/></w:pPr>', rpr()),
    "Compact": ('<w:pPr><w:spacing w:before="0" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="both"/></w:pPr>', rpr()),
    "Heading1": ('<w:pPr><w:keepNext/><w:pageBreakBefore/><w:spacing w:before="0" w:after="360"/><w:jc w:val="center"/>'
                 '<w:outlineLvl w:val="0"/></w:pPr>', rpr(28, bold=True)),
    "Heading2": ('<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:jc w:val="left"/>'
                 '<w:outlineLvl w:val="1"/></w:pPr>', rpr(28, bold=True)),
    "Heading3": ('<w:pPr><w:keepNext/><w:spacing w:before="200" w:after="100"/><w:jc w:val="left"/>'
                 '<w:outlineLvl w:val="2"/></w:pPr>', rpr(24, bold=True)),
    "ImageCaption": ('<w:pPr><w:spacing w:before="60" w:after="240"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>', rpr(24)),
    "TableCaption": ('<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="60"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>', rpr(24)),
    "CaptionedFigure": ('<w:pPr><w:keepNext/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>', rpr()),
    "Figure": ('<w:pPr><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>', rpr()),
    # outlineLvl 9 = văn bản thường: tiêu đề "MỤC LỤC" không tự xuất hiện trong chính mục lục
    "TOCHeading": ('<w:pPr><w:spacing w:before="0" w:after="240"/><w:ind w:firstLine="0"/><w:jc w:val="center"/>'
                   '<w:outlineLvl w:val="9"/></w:pPr>', rpr(28, bold=True)),
    "Bibliography": ('<w:pPr><w:spacing w:after="80"/><w:ind w:left="567" w:hanging="567"/><w:jc w:val="left"/></w:pPr>', rpr()),
    "SourceCode": ('<w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:firstLine="0"/><w:jc w:val="left"/></w:pPr>', None),
}


def restyle(styles_xml):
    for sid, (ppr, rp) in STYLES.items():
        m = re.search(rf'<w:style [^>]*w:styleId="{sid}".*?</w:style>', styles_xml, re.S)
        if not m:
            continue
        block = re.sub(r"<w:pPr>.*?</w:pPr>|<w:pPr/>", "", m.group(0), flags=re.S)
        if rp:
            block = re.sub(r"<w:rPr>.*?</w:rPr>|<w:rPr/>", "", block, flags=re.S)
        insert = ppr + (rp or "")
        block = block.replace("</w:style>", insert + "</w:style>")
        styles_xml = styles_xml.replace(m.group(0), block)
    # Phông mặc định của cả tài liệu
    styles_xml = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                        "<w:rPrDefault>" + rpr() + "</w:rPrDefault>", styles_xml, flags=re.S)
    return styles_xml


FOOTER = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
SECT = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooterTrang"/><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1418" w:right="1134" w:bottom="1418" w:left="1701" w:header="709" w:footer="709" w:gutter="0"/>'
        '<w:titlePg/></w:sectPr>')  # titlePg: trang bìa không có số trang


def make_reference(pandoc, dest):
    data = subprocess.run([pandoc, "--print-default-data-file", "reference.docx"],
                          capture_output=True, check=True).stdout
    src = zipfile.ZipFile(io.BytesIO(data))
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for item in src.infolist():
            content = src.read(item.filename)
            if item.filename == "word/styles.xml":
                content = restyle(content.decode("utf-8")).encode("utf-8")
            elif item.filename == "word/document.xml":
                content = re.sub(r"<w:sectPr.*?</w:sectPr>", SECT, content.decode("utf-8"), flags=re.S).encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                content = content.decode("utf-8").replace(
                    "</Relationships>",
                    '<Relationship Id="rIdFooterTrang" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    'relationships/footer" Target="footer_trang.xml"/></Relationships>').encode("utf-8")
            elif item.filename == "[Content_Types].xml":
                content = content.decode("utf-8").replace(
                    "</Types>",
                    '<Override PartName="/word/footer_trang.xml" ContentType="application/vnd.openxmlformats-'
                    'officedocument.wordprocessingml.footer+xml"/></Types>').encode("utf-8")
            z.writestr(item, content)
        z.writestr("word/footer_trang.xml", FOOTER)


# ---------- 4. Trang bìa ----------

BLUE = "003399"  # cùng màu xanhbien trong chapters/bia.tex
CM = 360000      # EMU mỗi cm


def para(text="", size=24, bold=False, color="000000", jc="center", before=0, after=0, tabs="", ind=""):
    run = "".join(
        (f'<w:r>{rpr(size, bold, color=color)}<w:tab/></w:r>' if part == "\t" else
         f'<w:r>{rpr(size, bold, color=color)}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
        for part in re.split(r"(\t)", text) if part)
    return (f'<w:p><w:pPr>{tabs}<w:spacing w:before="{before}" w:after="{after}" w:line="276" w:lineRule="auto"/>'
            f'{ind}<w:jc w:val="{jc}"/></w:pPr>{run}</w:p>')


def picture(rid, cx, cy, pid, anchor=False):
    graphic = (f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData '
               f'uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic '
               f'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{pid}" '
               f'name="bia{pid}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch>'
               f'<a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
               f'</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>')
    if not anchor:
        return (f'<w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{pid}" name="bia{pid}"/>'
                f'{graphic}</wp:inline></w:drawing>')
    return (f'<w:drawing><wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="0" '
            f'behindDoc="1" locked="1" layoutInCell="1" allowOverlap="1"><wp:simplePos x="0" y="0"/>'
            f'<wp:positionH relativeFrom="page"><wp:align>center</wp:align></wp:positionH>'
            f'<wp:positionV relativeFrom="page"><wp:align>center</wp:align></wp:positionV>'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:wrapNone/><wp:docPr id="{pid}" name="bia{pid}"/>{graphic}</wp:anchor></w:drawing>')


def cover_xml():
    vien = picture("rIdBiaVien", round(17.18 * CM), round(26.5 * CM), 9001, anchor=True)
    logo = picture("rIdBiaLogo", round(3.5 * CM), round(3.5 * CM), 9002)
    info_tab = '<w:tabs><w:tab w:val="left" w:pos="4820"/></w:tabs>'
    info_ind = '<w:ind w:left="1701"/>'
    # (nhãn, giá trị, giá trị màu xanh đậm?, khoảng sau): nhãn luôn đen đậm; tên nhóm, thành viên, lớp xanh đậm
    rows = [("Tên nhóm:", "NHÓM 13", True, 120), ("Sinh viên thực hiện:", "Đặng Hoàng An", True, 0),
            ("", "Trần Văn Dưỡng", True, 0), ("", "Lớp KHMT K22A", True, 120),
            ("Giáo viên hướng dẫn:", "TS. Ngô Hữu Huy", False, 0)]
    parts = [
        f'<w:p><w:pPr><w:spacing w:before="600" w:after="0"/><w:jc w:val="center"/></w:pPr><w:r>{vien}</w:r>'
        f'<w:r>{rpr(24, True)}<w:t>ĐẠI HỌC THÁI NGUYÊN</w:t></w:r></w:p>',
        para("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG", bold=True),
        para("KHOA CÔNG NGHỆ THÔNG TIN", bold=True, after=480),
        f'<w:p><w:pPr><w:spacing w:after="600"/><w:jc w:val="center"/></w:pPr><w:r>{logo}</w:r></w:p>',
        para("BÁO CÁO DỰ ÁN", size=34, bold=True),
        para("HỌC PHẦN HỌC SÂU (DEEP LEARNING)", size=28, bold=True, after=720),
        para("Đề tài:", bold=True, after=120),
        para("PHÁT HIỆN VÀ ĐỊNH VỊ BẤT THƯỜNG", size=34, bold=True),
        para("TRÊN SẢN PHẨM CÔNG NGHIỆP", size=34, bold=True),
        para("BẰNG HỌC KHÔNG GIÁM SÁT", size=34, bold=True, after=720),
    ]
    for label, value, blue, after in rows:
        runs = (f'<w:r>{rpr(24, True)}<w:t xml:space="preserve">{escape(label)}</w:t></w:r>' if label else "")
        runs += f'<w:r>{rpr(24)}<w:tab/></w:r>'
        runs += f'<w:r>{rpr(24, blue, color=BLUE if blue else "000000")}<w:t>{escape(value)}</w:t></w:r>'
        parts.append(f'<w:p><w:pPr>{info_tab}<w:spacing w:before="0" w:after="{after}" w:line="276" w:lineRule="auto"/>'
                     f'{info_ind}<w:jc w:val="left"/></w:pPr>{runs}</w:p>')
    parts.append(para("THÁI NGUYÊN, NĂM 2026", bold=True, before=1600))
    parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return "".join(parts)


PAGE_BREAK = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def caption_list(title, style):
    """Tiêu đề + trường TOC gom các đoạn có kiểu chú thích `style`; Word điền nội dung khi cập nhật."""
    return (para(title, size=28, bold=True, after=360)
            + '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> TOC \\h \\z \\t "{style},1" </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Cập nhật trường để hiện danh mục.</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')


def finish(docx, groups, pandoc):
    src = zipfile.ZipFile(docx)
    files = {i.filename: src.read(i.filename) for i in src.infolist()}
    src.close()
    doc = files["word/document.xml"].decode("utf-8")
    # Trích dẫn và danh mục tài liệu tham khảo thành trường Zotero (xem zotero_word.py)
    doc, linked, total = zotero_word.apply(doc, groups, pandoc, HERE / "references.bib", HERE / "ieee.csl")
    zotero_word.doc_prefs(files)
    print(f"Zotero: {total} nguồn được trích, {linked} nguồn đã gắn với thư viện Zotero trên máy")
    # Bảng: rộng hết khổ, Word tự chia cột theo nội dung (pandoc không giữ độ rộng cột của LaTeX)
    doc = re.sub(r"<w:tblW [^>]*/>", '<w:tblW w:w="5000" w:type="pct"/><w:tblLayout w:type="autofit"/>', doc)
    doc = re.sub(r"<w:tblLayout w:type=\"fixed\"/>", "", doc)
    doc = re.sub(r'<w:gridCol w:w="\d+"/>', "<w:gridCol/>", doc)
    doc = re.sub(r'<w:tcW [^>]*/>', '<w:tcW w:w="0" w:type="auto"/>', doc)
    # Kẻ khung đầy đủ: viền ngoài và các đường dọc, ngang giữa ô (bỏ qua các "bảng" pandoc dùng để xếp nhiều ảnh
    # trong một hình)
    rule = "<w:tblBorders>" + "".join(
        f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        for side in ["top", "left", "bottom", "right", "insideH", "insideV"]) + "</w:tblBorders>"
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: m.group(0) if "<w:drawing" in m.group(0)
                 else m.group(0).replace("<w:tblLayout", rule + "<w:tblLayout", 1), doc, flags=re.S)
    # Ô bảng dùng kiểu Compact (không thụt dòng đầu như đoạn văn) và giãn dòng 1,15 thay vì 1,5 của thân bài,
    # để bảng không cao quá mức và tràn sang trang chỉ vài dòng
    def table_cells(m):
        t = re.sub(r'w:val="(BodyText|FirstParagraph)"', 'w:val="Compact"', m.group(0))
        return re.sub(r'<w:pStyle w:val="Compact"\s*/>',
                      '<w:pStyle w:val="Compact"/><w:spacing w:before="0" w:after="0" w:line="276" w:lineRule="auto"/>', t)
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", table_cells, doc, flags=re.S)
    # Tiêu đề không vào mục lục (Mở đầu, Phân công): căn giữa, không thụt dòng
    for title in ["MỞ ĐẦU", "PHÂN CÔNG NHIỆM VỤ"]:
        i = doc.find(f">{title}<")
        j = max(doc.rfind("<w:p>", 0, i), doc.rfind("<w:p ", 0, i))
        k = doc.find(">", j) + 1
        doc = doc[:k] + '<w:pPr><w:spacing w:after="360"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>' + \
            re.sub(r"^<w:pPr>.*?</w:pPr>", "", doc[k:], count=1, flags=re.S)
    doc = re.sub(r"(<w:body>)", r"\1" + cover_xml(), doc, count=1)
    # Sau mục lục: danh mục hình, danh mục bảng (trường TOC lấy theo kiểu chú thích), mỗi phần một trang
    doc = doc.replace("</w:sdt>", "</w:sdt>" + PAGE_BREAK + caption_list("DANH MỤC HÌNH", "Image Caption")
                      + PAGE_BREAK + caption_list("DANH MỤC BẢNG", "Table Caption") + PAGE_BREAK, 1)
    i = doc.find(">MỞ ĐẦU<")
    j = doc.rfind("<w:p>", 0, i) if doc.rfind("<w:p>", 0, i) > doc.rfind("<w:p ", 0, i) else doc.rfind("<w:p ", 0, i)
    doc = doc[:j] + PAGE_BREAK + doc[j:]
    if "xmlns:wp=" not in doc[:2000]:
        doc = doc.replace("<w:document ", '<w:document xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/'
                          'wordprocessingDrawing" ', 1)
    files["word/document.xml"] = doc.encode("utf-8")
    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    image = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    rels = rels.replace("</Relationships>",
                        f'<Relationship Id="rIdBiaVien" Type="{image}" Target="media/bia_vien.png"/>'
                        f'<Relationship Id="rIdBiaLogo" Type="{image}" Target="media/bia_logo.png"/></Relationships>')
    files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    types = files["[Content_Types].xml"].decode("utf-8")
    if 'Extension="png"' not in types:
        types = types.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
    files["[Content_Types].xml"] = types.encode("utf-8")
    files["word/media/bia_vien.png"] = (HERE / "images/vien_bia_xanh.png").read_bytes()
    files["word/media/bia_logo.png"] = (HERE / "images/img01.png").read_bytes()
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


WORD_PS = r"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open('{docx}')
  foreach ($t in $d.Tables) { $t.AutoFitBehavior(1); $t.AutoFitBehavior(2) }
  $d.Tables(1).Columns(1).Width = 100; $d.Tables(1).Columns(2).Width = 353   # Phân công: cột tên không ngắt dòng
  # Không chẻ một hàng bảng qua hai trang. Giữ bảng ngắn và các lưới heatmap cùng chú thích.
  foreach ($t in $d.Tables) {
    $t.Rows.AllowBreakAcrossPages = 0
    $hasPictures = $t.Range.InlineShapes.Count -gt 0
    if ($hasPictures -or $t.Rows.Count -le 8) {
      $t.Range.ParagraphFormat.KeepWithNext = -1
      if (-not $hasPictures) { $t.Range.Paragraphs.Last.Range.ParagraphFormat.KeepWithNext = 0 }
    }
    if ($t.Rows.Count -gt 8) {
      $t.Rows(1).HeadingFormat = -1
      if ($t.Range.Text.StartsWith('Image-AUROC')) { $t.Rows(2).HeadingFormat = -1 }
    }
    if (($t.Range.Text -replace '[\s\x07]', '').StartsWith('ThưviệnVaitrò')) {
      $t.Range.Font.Size = 11
      $t.Range.ParagraphFormat.LineSpacingRule = 0
      $t.Range.ParagraphFormat.SpaceBefore = 0
      $t.Range.ParagraphFormat.SpaceAfter = 0
    }
    $following = $d.Range($t.Range.End, $t.Range.End).Paragraphs(1)
    if ($hasPictures) { $following.Range.ParagraphFormat.KeepWithNext = 0 }
    else { $following.Range.ParagraphFormat.SpaceBefore = 6 }
  }
  $inLimitations = $false
  foreach ($p in $d.Paragraphs) {
    $text = $p.Range.Text.Trim()
    if ($p.OutlineLevel -eq 2 -and $text -match '^3\.7\. Hạn chế') { $inLimitations = $true }
    if ($inLimitations -and $p.OutlineLevel -eq 1) { $inLimitations = $false }
    if ($inLimitations) {
      $p.Range.ParagraphFormat.KeepWithNext = -1
      if ($text.StartsWith('Chưa chọn ngưỡng quyết định')) { $p.Range.ParagraphFormat.KeepWithNext = 0 }
    }
    if ($p.OutlineLevel -eq 2 -and $text -match '^1\.9\. Công cụ') {
      $p.Range.ParagraphFormat.SpaceBefore = 6; $p.Range.ParagraphFormat.SpaceAfter = 3
    }
  }
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  # Zotero tạo lại trích dẫn và danh mục (tiếng Việt, theo thư viện); plugin chỉ chạy khi cửa sổ Word hiển thị
  $dotm = "$env:APPDATA\Microsoft\Word\STARTUP\Zotero.dotm"
  if ((Get-Process zotero -ErrorAction SilentlyContinue) -and (Test-Path $dotm)) {
    $w.Visible = $true; $null = $w.AddIns.Add($dotm, $true); $w.Activate(); $d.Activate(); Start-Sleep 2
    try { $w.Run("ZoteroRefresh"); Start-Sleep 10; Write-Output "zotero-refresh-ok" } catch { Write-Output "zotero-refresh-loi" }
    # Zotero để dấu kết thúc trường danh mục ở một đoạn trống riêng; đoạn này có thể tràn thành một trang trắng
    foreach ($f in $d.Fields) { if ($f.Code.Text -match 'ZOTERO_BIBL') {
      $p = $d.Range($f.Result.End, $f.Result.End).Paragraphs(1)
      if ($p.Range.Text.Trim() -eq '') {   # gộp đoạn trống vào nguồn cuối: xóa dấu xuống dòng của đoạn trước nó
        $prev = $p.Previous(1); $d.Range($prev.Range.End - 1, $prev.Range.End).Delete() | Out-Null } } }
    foreach ($t in $d.TablesOfContents) { $t.Update() }
  }
  $d.Save()
  if ('{pdf}') { $d.ExportAsFixedFormat('{pdf}', 17) }
  $d.Close()
} finally { $w.Quit() }
"""


def finish_in_word(docx, pdf=""):
    """Co giãn bảng theo nội dung và cập nhật mục lục bằng chính Word (cần Windows có Word)."""
    script = WORD_PS.replace("{docx}", str(docx)).replace("{pdf}", str(pdf))
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True)
    if r.returncode != 0:
        print("Không chạy được Word; mở bao_cao.docx, chọn cập nhật mục lục khi được hỏi.",
              r.stderr.decode("utf-8", "replace")[:500])
    out = r.stdout.decode("utf-8", "replace")
    print("Zotero Refresh:", "đã chạy" if "zotero-refresh-ok" in out else
          "lỗi" if "zotero-refresh-loi" in out else "bỏ qua (Zotero chưa mở hoặc chưa có plugin Word)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aux", default=HERE / "build/main.aux", type=Path)
    p.add_argument("--pandoc", default="pandoc")
    p.add_argument("--out", default=HERE / "bao_cao.docx", type=Path)
    args = p.parse_args()

    build = HERE / "build"
    build.mkdir(exist_ok=True)
    tex = build / "word_input.tex"
    content, groups = zotero_word.mark_citations(prepare(flatten(HERE / "main.tex"), read_labels(args.aux)))
    tex.write_text(content, encoding="utf-8")
    ref = build / "reference.docx"
    make_reference(args.pandoc, ref)
    subprocess.run([args.pandoc, str(tex), "-f", "latex", "-o", str(args.out),
                    "--reference-doc", str(ref), "--toc", "--toc-depth=3", "-M", "toc-title=MỤC LỤC",
                    "-M", "lang=vi-VN", "--resource-path", str(HERE)], check=True)
    finish(args.out, groups, args.pandoc)
    finish_in_word(args.out.resolve(), (build / "bao_cao_word.pdf").resolve())
    print("Đã tạo", args.out, "(bản xem thử từ Word: build/bao_cao_word.pdf)")


if __name__ == "__main__":
    main()
