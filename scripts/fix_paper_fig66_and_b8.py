"""Refresh Figure 6-6 and keep Appendix B.8 troubleshooting table together."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "final-project-paper" / "论文" / "实时视线追踪技术研究与实现.docx"
IMG = ROOT / "docs" / "final-project-paper" / "引用图片"


def find_paragraph(doc: Document, text: str) -> Paragraph:
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def paragraph_index(doc: Document, target: Paragraph) -> int:
    return next(i for i, paragraph in enumerate(doc.paragraphs) if paragraph._p is target._p)


def clear_paragraph(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        paragraph._p.remove(child)


def replace_image_before_caption(doc: Document, caption: str, image_path: Path, width_cm: float) -> None:
    caption_p = find_paragraph(doc, caption)
    caption_i = paragraph_index(doc, caption_p)
    for i in range(caption_i - 1, max(-1, caption_i - 8), -1):
        paragraph = doc.paragraphs[i]
        if paragraph._p.xpath(".//w:drawing") or paragraph._p.xpath(".//w:pict"):
            clear_paragraph(paragraph)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
            return
    raise ValueError(f"Image paragraph not found before caption: {caption}")


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = tr_pr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cantSplit")
    if cant_split is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def keep_b8_table_together(doc: Document) -> None:
    heading = None
    for paragraph in doc.paragraphs:
        if paragraph.text.replace("\n", "").strip() == "B.8 常见问题与处理方式":
            heading = paragraph
            break
    if heading is None:
        raise ValueError("B.8 heading not found")

    previous = OxmlElement("w:p")
    heading._p.addprevious(previous)
    page_break_paragraph = Paragraph(previous, heading._parent)
    page_break_paragraph.add_run().add_break()
    clear_paragraph(heading)
    heading.add_run("B.8 常见问题与处理方式")
    heading.paragraph_format.keep_with_next = True

    target_table = None
    for table in doc.tables:
        header = " | ".join(cell.text.strip() for cell in table.rows[0].cells)
        if header == "问题 | 处理建议":
            target_table = table
            break
    if target_table is None:
        raise ValueError("Appendix B.8 troubleshooting table not found")
    for row in target_table.rows:
        set_row_cant_split(row)
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_together = True


def main() -> None:
    doc = Document(DOCX)
    replace_image_before_caption(doc, "图6-6 不同校准策略与最终25点校准结果对比", IMG / "6-6.png", 15.0)
    keep_b8_table_together(doc)
    doc.save(DOCX)


if __name__ == "__main__":
    main()
