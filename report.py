"""İndirilebilir PDF rapor (fpdf2). Türkçe karakterler için assets/fonts içindeki DejaVu yazı tipi kullanılır."""
from __future__ import annotations

import io
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos

import i18n
from i18n import t

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")

INK = (31, 58, 68)
MUTED = (96, 116, 124)
TEAL = (15, 118, 110)
LINE = (214, 226, 224)
SOFT = (244, 247, 247)
GROUP_RGB = {"urgent": (166, 61, 87), "refer": (176, 112, 26), "benign": (47, 127, 120)}
GROUP_FILL = {"urgent": (250, 238, 241), "refer": (251, 244, 231), "benign": (234, 244, 242)}


class _PDF(FPDF):
    def footer(self):
        self.set_y(-13)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, t("pdf_footer"), align="L")
        self.cell(0, 5, f"{self.page_no()}/{{nb}}", align="R")


def _hex(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def build_pdf(res: dict, a: dict, is_mock: bool, *, version: str, group_colors: dict, groups_of: dict,
              group_order: list, meta_lines: list[tuple[str, str]], physician: list | None = None, qr: bytes | None = None) -> bytes:
    pdf = _PDF(format="A4")
    pdf.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()
    W = pdf.w - 30

    # Başlık bandı
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, pdf.w, 24, style="F")
    pdf.set_xy(15, 6.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("DejaVu", "B", 15)
    pdf.cell(W, 7, t("pdf_title"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 8.5)
    pdf.cell(W, 5, f"{t('app_name')} · {t('app_sub')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if qr:  # vaka numarasının QR kodu (beyaz zemin, bandın sağında)
        pdf.set_fill_color(255, 255, 255)
        pdf.rect(pdf.w - 15 - 19, 2.5, 19, 19, style="F")
        pdf.image(io.BytesIO(qr), x=pdf.w - 15 - 18, y=3, w=17, h=17)

    pdf.set_y(29)
    pdf.set_text_color(*MUTED)
    pdf.set_font("DejaVu", "", 8.5)
    pdf.cell(W, 5, t("pdf_meta", case=res["case"], date=res["created"], model=res["model_name"], v=version),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    def notice(text, fill, border):
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*border)
        pdf.set_text_color(*INK)
        pdf.set_font("DejaVu", "", 8.5)
        pdf.multi_cell(W, 4.4, text, border=1, fill=True, padding=1.8, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.2)

    notice(t("pdf_nodx"), (255, 248, 230), (240, 220, 164))
    if is_mock:
        notice(t("pdf_demo"), (255, 244, 221), (229, 196, 128))

    def heading(text):
        pdf.ln(1)
        pdf.set_font("DejaVu", "B", 10.5)
        pdf.set_text_color(*INK)
        pdf.cell(W, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(*LINE)
        pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
        pdf.ln(2)

    # Hasta bilgileri (sol) + görüntüler (sağ)
    top = pdf.get_y()
    heading(t("pdf_patient"))
    pdf.set_font("DejaVu", "", 9)
    lab_w = max(pdf.get_string_width(lab) for lab, _ in meta_lines) + 5
    for lab, val in meta_lines:
        pdf.set_text_color(*MUTED)
        pdf.cell(lab_w, 6, lab)
        pdf.set_text_color(*INK)
        pdf.cell(90 - lab_w, 6, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    left_end = pdf.get_y()

    pairs = [(res.get("clin_jpg"), t("m_clin")), (res.get("derm_jpg"), t("m_derm")), (res.get("cam_jpg"), "Grad-CAM")]
    pairs = [(i, c) for i, c in pairs if i]
    imgs = [i for i, _ in pairs]
    if imgs:
        size = 30 if len(imgs) <= 2 else 24
        x = 15 + W - len(imgs) * (size + 3) + 3
        pdf.set_xy(x, top + 1.5)
        pdf.set_font("DejaVu", "B", 10.5)
        pdf.set_text_color(*INK)
        pdf.cell(size, 6, t("pdf_images"))
        caps = [c for _, c in pairs]
        pdf.set_font("DejaVu", "", 7)
        pdf.set_text_color(*MUTED)
        for k, data in enumerate(imgs):
            pdf.image(io.BytesIO(data), x=x + k * (size + 3), y=top + 8.5, w=size, h=size)
            pdf.set_xy(x + k * (size + 3), top + 8.5 + size + 0.8)
            pdf.cell(size, 3.5, caps[k].capitalize(), align="C")
        left_end = max(left_end, top + 8.5 + size + 5)
    pdf.set_y(left_end + 1)

    # Genel değerlendirme
    heading(t("pdf_assess"))
    lvl = a["level"]
    pdf.set_fill_color(*GROUP_FILL[lvl])
    pdf.set_draw_color(*GROUP_FILL[lvl])
    y0 = pdf.get_y()
    pdf.set_x(18)
    pdf.set_text_color(*GROUP_RGB[lvl])
    # önce kutunun yüksekliğini ölçmek için metni yazdırmadan satırlara böl
    pdf.set_font("DejaVu", "", 9)
    lines = pdf.multi_cell(W - 6, 4.8, a["text"], dry_run=True, output="LINES")
    box_h = 2.5 + 6 + len(lines) * 4.8 + 3
    pdf.set_font("DejaVu", "B", 11)
    pdf.rect(15, y0, W, box_h, style="F")
    pdf.set_xy(18, y0 + 2.5)
    pdf.cell(W - 6, 6, a["title"], new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(W - 6, 4.8, a["text"], align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(y0 + box_h + 2)
    if a["uncertain"]:
        notice(t("pdf_unsure"), (255, 248, 230), (240, 220, 164))

    # Grup çubuğu
    x = 15
    y = pdf.get_y() + 1
    for g in group_order:
        w = W * a["totals"][g]
        if w > 0:
            pdf.set_fill_color(*_hex(group_colors[g]))
            pdf.rect(x, y, w, 3.2, style="F")
        x += w
    pdf.set_y(y + 5)
    pdf.set_font("DejaVu", "", 8.5)
    for g in group_order:
        pdf.set_fill_color(*_hex(group_colors[g]))
        cy = pdf.get_y() + 1.6
        pdf.ellipse(pdf.get_x(), cy, 2.4, 2.4, style="F")
        pdf.set_x(pdf.get_x() + 3.6)
        pdf.set_text_color(*INK)
        pdf.cell(54, 5.5, f"{i18n.group(g)}  {i18n.pct(a['totals'][g])}")
    pdf.ln(8)

    # Tüm türler tablosu (ilk üç satır kalın)
    heading(t("pdf_all"))
    cols = [(t("col_plain"), 56), (t("col_medical"), 77), (t("col_group"), 33), (t("col_prob"), W - 166)]

    def fit(text, w):
        if pdf.get_string_width(text) <= w - 1.5:
            return text
        while text and pdf.get_string_width(text + "…") > w - 1.5:
            text = text[:-1]
        return text.rstrip() + "…"
    pdf.set_font("DejaVu", "B", 8.5)
    pdf.set_text_color(*MUTED)
    for name, w in cols:
        pdf.cell(w, 6, name, align="R" if w == cols[-1][1] else "L")
    pdf.ln(6)
    for i, (c, p) in enumerate(a["ranked"]):
        if i % 2 == 0:
            pdf.set_fill_color(*SOFT)
            pdf.rect(15, pdf.get_y(), W, 5.2, style="F")
        pdf.set_font("DejaVu", "B" if i < 3 else "", 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(56, 5.2, fit(i18n.plain(c), 56), new_x=XPos.RIGHT)
        pdf.set_font("DejaVu", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(77, 5.2, fit(f"{i18n.cname(c)} ({c})", 77))
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(*_hex(group_colors[groups_of[c]]))
        pdf.cell(33, 5.2, fit(i18n.group(groups_of[c]), 33))
        pdf.set_font("DejaVu", "B" if i < 3 else "", 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(W - 166, 5.2, i18n.pct(p), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Hekim değerlendirmesi: girildiyse yazılır, girilmediyse elle doldurulacak boş alan bırakılır
    pdf.ln(1)
    need = 10 + (len(physician) * 5.2 if physician else 5 + 3 * 7.2) + 9
    if pdf.get_y() + need > pdf.h - 18:  # sığmazsa yeni sayfaya geç
        pdf.add_page()
    heading(t("pdf_physician"))
    pdf.set_font("DejaVu", "", 9)
    if physician:
        lab_w = max(pdf.get_string_width(lab) for lab, _ in physician) + 5
        for lab, val in physician:
            pdf.set_text_color(*MUTED)
            pdf.cell(lab_w, 5.2, lab)
            pdf.set_text_color(*INK)
            pdf.multi_cell(W - lab_w, 5.2, val, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_text_color(*MUTED)
        pdf.set_font("DejaVu", "", 8)
        pdf.cell(W, 5, t("pdf_physician_blank"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", "", 9)
        pdf.set_draw_color(*LINE)
        for lab in (t("fb_dx"), t("fb_action"), t("fb_note")):
            pdf.ln(1.2)
            pdf.cell(62, 6, lab)
            y = pdf.get_y() + 5.2
            pdf.line(15 + 62, y, 15 + W, y)
            pdf.ln(6)
    pdf.ln(2)
    pdf.set_text_color(*MUTED)
    pdf.set_font("DejaVu", "", 8.5)
    pdf.cell(W / 2, 6, f"{t('pdf_signature')}:")
    pdf.set_draw_color(*LINE)
    y = pdf.get_y() + 5.2
    pdf.line(15 + 30, y, 15 + W / 2 - 5, y)

    pdf.alias_nb_pages()
    return bytes(pdf.output())
