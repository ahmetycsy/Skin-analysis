"""İndirilebilir PDF rapor (fpdf2). Türkçe karakterler için assets/fonts içindeki DejaVu yazı tipi kullanılır."""
from __future__ import annotations

import io
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos

import i18n
from i18n import t

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")

LOGO = os.path.join(os.path.dirname(__file__), "assets", "bau_logo.png")

# sitedeki renklerle aynı (app.py: INK, MUTED, TEAL, LINE, BG)
INK = (28, 43, 49)
MUTED = (85, 104, 111)
TEAL = (14, 94, 87)
LINE = (220, 228, 227)
SOFT = (245, 248, 248)
GROUP_RGB = {"urgent": (166, 61, 87), "refer": (176, 112, 26), "benign": (47, 127, 120)}
GROUP_FILL = {"urgent": (250, 238, 241), "refer": (251, 244, 231), "benign": (234, 244, 242)}


class _PDF(FPDF):
    def footer(self):
        # ince çizgi + iki satır: proje kimliği / veri lisansı solda, sayfa no sağda
        self.set_draw_color(*LINE)
        self.line(15, self.h - 15.5, self.w - 15, self.h - 15.5)
        self.set_y(-14)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(*MUTED)
        self.cell(self.w - 45, 3.6, t("pdf_footer_1"), align="L")
        self.cell(15, 3.6, f"{self.page_no()} / {{nb}}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(self.w - 30, 3.6, t("pdf_footer"), align="L")


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

    # Başlık: solda BAU logosu, sağda rapor adı ve proje bilgisi, en sağda vaka QR'ı; altında ince ayraç
    top_y = 12
    logo_h = 11
    if os.path.exists(LOGO):
        pdf.image(LOGO, x=15, y=top_y, h=logo_h)
    right = pdf.w - 15 - (22 if qr else 0)
    tx = 15 + 66
    pdf.set_xy(tx, top_y - 0.5)
    pdf.set_text_color(*INK)
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(right - tx, 6.5, t("pdf_title"), align="R", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(right - tx, 4.2, t("pdf_org"), align="R", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.cell(right - tx, 4.2, t("pdf_proto"), align="R")
    if qr:  # vaka numarasının QR kodu
        pdf.image(io.BytesIO(qr), x=pdf.w - 15 - 19, y=top_y - 2.5, w=19, h=19)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.6)
    pdf.line(15, top_y + 19, 15 + W, top_y + 19)
    pdf.set_line_width(0.2)

    # Vaka şeridi: dört alan, küçük etiket üstte, değer altta
    sy = top_y + 23
    fields = [(t("pdf_f_case"), res["case"]), (t("pdf_f_date"), res["created"]),
              (t("pdf_f_model"), res["model_name"]), (t("pdf_f_version"), version)]
    widths = [40, 37, W - 40 - 37 - 18, 18]
    pdf.set_fill_color(*SOFT)
    pdf.rect(15, sy, W, 12.5, style="F")
    x = 15
    for (lab, val), w in zip(fields, widths):
        pdf.set_xy(x + 3, sy + 1.8)
        pdf.set_font("DejaVu", "", 6.8)
        pdf.set_text_color(*MUTED)
        pdf.cell(w - 4, 3.5, lab, new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(*INK)
        txt = str(val)
        while txt and pdf.get_string_width(txt) > w - 5:
            txt = txt[:-1]
        pdf.cell(w - 4, 5, txt if txt == str(val) else txt.rstrip() + "…")
        x += w
    pdf.set_y(sy + 12.5 + 3)

    def notice(text, fill, border):
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*border)
        pdf.set_text_color(*INK)
        pdf.set_font("DejaVu", "", 8)
        pdf.multi_cell(W, 4.2, text, border=1, fill=True, padding=1.8, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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

    # Hasta bilgileri (sol) + görüntüler (sağ). Uzun satırlar kendi sütununda alt satıra geçer; görüntülerin yanında
    # sol sütun görüntülerin başladığı yere kadar, görüntülerin altında tam genişlik kullanılır (üst üste binmez).
    top = pdf.get_y()
    pairs = [(res.get("clin_jpg"), t("m_clin")), (res.get("derm_jpg"), t("m_derm")), (res.get("cam_jpg"), "Grad-CAM")]
    pairs = [(i, c) for i, c in pairs if i]
    img_left, img_bottom = 15 + W, top
    if pairs:
        size = 27 if len(pairs) <= 2 else 23
        img_left = 15 + W - len(pairs) * (size + 3) + 3
        img_bottom = top + 8.5 + size + 5
    heading(t("pdf_patient"))
    if pairs:  # başlık çizgisi görüntü sütununa girmesin diye görüntüleri başlıktan sonra çiz
        pdf.set_xy(img_left, top + 1.5)
        pdf.set_font("DejaVu", "B", 10.5)
        pdf.set_text_color(*INK)
        pdf.cell(size, 6, t("pdf_images"))
        pdf.set_font("DejaVu", "", 7)
        pdf.set_text_color(*MUTED)
        for k, (data, cap) in enumerate(pairs):
            pdf.image(io.BytesIO(data), x=img_left + k * (size + 3), y=top + 8.5, w=size, h=size)
            pdf.set_xy(img_left + k * (size + 3), top + 8.5 + size + 0.8)
            pdf.cell(size, 3.5, cap.capitalize(), align="C")
    pdf.set_y(top + 9)
    pdf.set_font("DejaVu", "", 9)
    lab_w = min(max(pdf.get_string_width(lab) for lab, _ in meta_lines) + 4, 52)
    lh = 4.6
    for lab, val in meta_lines:
        y = pdf.get_y()
        right = (img_left - 4) if y < img_bottom else (15 + W)
        val_w = max(right - 15 - lab_w, 30)
        lab_lines = pdf.multi_cell(lab_w - 1, lh, lab, dry_run=True, output="LINES")
        val_lines = pdf.multi_cell(val_w, lh, str(val), dry_run=True, output="LINES")
        h = max(len(lab_lines), len(val_lines)) * lh
        if y + h > pdf.h - 18:
            pdf.add_page()
            y = pdf.get_y()
            right, val_w = 15 + W, W - lab_w
        pdf.set_xy(15, y)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(lab_w - 1, lh, lab, align="L")
        pdf.set_xy(15 + lab_w, y)
        pdf.set_text_color(*INK)
        pdf.multi_cell(val_w, lh, str(val), align="L")
        pdf.set_y(y + h + 1.0)
    pdf.set_y(max(pdf.get_y(), img_bottom) + 1)

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
    # Yönlendirme cetveli: üç bölge, bu vakanın bölgesi koyu, işaret yaklaşık yeri gösterir (sitedekiyle aynı)
    if "pos" in a:
        zones = ("benign", "refer", "urgent")
        seg_w = (W - 4) / 3
        ry = pdf.get_y() + 1.5
        for k, z in enumerate(zones):
            pdf.set_fill_color(*(GROUP_RGB[z] if z == lvl else GROUP_FILL[z]))
            pdf.rect(15 + k * (seg_w + 2), ry, seg_w, 3.2, style="F")
        mx = 15 + (W) * a["pos"] / 100
        pdf.set_fill_color(*INK)
        pdf.rect(mx - 0.6, ry - 1.6, 1.2, 6.4, style="F")
        pdf.set_y(ry + 5)
        pdf.set_font("DejaVu", "", 8)
        for k, z in enumerate(zones):
            pdf.set_text_color(*(INK if z == lvl else MUTED))
            pdf.set_font("DejaVu", "B" if z == lvl else "", 8)
            pdf.cell(W / 3, 4.5, t("ru_" + z), align=("L", "C", "R")[k])
        pdf.ln(6)
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
            pdf.rect(15, pdf.get_y(), W, 4.9, style="F")
        pdf.set_font("DejaVu", "B" if i < 3 else "", 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(56, 4.9, fit(i18n.plain(c), 56), new_x=XPos.RIGHT)
        pdf.set_font("DejaVu", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(77, 4.9, fit(f"{i18n.cname(c)} ({c})", 77))
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(*_hex(group_colors[groups_of[c]]))
        pdf.cell(33, 4.9, fit(i18n.group(groups_of[c]), 33))
        pdf.set_font("DejaVu", "B" if i < 3 else "", 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(W - 166, 4.9, i18n.pct(p), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Hekim değerlendirmesi: girildiyse yazılır, girilmediyse elle doldurulacak boş alan bırakılır
    pdf.ln(1)
    need = 9 + (len(physician) * 5.2 if physician else 5 + 3 * 6.6) + 8
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
            pdf.ln(0.6)
            pdf.cell(62, 6, lab)
            y = pdf.get_y() + 5.2
            pdf.line(15 + 62, y, 15 + W, y)
            pdf.ln(6)
    pdf.ln(2)
    pdf.set_text_color(*MUTED)
    pdf.set_font("DejaVu", "", 8.5)
    sig = f"{t('pdf_signature')}:"
    pdf.cell(W / 2, 6, sig)
    pdf.set_draw_color(*LINE)
    y = pdf.get_y() + 5.2
    pdf.line(15 + pdf.get_string_width(sig) + 3, y, 15 + W / 2 + 10, y)

    pdf.alias_nb_pages()
    return bytes(pdf.output())
