"""Fills the 4992 Capstone Progress Report template with the project's current status (draft for the team to revise).

v2 (28 Nov 2026): advisor feedback, Capstone I comparison, threshold sensitivity analysis, robustness test, all figures as of v3.
Run: PYTHONPATH=<pylib with python-docx> python3 analysis/build_progress_report.py
"""
import copy
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/Downloads/4992_ProgressReportTemplate.docx")
OUT = os.path.join(os.path.dirname(HERE), "raporlar", "Capstone_Progress_Report_v2.docx")
FIG = os.path.join(os.path.dirname(HERE), "raporlar", "sekiller")

d = Document(SRC)
P = list(d.paragraphs)
T_TEAM, T_WP, T_CONTRIB = d.tables[0], d.tables[1], d.tables[2]  # template tables (fixed before inserting new ones)
BODY = Pt(11)
TBL = Pt(9)


# ------------------------------------------------------------------ helpers
def add_runs(p, text, size=BODY, italic=False, color=None):
    """Adds text with **bold** markup."""
    for i, part in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        r = p.add_run(part[2:-2] if bold else part)
        r.bold = bold or None
        r.italic = italic or None
        r.font.size = size
        if color:
            r.font.color.rgb = color


def clear(p):
    for r in list(p._p):
        if r.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:fldSimple")):
            p._p.remove(r)


def set_text(p, text, size=BODY):
    clear(p)
    add_runs(p, text, size)
    return p


def set_keep_format(p, text):
    """Cover page: keep the first run's formatting."""
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


class Cursor:
    def __init__(self, anchor):
        self.el = anchor._p if isinstance(anchor, Paragraph) else anchor

    def para(self, text="", style=None, size=BODY, align=None, space_after=None):
        new = OxmlElement("w:p")
        self.el.addnext(new)
        self.el = new
        p = Paragraph(new, d._body)
        if style:
            p.style = d.styles[style]
        if text:
            add_runs(p, text, size)
        if align:
            p.alignment = align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    def bullets(self, items, style="List Bullet"):
        for it in items:
            self.para(it, style=style)

    def table(self, rows, widths, header=True, size=TBL, caption=None, shade_col=None):
        if caption:
            self.caption("Table", caption)
        t = d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = d.styles["Table Grid"]
        self.el.addnext(t._tbl)
        self.el = t._tbl
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = t.cell(i, j)
                c.width = Cm(widths[j])
                p = c.paragraphs[0]
                add_runs(p, str(val), size)
                p.paragraph_format.space_after = Pt(0)
                if header and i == 0:
                    for r in p.runs:
                        r.bold = True
                    shade(c, "E7F2F0")
                if shade_col and i > 0 and j == shade_col[0]:
                    col = shade_col[1].get(str(val).split(" ")[0])
                    if col:
                        shade(c, col)
        for j, w in enumerate(widths):
            for cell in t.columns[j].cells:
                cell.width = Cm(w)
        # table grid must match the cell widths (otherwise Word/LibreOffice may stretch columns differently)
        for gc, w in zip(t._tbl.find(qn("w:tblGrid")).findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(round(w * 567))))
        sp = self.para("", space_after=0)
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.line_spacing = Pt(6)
        return t

    def caption(self, kind, text):
        p = self.para(style="Caption")
        r = p.add_run(f"{kind} ")
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), f" SEQ {kind} \\* ARABIC ")
        rr = OxmlElement("w:r")
        tt = OxmlElement("w:t")
        COUNT[kind] += 1
        tt.text = str(COUNT[kind])
        rr.append(tt)
        fld.append(rr)
        p._p.append(fld)
        p.add_run(f". {text}")
        p.paragraph_format.keep_with_next = kind == "Table"
        return p

    def figure(self, path, caption, width_cm=15.5):
        p = self.para(align=WD_ALIGN_PARAGRAPH.CENTER)
        p.add_run().add_picture(path, width=Cm(width_cm))
        p.paragraph_format.keep_with_next = True
        self.caption("Figure", caption)


COUNT = {"Table": 0, "Figure": 0}


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def set_widths(t, widths):
    """Template tables: set column widths (cells + grid) so long text columns get the room."""
    for row in t.rows:
        for j, w in enumerate(widths):
            if j < len(row.cells):
                row.cells[j].width = Cm(w)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(round(w * 567))))


def section(idx, first_text):
    """Replaces the grey placeholder paragraph idx and returns a cursor after it."""
    p = P[idx]
    set_text(p, first_text)
    return Cursor(p)


# ------------------------------------------------------------------ cover
set_keep_format(P[11], "DESIGN OF A MULTIMODAL CLINICAL DECISION-SUPPORT SYSTEM FOR SKIN LESION CLASSIFICATION USING THE MILK10K DATASET")
set_keep_format(P[13], "Tuğalp Ata, Bedirhan Sevencan, Ahmet Talat Yücesoy")
set_keep_format(P[14], "Department of Industrial Engineering")
P[15]._p.getparent().remove(P[15]._p)
set_keep_format(P[17], "Advisor: Asst. Prof. Dr. Zühal Özcan Yavuz")
set_keep_format(P[18], "Department of Industrial Engineering")
set_keep_format(P[21], "November, 2026")

# ------------------------------------------------------------------ AI declaration
for i in range(31, 36):
    p = P[i]
    p.runs[0].text = "☒ " + p.runs[0].text
set_text(P[36], "☒ Other (please specify): AI was used to implement the team's work. Because the Software Engineering sub-team "
                "did not continue in the fall term (Section 4), most of the application, training and analysis code was written "
                "with an AI coding assistant (Claude Code, Anthropic) from the team's instructions, and report text, tables and "
                "figures were drafted with AI from the team's own results. The ideas, methods and design decisions came from the "
                "team; options suggested by the assistant were only used where the team evaluated and chose them. The team ran "
                "the training and experiments and checked every result against the project files.")

# project team table
rows = ["Department\t: Industrial Engineering", "Member 1\t: Tuğalp Ata", "Member 2\t: Bedirhan Sevencan",
        "Member 3\t: Ahmet Talat Yücesoy", "Advisor \t: Asst. Prof. Dr. Zühal Özcan Yavuz (Industrial Engineering)",
        "Department 2\t: – (the Software Engineering sub-team of Capstone I did not continue in Capstone II; see Section 4)",
        "", "", ""]
for cell, txt in zip(T_TEAM.columns[0].cells, rows):
    p = cell.paragraphs[0]
    runs = p.runs
    if runs:
        runs[0].text = txt
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(txt)


def drop_page_break(par_el):
    for br in list(par_el.iter(qn("w:br"))):
        if br.get(qn("w:type")) == "page":
            br.getparent().remove(br)


body_els = list(d.element.body.iterchildren())
decl_idx = body_els.index(P[23]._p)
team_idx = body_els.index(P[43]._p)
for el in body_els[decl_idx:team_idx]:
    if el.tag == qn("w:p"):
        drop_page_break(el)
lot_idx = body_els.index(P[53]._p)
abbr_idx = body_els.index(P[61]._p)
for el in body_els[lot_idx:abbr_idx]:
    if el.tag == qn("w:p"):
        drop_page_break(el)
for hp in (P[57], P[61]):
    hp.paragraph_format.space_before = Pt(18)
P[29].paragraph_format.space_before = Pt(14)
P[43].paragraph_format.page_break_before = True  # project team table stays whole on its own page
for tr in list(T_TEAM.rows)[6:]:
    T_TEAM._tbl.remove(tr._tr)

# ------------------------------------------------------------------ abstract & keywords
set_text(P[46],
         "This capstone project designs, builds and evaluates a multimodal clinical decision-support (CDS) prototype that sorts skin "
         "lesions into the eleven diagnostic categories of the MILK10k dataset and turns the model output into a referral "
         "recommendation. The six objectives of the Capstone I proposal (a reproducible data pipeline, a dual-image model, "
         "systematic optimisation, Macro F1 ≥ 0.45, a web interface answering within ten seconds, and a skin-tone equity analysis) "
         "are retained; overall progress is about 90%. All figures in this report refer to the deployed version v3 unless stated "
         "otherwise. The main performance claim is the official ISIC MILK10k hidden test, scored by ISIC on 479 lesions whose "
         "labels the team never sees: Macro F1 0.483, above the 0.45 target. On the internal 1,048-lesion validation set the "
         "four-model EfficientNet ensemble reaches Macro F1 0.594 and balanced accuracy 58%; plain accuracy (80.1%) is reported "
         "only as a secondary figure because the most common class dominates it, and the rarest classes remain weak. Training "
         "data were extended with PAD-UFES-20 and ISIC 2019, a documented change of the original single-dataset design, with "
         "duplicates of all MILK10k images removed. The central decision problem is referral: the current thresholds refer 98% of "
         "malignant lesions but also about half of benign ones. A cost and sensitivity analysis shows that this operating point "
         "is optimal only if one missed cancer is valued at roughly 100 unnecessary referrals, and alternative operating points "
         "are prepared for joint finalisation with the advisor. A robustness test of the smartphone mode led to mandatory "
         "lesion cropping in the interface. The usability survey has 15 responses (mean SUS 79.3). Remaining work covers the "
         "threshold decision, physician feedback, a reproducibility run and the final report.", size=BODY)
set_text(P[48], "**Key Words:** clinical decision support, skin lesion classification, MILK10k, cost-sensitive referral, model calibration")

# ------------------------------------------------------------------ abbreviations
abbr = ['AI – Artificial intelligence', 'AKIEC – Actinic keratosis / intraepidermal carcinoma', 'AUC – Area under the ROC curve',
        'BA – Balanced accuracy (mean per-class recall)', 'BCC – Basal cell carcinoma', 'BEN_OTH – Other benign proliferations',
        'BKL – Benign keratinocytic lesion', 'CDS – Clinical decision support', 'CI – Confidence interval', 'CV – Cross-validation',
        'DCA – Decision curve analysis', 'DF – Dermatofibroma', 'ECE – Expected calibration error',
        'Grad-CAM – Gradient-weighted class activation mapping', 'IE – Industrial Engineering',
        'INF – Inflammatory and infectious conditions', 'ISIC – International Skin Imaging Collaboration',
        'MAL_OTH – Other malignant proliferations', 'MEL – Melanoma', 'NV – Melanocytic nevus',
        'PAD-UFES-20 – Smartphone skin-lesion dataset of the Federal University of Espírito Santo',
        'R / U – Referral / priority threshold', 'ROC – Receiver operating characteristic', 'SCCKA – Squamous cell carcinoma / keratoacanthoma',
        'SE – Software Engineering', 'SUS – System Usability Scale', 'TTA – Test-time augmentation',
        'VASC – Vascular lesions and haemorrhage', 'WP – Work package']
c = section(62, "The abbreviations used in this report are listed below.")
half = (len(abbr) + 1) // 2
rows = [["Abbreviation", "Meaning", "Abbreviation", "Meaning"]]
for k in range(half):
    a1 = abbr[k].split(" – ", 1)
    a2 = abbr[k + half].split(" – ", 1) if k + half < len(abbr) else ["", ""]
    rows.append([a1[0], a1[1], a2[0], a2[1]])
c.table(rows, widths=[2.4, 5.7, 2.4, 5.7], size=Pt(8.5))

# ------------------------------------------------------------------ 1 executive summary
c = section(65,
            "The project is on schedule. Overall progress is estimated at **about 90%**: data engineering (WP2), baseline "
            "modelling (WP3) and model development (WP4) are complete; the decision-support interface (WP5) is deployed at "
            "bau-skin-cds.streamlit.app and has been tested by 15 users; evaluation and reporting (WP6) are about 75% complete. "
            "All numbers refer to the deployed version **v3** unless stated otherwise, and the same numbers are used on the "
            "website and in the presentation.")
c.para("**Main result.** On the official ISIC MILK10k hidden test (479 lesions, labels held by ISIC) the deployed system "
       "scores **Macro F1 0.483**, above the proposal target of 0.45. This is the primary performance claim because the "
       "internal validation set was also used for model and threshold choices. On that validation set (1,048 lesions) the "
       "system reaches Macro F1 0.594, balanced accuracy 58% and macro ROC-AUC 0.93; its accuracy of 80.1% looks high mainly "
       "because basal cell carcinoma makes up 48% of the lesions. Rare classes remain the weak point (Section 6.2).")
c.para("**Change of design decision.** The proposal limited training to MILK10k. To make the smartphone (patient) mode usable "
       "and to strengthen rare dermoscopic classes, PAD-UFES-20 and ISIC 2019 were added with the advisor's approval. The "
       "change, the training data of each model, the duplicate check against the validation set and the licences are "
       "documented in Section 2.2; MILK10k-only results are kept as a separate reference.")
c.para("**Key open decision.** With the current referral thresholds the system refers 98% of malignant lesions (98.7% with "
       "the melanoma safety rule) but also 45% of benign lesions (51% with the rule). Section 6.4 treats this as an explicit "
       "cost-versus-safety decision problem. The analysis shows that the current point is optimal only when a missed cancer is "
       "valued at roughly 100 unnecessary referrals, and that the priority threshold sends far more patients to the priority "
       "queue than a cost model supports. Three operating points are prepared to be finalised together with the advisor.")
c.para("**Other progress.** A robustness test of the smartphone mode (Tuğalp Ata) showed that small lesions on a fabric "
       "background can change the prediction in about half of the photos; the interface now requires a square crop centred "
       "on the lesion and flags borderline results. No issue currently blocks the remaining work.")

# ------------------------------------------------------------------ 2 objectives
set_text(P[67], "The objectives below are restated from the Capstone I project report (June 2026) [25]. Section 2.2 explains what "
                "changed since then and why.")
c = section(69, "The Capstone I report defined the following specific, measurable objectives:")
c.bullets([
    "Develop a reproducible data-engineering pipeline for MILK10k image pairs and metadata that handles class imbalance and "
    "produces stratified training and validation splits.",
    "Implement a dual-input EfficientNet-B3 fusion model combining both lesion images with patient metadata, and quantify each "
    "input stream through an ablation study.",
    "Optimise the model through hyperparameter tuning, class-imbalance handling and per-class decision-threshold tuning, and "
    "report the improvement over an untuned baseline.",
    "Achieve a Macro F1 of at least 0.45 on the held-out validation split, meaningfully better than a single-modality baseline.",
    "Deliver a web-based decision-support interface that returns the full eleven-class probability distribution within ten "
    "seconds, with an explicit non-diagnostic disclaimer.",
    "Conduct a skin-tone-stratified evaluation and report the findings transparently.",
], style="List Number")
c = section(71, "All six objectives are retained. The way they are achieved changed in the areas listed in Table 1; the most "
                "important change, the use of additional training data, is described in detail below the table.")
c.table([
    ["Area", "Capstone I plan", "Current approach", "Reason"],
    ["Training data", "MILK10k only; other datasets explicitly out of scope",
     "MILK10k plus PAD-UFES-20 (clinical models) and ISIC 2019 (dermoscopic models)",
     "Phone-photo Macro F1 was 0.399 with MILK10k only; rare dermoscopic classes had few examples. Advisor approval, November 2026."],
    ["Application area", "Decision support for clinicians using a paired clinical and dermoscopic image",
     "Two use cases: physician mode (both images) and a patient self-check mode with one smartphone photo; case tracking and follow-up",
     "Most people first notice a spot at home with a phone; this widened the use but required phone-photo data and careful risk communication."],
    ["Team composition", "Joint IE–SE project, six students and two advisors",
     "Three IE students, one IE advisor", "The SE students took Capstone II in the summer term; the IE team continued alone in the fall (Section 4)."],
    ["Development approach", "Code written by the joint team (SE members for software)",
     "Most code written with an AI coding assistant under the IE team's direction; team runs, tests and verifies",
     "No software engineers in the fall team; see Section 7.3."],
    ["Model architecture", "End-to-end late-fusion network (two images + metadata MLP)",
     "Calibrated probability-level ensemble of four image models; metadata network kept as an evaluated alternative",
     "The fusion network (0.497) and metadata stacking did not beat the image ensemble (0.546) on the same split."],
    ["Optimisation", "Grid/random hyperparameter search; per-class thresholds maximising Macro F1",
     "Fixed training settings from early runs; temperature calibration; conformal sets; cost-based referral thresholds",
     "GPU quota limited searches. Per-class thresholds lost 0.01–0.015 Macro F1 under nested CV, so the decision layer was "
     "redesigned around referral cost."],
    ["Output layer", "Per-class sigmoid with weighted binary cross-entropy",
     "Softmax with square-root class weights and label smoothing", "One diagnosis per lesion in MILK10k; better calibration."],
    ["Framework", "TensorFlow/Keras on Colab", "PyTorch + timm on Kaggle", "Pretrained EfficientNetV2 weights and longer free GPU sessions."],
    ["Privacy", "Uploaded images are not stored", "Images are still never stored; anonymous results are stored only with consent",
     "Needed for case tracking across devices and for usage statistics."],
    ["Interface (extended)", "Upload, metadata entry, probability display",
     "Patient mode, case tracking, physician review, PDF/QR reports, Grad-CAM, model-analysis pages",
     "Turns the model into a usable workflow tool, the industrial-engineering focus of the project."],
    ["Leaderboard", "One submission", "Two valid submissions (MILK10k-only and extended data)", "To report both result sets honestly."],
], widths=[2.5, 3.5, 5.0, 5.2], caption="Changes to the approach since the Capstone I report")

c.para("**Design decision change: use of additional training data.** The Capstone I report committed to MILK10k as the only "
       "dataset. This decision was deliberately revised for two reasons. First, the patient mode works with smartphone photos, "
       "while MILK10k clinical images are professional close-ups; on 433 smartphone photos the MILK10k-only clinical ensemble "
       "reached only 46% accuracy (Macro F1 0.399). Second, several dermoscopic classes had too few examples to learn from. "
       "Two pre-registered experiments were run on 21–22 November 2026 with adoption rules fixed before training (for ISIC 2019: physician-mode Macro F1 at least +0.01, missed malignant lesions "
       "not increased, calibration error ≤ 0.03), and the advisor confirmed the use of additional public data at the in-person "
       "meeting in November 2026. The validation set was not changed, so every version is compared on the "
       "same 1,048 MILK10k lesions, and the MILK10k-only version (v1) is reported alongside. The same explanation is shown on "
       "the website ('Data and limitations' page). Table 2 lists the training data of each deployed model.")
c.table([
    ["Deployed model", "Image type", "Training data", "Licence"],
    ["EfficientNet-B3 (300 px)", "Dermoscopic", "MILK10k training split (4,192 lesions) + ISIC 2019 (16,423 images)", "CC BY-NC 4.0"],
    ["EfficientNetV2-S (384 px)", "Dermoscopic", "MILK10k training split (4,192 lesions) + ISIC 2019 (16,423 images)", "CC BY-NC 4.0"],
    ["EfficientNet-B3 (300 px)", "Clinical", "MILK10k training split (4,192 lesions) + PAD-UFES-20 (80% of patients)", "CC BY-NC 4.0 / CC BY 4.0"],
    ["EfficientNetV2-S (384 px)", "Clinical", "MILK10k training split (4,192 lesions) + PAD-UFES-20 (80% of patients)", "CC BY-NC 4.0 / CC BY 4.0"],
], widths=[3.6, 2.2, 7.2, 3.2], caption="Training data of the four deployed models")
c.bullets([
    "**No overlap with the validation set.** ISIC 2019 shares its archive with MILK10k, so every ISIC 2019 image was compared "
    "with all MILK10k images, including the 1,048 validation lesions. No ISIC identifier matched; a 256-bit perceptual hash "
    "(distance ≤ 22, calibrated on MILK10k, where identical photos scored ≤ 20 and different photos ≥ 28) found 33 "
    "near-duplicates, which were removed before training (16,423 images used; duplicate report kept with the training output).",
    "**PAD-UFES-20 test split.** 20% of PAD-UFES-20 patients (433 photos, 277 patients) were held out by patient and never used "
    "in training; they form the independent phone-photo test.",
    "**Class mapping.** PAD-UFES-20 and ISIC 2019 labels were mapped to the eleven MILK10k categories (for example actinic "
    "keratosis → AKIEC, squamous cell carcinoma → SCCKA); the ISIC 2019 'unknown' class was excluded and nevi were capped at 4,000.",
    "**Licences and attribution.** MILK10k and ISIC 2019 are released under CC BY-NC 4.0 (non-commercial use with attribution) "
    "and PAD-UFES-20 under CC BY 4.0 (attribution). The project is non-commercial, images are not redistributed, and the "
    "required citations are given in the reference list [1]–[3], [5]–[7], [26]–[27].",
])

# ------------------------------------------------------------------ 3.1 WP table
set_text(P[74], "Table 3 summarises progress against the work breakdown structure of the Capstone I report (Table 4 of that "
                "report). Because the Capstone I team included three SE students, the lead roles were redistributed among the "
                "three IE members.")
wp = [["WP1", "Project management: coordination, milestone and risk tracking, GitHub, advisor communication", "Week 15", "85", "Ahmet Talat Yücesoy, Tuğalp Ata"],
      ["WP2", "Data engineering: download, EDA, preprocessing, imbalance handling, stratified split", "Week 4", "100", "Tuğalp Ata"],
      ["WP3", "Baseline models: single-modality baselines, training loop, Macro F1 evaluation", "Week 6", "100", "Ahmet Talat Yücesoy, Tuğalp Ata"],
      ["WP4", "Model development: EfficientNet ensemble, metadata fusion network, ablation, calibration, extra-data models", "Week 11", "100",
       "Tuğalp Ata, Ahmet Talat Yücesoy (fusion network: Bedirhan Sevencan)"],
      ["WP5", "Decision-support interface: application, model integration, deployment, robustness fixes, usability testing", "Week 13", "95",
       "Ahmet Talat Yücesoy, Tuğalp Ata"],
      ["WP6", "Evaluation and reporting: metrics, equity, threshold decision, ISIC submission, reproducibility, report", "Week 15", "75", "Ahmet Talat Yücesoy, Tuğalp Ata"]]
t = T_WP
for i, row in enumerate(wp[:5], start=1):
    for j, val in enumerate(row):
        cell = t.cell(i, j)
        clear(cell.paragraphs[0])
        add_runs(cell.paragraphs[0], val, TBL)
new_tr = copy.deepcopy(t.rows[5]._tr)
t._tbl.append(new_tr)
for j, val in enumerate(wp[5]):
    cell = t.cell(6, j)
    for r in list(cell.paragraphs[0].runs):
        r._r.getparent().remove(r._r)
    add_runs(cell.paragraphs[0], val, TBL)
set_widths(t, [1.5, 6.6, 2.2, 1.9, 4.8])
cap = Cursor(P[74])
cap.caption("Table", "Task-by-task progress against the Capstone I work packages")
c = Cursor(P[75])
c.para("Actual status is the share of each work package's planned scope that is finished. WP5 is 95% because physician "
       "feedback on the interface is still missing; WP6 is 75% because the threshold decision with the advisor, the "
       "reproducibility run and the final report remain.")

# ------------------------------------------------------------------ 3.2 detailed description
c = section(77, "**Timeline of the fall term.** Work started in the first week of November 2026 and the task plan was agreed "
                "on 14 November. The main steps were: 18 November – first online prototype (demo model) and shared Kaggle "
                "training notebook; 19 November – fusion network delivered; 20 November – four image models trained with "
                "cross-validation and connected to the site, bilingual interface and case tracking; 21 November – anonymous "
                "database, first ISIC hidden-test submission (0.485), patient mode, decision thresholds and the model-analysis "
                "pages; 22 November – PAD-UFES-20 and ISIC 2019 models (v2, v3) and second ISIC submission (0.483), patient-"
                "experience redesign; 24 November – melanoma safety rule, scope warning, survey consent and a visual redesign; "
                "27–28 November – advisor feedback, robustness fixes (mandatory crop, borderline warning) and this report.")
c.para("**WP2 – Data engineering.** MILK10k provides 5,240 lesions, each with a clinical close-up and a dermoscopic "
                "image (10,480 images), and metadata on age (5-year bands), sex, skin tone (0 = very dark to 5 = very light) and "
                "anatomical site. The class distribution is highly imbalanced: basal cell carcinoma accounts for 48% of lesions, "
                "while other malignant proliferations have only 9 (Appendix C). Because the dataset has no patient identifier, a "
                "lesion-level stratified 80/20 split with seed 42 was created (4,192 training and 1,048 validation lesions) and "
                "frozen in a split file that every later experiment reuses. The official test files were never used for training.")
c.para("**WP3 – Baselines.** A first dermoscopic EfficientNet-B3 baseline reached a validation Macro F1 of 0.477 and established "
       "the training and evaluation loop. Single-modality ensembles (dermoscopic only, clinical only) are the baselines of the "
       "ablation study (Section 6.3).")
c.para("**WP4 – Model development.** Four ImageNet-pretrained models were trained with identical settings (12 epochs, learning "
       "rate 3×10⁻⁴, batch size 24, square-root class weights, label smoothing 0.05, flip test-time augmentation): "
       "EfficientNet-B3 (300 px) and EfficientNetV2-S (384 px) for each image type. During development (MILK10k only), "
       "five-fold cross-validation gave a physician-mode Macro F1 of 0.542; this figure is supporting evidence only, because "
       "lesions of the same patient may fall into different folds. Probabilities are averaged within and across image types and "
       "calibrated with temperature scaling (T = 0.60 physician mode, 0.76 patient mode). Split-conformal prediction gives a list "
       "of possible diagnoses that contains the true class about 90% of the time (validation coverage 90.5%, mean size 1.46).")
c.para("Bedirhan Sevencan trained a multimodal fusion network combining both images with age, sex, skin tone and site and "
       "delivered its weights and confusion matrix. Tuğalp Ata compared it with the image ensemble on the same validation fold "
       "(fusion 0.497 versus ensemble 0.547), and cross-validated stacking experiments (Tuğalp Ata; repeated by Ahmet Talat "
       "Yücesoy for the ablation) confirmed the pattern: metadata adds a small signal (+0.007 to +0.009 Macro F1) but not enough "
       "to beat simple averaging. The image ensemble was therefore selected as the deployed model because it performed better "
       "on the same fold (0.547 versus 0.497). The training code of the fusion network was not shared, so the network could not "
       "be retrained or reproduced and was used for comparison only. Metadata is still collected for reports, low-data "
       "warnings and the equity analysis.")
c.para("**Decision layer.** The interface turns probabilities into one of three recommendations: priority dermatology review "
       "when the malignant probability total is at least U = 30%, dermatology review when malignant plus pre-malignant is at "
       "least R = 20% or the model is uncertain, and otherwise 'no clear warning sign'. A melanoma safety rule also refers a "
       "lesion when the melanoma probability is at least 10% (physician mode) or 15% (patient mode). Section 6.4 analyses how "
       "these thresholds trade missed cancers against unnecessary referrals.")
c.para("**Robustness test of the patient mode (Tuğalp Ata).** The smartphone mode was rebuilt locally and tested on the 433 "
       "held-out PAD-UFES-20 photos with controlled changes of framing, background, lighting and compression. Results and the "
       "measures taken are reported in Section 6.5.")
c.para("**WP5 – Decision-support interface.** The Streamlit application is deployed at bau-skin-cds.streamlit.app in Turkish and "
       "English. Its main features are listed below.")
c.bullets([
    "A three-step workflow (images, patient information, result) with rotation and image-quality checks (resolution, "
    "sharpness, lighting, skin-likeness). In patient mode a square crop centred on the lesion is mandatory (Section 6.5).",
    "Two modes: physician mode with both images, and patient mode with one smartphone photo and a scope question that warns "
    "against rashes and eczema.",
    "A result screen led by a referral recommendation and a three-zone referral ruler, with an uncertainty warning, a "
    "'borderline result' note near thresholds, possible diagnoses, a reliability box based on validation precision and a "
    "Grad-CAM heat map.",
    "Case tracking with physician review (PIN-protected role), an ABCDE checklist, follow-up dates, lesion timelines and an "
    "append-only audit log.",
    "PDF reports with the university logo, QR code and calendar reminders; CSV export, bulk PDF export and case comparison; "
    "statistics including model–physician agreement (Cohen's kappa).",
    "Model pages (version history, threshold simulator, calibration, decision curve, equity, outpatient-clinic scenario, "
    "architecture, data and limitations), a skin-health information page with AAD and WHO sources, Turkish and English "
    "versions, a dark theme and a mobile layout.",
    "Anonymous storage of results in a PostgreSQL database (Supabase), only with consent; images and personal data are never stored.",
])
c.para("Patient mode leads with an action ('show it to a dermatologist within one to two weeks') rather than cancer "
       "percentages, explains in absolute numbers that most referred patients do not have cancer, and keeps detailed "
       "probabilities behind a 'Details' section, to limit health anxiety caused by the high referral rate.")
c.para("**WP6 – Evaluation.** Completed: validation metrics, ablation, ROC-AUC, calibration, decision-curve, cost and threshold "
       "sensitivity analyses, robustness test, skin-tone, sex and age subgroup analysis, two valid ISIC submissions and a "
       "15-response usability survey. Remaining: the threshold decision with the advisor, physician feedback and the "
       "reproducibility run.")

# ------------------------------------------------------------------ 3.3 contributions
set_text(P[79], "Contributions were not equal. The team started working on Capstone II in the first week of November 2026, "
                "coordinating through WhatsApp. On 14 November 2026 a written task plan assigned each member a work package "
                "with named deliverables and dates, and asked everyone to commit their work under their own name so that "
                "contributions would be visible; all three members accepted the plan. Table 4 is based on that plan and on the verifiable outputs "
                "(repository commits, shared result files and messages). The shares are the team's estimate of effort during "
                "the reporting period.")
contrib = [
    ["Tuğalp Ata", "Industrial Eng.",
     "Exploratory data analysis and verification of the stratified split; training of the four image models that form the "
     "deployed ensemble, with five-fold cross-validation; temperature calibration, conformal prediction sets and the "
     "cost-sensitive decision analysis that set the first thresholds; learned-combination, metadata-stacking and per-class "
     "threshold experiments under nested cross-validation; fusion-versus-ensemble comparison; model configuration and "
     "integration package; robustness test of the patient mode on 433 PAD-UFES-20 photos with Grad-CAM inspection, which led "
     "to the mandatory crop and the borderline warning; data, literature and baseline content of the reports.", "40"],
    ["Bedirhan Sevencan", "Industrial Eng.",
     "Trained the multimodal fusion network (two images + metadata) and delivered its weights and confusion matrix (Macro F1 "
     "0.497 on validation fold 0, 19 November 2026). No training code, analysis or report text was delivered. The other tasks "
     "of his work package (metadata-only model, per-class threshold tuning, skin-tone equity analysis, hand-over of the final "
     "model, ISIC submission, method/results/equity report sections) were completed by the other two members.", "20"],
    ["Ahmet Talat Yücesoy", "Industrial Eng.",
     "Task plan and coordination; first dermoscopic baseline; decision-support web application and its deployment (Streamlit "
     "Cloud, Hugging Face model repository, Supabase database), taking over the interface work planned with the SE sub-team; "
     "Grad-CAM and final-model integration; additional-data experiments (PAD-UFES-20, ISIC 2019) including the duplicate check; "
     "ISIC benchmark submissions; skin-tone, sex and age equity analysis; model-analysis pages (threshold simulator, "
     "calibration, decision curve, clinic scenario) and the threshold sensitivity analysis; usability survey; drafting and "
     "assembly of the reports.", "40"],
]
t = T_CONTRIB
for i, row in enumerate(contrib, start=1):
    for j, val in enumerate(row):
        clear(t.cell(i, j).paragraphs[0])
        add_runs(t.cell(i, j).paragraphs[0], val, TBL)
for tr in list(t.rows)[4:]:
    t._tbl.remove(tr._tr)
set_widths(t, [3.0, 2.4, 10.0, 1.6])
Cursor(P[79]).caption("Table", "Individual contributions during the reporting period")

# ------------------------------------------------------------------ 4 collaboration
set_text(P[81], "In Capstone I the project was a joint project of the Industrial Engineering and Software Engineering "
                "departments, with three students from each department and one advisor from each department (Asst. Prof. Dr. "
                "Zühal Özcan Yavuz and Asst. Prof. Dr. Serkan Şimşek). The SE students took Capstone II in the summer term of 2026, while the "
                "IE students take it in the fall term, so the project was continued by the three IE members alone; the SE "
                "students and the SE advisor are not part of this term's team. There was therefore no inter-departmental "
                "collaboration in this reporting period. The software tasks that the joint "
                "plan shared with the SE sub-team (interface development, deployment, database) were taken over by the IE team, "
                "largely with the help of an AI coding assistant (Section 7.3). "
                "The subsections below describe collaboration within the team.")
set_text(P[83], "The team has worked together since the first week of November 2026, coordinating mainly through a WhatsApp "
                "group, with short calls when decisions were needed; the weekly "
                "meetings planned in Capstone I were replaced by this asynchronous routine. Technical work was shared through a "
                "GitHub repository with one branch per member, a private Kaggle dataset holding MILK10k and the fixed split file "
                "so that every member trained on identical data, and a private Hugging Face repository for model weights. Written "
                "summaries after every experiment (results, files, open questions) kept decisions traceable.")
set_text(P[85], "The main integration challenge was combining work developed in parallel. Branches created from different "
                "versions of the code could not be merged without overwriting newer work, so results were integrated "
                "selectively: model weights, prediction files and metrics were imported instead of whole branches. A single "
                "model-configuration file (model list, input sizes, temperatures, conformal thresholds) decouples training from "
                "the application, so new models are deployed without code changes. When Hugging Face Docker Spaces became a paid "
                "service, the site stayed on Streamlit Community Cloud and the weights are loaded from a free private model "
                "repository. The fusion model was delivered on a branch created from an older version of the "
                "code; merging the branch would have deleted newer files, so only its result was used for comparison. Taking over the SE share of the work increased the workload of the IE team "
                "but did not delay the interface milestone.")
set_text(P[87], "Deliverables that needed input from more than one member are the deployed model package (four image models with "
                "calibration and conformal settings), the shared validation prediction file used by every evaluation, the "
                "robustness test and its interface fixes, the threshold analysis, and this report. All are complete except the "
                "final report and the final threshold decision.")

# ------------------------------------------------------------------ 5 schedule
c = section(90, "The calendar dates differ from Capstone I because of the irregular term; Table 5 compares progress against the "
                "Capstone I milestones using its fifteen-week structure.")
c.table([
    ["Milestone", "Planned", "Status", "Comment"],
    ["M1 Preprocessing pipeline validated", "Week 3", "Completed", "Split file fixed (seed 42) and reused by all experiments."],
    ["M2 Baseline Macro F1 recorded", "Week 6", "Completed", "Dermoscopic baseline 0.477; single-modality baselines in the ablation."],
    ["M3 Fusion model and ablation complete", "Week 11", "Completed", "Ensemble selected; fusion network and metadata evaluated; ablation final."],
    ["M4 Interface functional", "Week 13", "Completed (ahead)", "Deployed online with real models, database and robustness fixes."],
    ["M5 Leaderboard submission and final report", "Week 15", "In progress", "Leaderboard done (two valid submissions); final report due 6 Dec 2026."],
], widths=[4.6, 1.8, 2.8, 7.0], caption="Capstone I milestones and current status",
    shade_col=(2, {"Completed": "D9EFE0", "In": "FBEFC9"}))
c = section(92, "Figure 1 shows the updated schedule mapped onto the fifteen weeks of the plan. Green bars are completed, yellow "
                "in progress, red delayed (none) and blue planned.")
c.figure(f"{FIG}/fig_gantt.png", "Updated Gantt chart (weeks of the Capstone I plan)", 14.5)
c = section(94, "The remaining work is listed below in order of priority. The final report is due on 6 December 2026 (23:59).")
c.table([
    ["Task", "Description", "Target", "Owner"],
    ["Threshold decision", "Present the cost and sensitivity analysis (Section 6.4) to the advisor and fix the final operating point", "by 2 Dec 2026", "All"],
    ["Robustness tables", "Add the full robustness tables (scripts and outputs) to the final report; document mitigations", "by 2 Dec 2026", "Tuğalp Ata"],
    ["Physician feedback", "Collect SUS responses and comments from physicians or residents (none so far)", "by 3 Dec 2026", "All"],
    ["Reproducibility test", "Fresh Kaggle re-run of a training notebook; compare Macro F1 and run time (target ≤ 4 h)", "by 2 Dec 2026", "Ahmet Talat Yücesoy"],
    ["Live demonstration", "Rehearse the 15–30 minute presentation with a live demo; check database and model status beforehand", "by 6 Dec 2026", "All"],
    ["Final deliverables", "Final report (due 6 Dec 2026, 23:59) and presentation", "by 6 Dec 2026", "All"],
], widths=[3.2, 8.0, 2.2, 2.8], caption="Remaining work plan")

# ------------------------------------------------------------------ 6 results
c = section(96, "**6.1 Main result: official hidden test.** ISIC scores submissions on 479 hidden lesions whose labels are not "
                "released, so this is the only evaluation that played no part in any design decision. The deployed version v3 "
                "scores **Macro F1 0.483** (target ≥ 0.45, met); the MILK10k-only version v1 scored 0.485. The validation "
                "results below are therefore supporting evidence and should be read as upper bounds: the best epoch and the "
                "thresholds were chosen on the validation set, and the lesion-level split may place lesions of the same patient "
                "on both sides.")
c.table([
    ["Version", "Physician Macro F1 (val.)", "Missed malignant (of 691)", "Patient Macro F1 (val.)", "Phone-photo test Macro F1", "ISIC hidden test"],
    ["v1 MILK10k only", "0.546", "19", "0.484", "0.399", "0.485"],
    ["v2 + PAD-UFES-20 (clinical)", "0.535", "18", "0.496", "0.668", "–"],
    ["v3 + ISIC 2019 (dermoscopic), deployed", "0.594", "14", "0.496", "0.668", "0.483"],
], widths=[4.4, 2.5, 2.4, 2.3, 2.4, 2.2], caption="Performance of the three model versions")
c.figure(f"{FIG}/fig_versions.png", "Macro F1 of the three model versions on validation, phone-photo and hidden test sets", 12)
c.para("The extended data clearly helped where it was meant to: on the independent phone-photo test (433 photos of 277 "
       "held-out patients) Macro F1 rose from 0.399 to 0.668 (95% bootstrap CI of the difference +0.18 to +0.34) and calibration "
       "error fell from 0.123 to 0.046. On the hidden test, however, the validation gain of v3 (+0.059) did not appear. Likely "
       "reasons are a different class mix in the hidden test (the model predicts far more squamous cell carcinoma and actinic "
       "keratosis there), optimistic validation estimates, and the strong influence of a few rare-class lesions on Macro F1 with "
       "only 479 lesions.")

c.para("**6.2 Validation metrics and rare classes (v3, physician mode).** Table 8 gives the headline validation metrics. "
       "Balanced accuracy, the mean recall over the eleven classes, is the more honest summary than plain accuracy because "
       "basal cell carcinoma alone makes up 48% of the validation lesions.")
c.table([
    ["Metric", "Physician mode", "Patient mode", "Comment"],
    ["Macro F1", "0.594", "0.496", "Official benchmark metric; each class weighs equally"],
    ["Balanced accuracy", "58.0%", "51.0%", "Mean per-class recall"],
    ["Macro ROC-AUC", "0.932", "–", "Ranking quality, threshold-free"],
    ["Expected calibration error", "0.021", "0.029", "Low: stated confidence matches observed accuracy"],
    ["Accuracy (secondary)", "80.1%", "72.4%", "Dominated by the most common class (BCC)"],
], widths=[4.0, 2.6, 2.6, 7.0], caption="Validation metrics of the deployed system (1,048 lesions)")
c.table([
    ["Class", "Validation lesions", "Recall (sensitivity)", "F1"],
    ["BCC", "504", "0.95", "0.91"], ["NV", "149", "0.87", "0.85"], ["BKL", "109", "0.48", "0.54"],
    ["SCCKA", "95", "0.69", "0.68"], ["MEL", "90", "0.66", "0.69"], ["AKIEC", "61", "0.52", "0.62"],
    ["DF", "10", "0.90", "0.90"], ["INF", "10", "0.30", "0.33"], ["BEN_OTH", "9", "0.22", "0.29"],
    ["VASC", "9", "0.78", "0.74"], ["MAL_OTH", "2", "0.00", "0.00"],
], widths=[3.0, 3.6, 4.0, 3.0], caption="Per-class recall and F1 of the deployed physician mode")
c.figure(f"{FIG}/fig_confusion.png", "Confusion matrix of the deployed physician mode (counts; colour = share of each true class)", 10)
c.para("The rare classes are the main weakness. Other malignant proliferations (two validation lesions, nine in the whole "
       "dataset) were never recognised, and other benign proliferations (22%) and inflammatory conditions (30%) are mostly "
       "misclassified; with nine or ten validation lesions each, these recalls are also very uncertain. Benign keratinocytic "
       "lesions (48%) and actinic keratosis (52%) are often confused with basal and squamous cell carcinoma, and melanoma with "
       "nevus. Because Macro F1 weighs every class equally, these classes pull the overall score down and explain most of the "
       "gap to the top of the ISIC leaderboard (about 0.56–0.72). For referral, the confusions within the malignant group "
       "matter less than melanomas predicted as nevi, which motivates the safety rule in Section 6.4.")

c.para("**6.3 Ablation study.** Table 10 shows the contribution of each input.")
c.table([
    ["Input", "Macro F1 (MILK10k only)", "Macro F1 (extended data)", "Macro ROC-AUC (MILK10k only / extended)"],
    ["Dermoscopic image only", "0.477", "0.583", "0.904 / 0.931"],
    ["Clinical image only", "0.484", "0.496", "0.882 / 0.862"],
    ["Both images (deployed ensemble)", "0.546", "0.594", "0.922 / 0.932"],
    ["Both images, logistic stacking (5-fold CV)", "0.438", "0.501", "0.865 / 0.906"],
    ["Both images + metadata, logistic stacking (5-fold CV)", "0.447", "0.508", "0.883 / 0.919"],
], widths=[6.2, 3.0, 3.0, 4.0], caption="Ablation study: contribution of each input")
c.para("Combining both images improves Macro F1 by 0.06–0.07 over the best single image (proposal criterion ≥ +0.05). Metadata "
       "improves a like-for-like stacked model slightly, but simple averaging of the image models remains best on this data "
       "size. Per-class threshold tuning appeared to raise Macro F1 from 0.594 to 0.630 when tuned and measured on the same "
       "set, but under nested five-fold cross-validation the change was −0.009 to −0.015; it was not deployed.")

c.para("**6.4 Referral thresholds as a decision problem.** The system's practical output is a referral decision, and its two "
       "error types have very different costs: a missed cancer can delay treatment, while an unnecessary referral costs a "
       "dermatologist appointment and patient anxiety. On the validation set the current thresholds (U = 30%, R = 20%) refer "
       "98.0% of malignant lesions but also 44.6% of benign lesions; with the melanoma safety rule these figures become 98.7% "
       "and 51.0%. This is the central trade-off of the project, so the thresholds are justified here by an explicit cost "
       "model and a sensitivity analysis rather than by a team or advisor decision alone, following the threshold approach to clinical decisions [24] and decision-curve analysis [13].")
c.para("The validation set is 66% malignant, far more than a primary-care or general dermatology clinic. All rates are "
       "therefore re-weighted to an assumed clinic mix (8% malignant, 10% pre-malignant, 82% benign) and expressed per 1,000 "
       "patients. The cost model is the one used by the threshold simulator on the website: a missed malignant lesion costs c, "
       "a missed pre-malignant lesion c/4, a malignant lesion sent to routine instead of priority review c/20, an unnecessary "
       "priority referral 2 and an unnecessary routine referral 1. The cost ratio c, how many unnecessary referrals one missed "
       "cancer is 'worth', is a value judgement that the analysis makes visible rather than hides.")
c.table([
    ["Referral threshold R", "Malignant referred", "Missed malignant / 1,000", "Referred / 1,000", "Benign lesions referred"],
    ["10%", "99.0%", "0.8", "675", "60.5%"],
    ["15%", "98.6%", "1.2", "589", "50.0%"],
    ["20% (current)", "98.0%", "1.6", "544", "44.6%"],
    ["25%", "97.7%", "1.9", "496", "39.2%"],
    ["30%", "97.0%", "2.4", "460", "34.8%"],
], widths=[3.4, 3.0, 3.4, 3.0, 3.4], caption="Effect of the referral threshold (physician mode, U = 30%, clinic mix 8% / 10%, melanoma rule not included)")
c.para("Moving the referral threshold from 20% to 30% would avoid about 84 referrals per 1,000 patients at the cost of about "
       "0.8 additional missed malignant lesions per 1,000, roughly 100 avoided referrals per extra missed cancer. Table 12 "
       "shows which thresholds minimise expected cost for different values of c and different patient mixes.")
c.table([
    ["Patient mix (malignant / pre-malignant)", "c = 20", "c = 50", "c = 100", "c = 200"],
    ["Low-risk clinic (5% / 10%)", "U 95%, R 65%", "U 95%, R 30%", "U 80%, R 20%", "U 75%, R 15%"],
    ["Typical clinic (8% / 10%)", "U 95%, R 65%", "U 90%, R 30%", "U 80%, R 20%", "U 65%, R 5%"],
    ["Referral clinic (15% / 15%)", "U 95%, R 30%", "U 80%, R 15%", "U 65%, R 5%", "U 40%, R 5%"],
], widths=[5.6, 2.6, 2.6, 2.6, 2.6], caption="Cost-minimising thresholds by cost ratio c and patient mix (physician mode)")
c.figure(f"{FIG}/fig_cost_curve.png", "Expected cost relative to the minimum (left) and missed malignant versus benign referrals per 1,000 patients (right) as a function of the referral threshold", 15.5)
c.para("Three conclusions follow. First, the current referral threshold of 20% is cost-optimal when a missed cancer is valued "
       "at about 100 unnecessary referrals in a typical or low-risk clinic; if the clinic values it at 50, a threshold of 30% "
       "is better, and in a referral clinic with more malignant cases a lower threshold is justified. Second, the priority "
       "threshold of 30% is the weaker part of the design: in the typical clinic mix it sends 381 of every 1,000 patients to "
       "the priority queue, although only 80 have a malignant lesion. The cost model prefers a much higher priority threshold "
       "because, under its assumptions, a malignant lesion seen through routine referral costs only c/20; this assumption, "
       "the acceptable waiting time for routine appointments, needs clinical input. This matches the decision-curve analysis on "
       "the website, where the model adds net benefit for the referral decision over a wide range of threshold probabilities "
       "(about 1–41%) but for the priority decision only up to about 14%. Third, differences between neighbouring thresholds "
       "rest on a few lesions (the difference between R = 20% and R = 30% is seven missed malignant lesions out of 691), so "
       "the choice should not be over-tuned to this validation set.")
c.para("The melanoma safety rule addresses a specific error: 12 of the 14 malignant lesions missed at the current thresholds "
       "were melanomas predicted as nevi. Table 13 shows its trade-off.")
c.table([
    ["Setting", "Missed melanoma (of 90)", "Missed malignant (of 691)", "Benign lesions referred"],
    ["Physician mode, no rule", "12", "14", "44.6%"],
    ["Physician mode, melanoma ≥ 10% (current)", "7", "9", "51.0%"],
    ["Physician mode, melanoma ≥ 5%", "4", "6", "57.1%"],
    ["Patient mode, no rule", "13", "20", "58.1%"],
    ["Patient mode, melanoma ≥ 15% (current)", "11", "18", "59.5%"],
], widths=[6.4, 3.2, 3.2, 3.4], caption="Trade-off of the melanoma safety rule on the validation set")
c.para("Based on this analysis, three operating points are proposed for the joint decision with the advisor (Table 14). "
       "Option A keeps the current safety-first setting; Option B keeps the same sensitivity but halves the pressure on the "
       "priority queue; Option C also reduces routine referrals at the cost of more missed cancers. The team's recommendation "
       "is Option B, subject to the advisor's view on the acceptable cost ratio and waiting time.")
c.table([
    ["Option (physician mode, + melanoma rule)", "U / R", "Malignant referred (no rule)", "Missed malignant / 1,000", "Priority / 1,000", "Referred / 1,000", "Benign referred"],
    ["A – current", "30% / 20%", "98.0%", "1.6", "381", "544", "44.6%"],
    ["B – recommended", "50% / 20%", "98.0%", "1.6", "295", "544", "44.6%"],
    ["C – fewer referrals", "50% / 30%", "97.0%", "2.4", "295", "460", "34.8%"],
], widths=[4.2, 1.9, 2.4, 2.2, 1.8, 1.8, 1.9], caption="Operating points prepared for the advisor decision (clinic mix 8% / 10%; values without the melanoma rule, which adds about 6 points of benign referrals)")

c.para("**6.5 Robustness of the patient mode.** Tuğalp Ata rebuilt the smartphone mode and tested it on the 433 held-out "
       "PAD-UFES-20 photos with controlled perturbations. When the lesion occupied only a small part of the frame and fabric was "
       "visible in the background, the predicted class changed in 47–67% of the photos and Macro F1 fell from 0.67 to "
       "0.18–0.38, with predictions drifting towards 'nevus'. In about one third of these photos the hottest Grad-CAM region lay "
       "on the fabric rather than on the lesion. Changes in lighting and JPEG quality had a small effect (5–10%). The cause is "
       "that training augmentation only zoomed in, never out, while the preprocessing crops the centre of the image.")
c.bullets([
    "**Implemented in the interface:** in patient mode a square crop is mandatory; the user is told to centre the lesion so "
    "that it fills about one third of the box and to keep fabric and objects outside, and cannot continue without confirming. "
    "The photo guidance gained a 'plain background' tip.",
    "**Implemented in the interface:** results whose deciding total is within 5 points of a threshold are flagged as "
    "borderline (7.6% of validation cases in physician mode, 9.9% in patient mode); patients are asked to retake one centred "
    "close-up and to follow the more cautious result. The referral itself is not changed.",
    "**Tested, not deployed:** averaging predictions over three crop scales (1.0 / 0.75 / 0.5) reduced referral changes on "
    "distant photos from 32% to 13%, but raised benign referrals on normal photos from 22% to 30%; it is not deployed until "
    "the thresholds are re-checked.",
    "**Future work:** retraining with zoom-out and random-background augmentation.",
])

c.para("**6.6 Skin-tone equity.** Figure 5 reports performance by MILK10k skin-tone group (0 = very dark, 5 = very light). "
       "Malignant lesions were referred in 95–100% of cases in every measurable group, but Macro F1 varied from 0.37 to 0.62. "
       "Macro F1 within small groups is unstable because rare classes are missing or represented by one or two lesions. The "
       "darkest groups (0–1) contain only 19 validation lesions, so performance on dark skin cannot be established with this "
       "dataset, and the added datasets are also mostly light-skinned. This remains the most important equity limitation.")
c.figure(f"{FIG}/fig_equity.png", "Physician-mode performance by skin-tone group (validation set)", 12)
c.para("**6.7 Usability and interface performance.** The SUS survey (with an informed-consent statement and mandatory "
       "agreement) has 15 responses from 22 to 27 November 2026: mean 79.3 (SD 18.9, 95% CI 68.8–89.8, range 40–100), which "
       "corresponds to 'good' usability on the Bangor scale. Two respondents are medical students and 13 are other users; "
       "no physician has answered yet, so clinical usability is not yet evidenced. Sixteen analyses were stored with consent "
       "(15 in patient mode). On the free cloud tier the models load in about 6 s once per session and an analysis takes "
       "about 2 s (Grad-CAM about 1 s), well within the ten-second target, with peak memory below 800 MB.")
c.table([
    ["Success criterion (Capstone I)", "Target", "Current value (v3)", "Status"],
    ["Macro F1", "≥ 0.45", "0.483 on the ISIC hidden test (validation 0.594; MILK10k only 0.546)", "Met"],
    ["Improvement over single modality", "≥ +0.05", "+0.069 (MILK10k only), +0.011 (extended)", "Met"],
    ["Optimisation gain over untuned baseline", "Improvement", "Calibration and cost-based thresholds; per-class tuning tested, no gain", "Partly met"],
    ["Melanoma recall (top-1)", "≥ 0.65", "0.656 (95% CI 0.55–0.75); 0.533 with MILK10k only", "Borderline"],
    ["Inference time", "≤ 10 s", "≈ 2 s", "Met"],
    ["Equity analysis", "Macro F1 per skin-tone band", "Reported; groups 0–1 too small", "Met"],
    ["Leaderboard submission", "≥ 1", "2 valid submissions (0.485, 0.483)", "Met"],
    ["Reproducibility", "≤ 4 h end-to-end", "Fresh re-run scheduled", "In progress"],
], widths=[4.6, 2.8, 6.0, 2.8], caption="Status of the Capstone I success criteria",
    shade_col=(3, {"Met": "D9EFE0", "Partly": "FBEFC9", "Borderline": "FBEFC9", "In": "FBEFC9"}))

# ------------------------------------------------------------------ 7 design
c = section(99, "The main design changes since Capstone I are listed below; Table 1 gives the reasons.")
c.bullets([
    "Training data extended beyond MILK10k (design decision change, Section 2.2).",
    "The deployed model is a calibrated ensemble of four separately trained image models (probability-level late fusion) "
    "instead of one end-to-end network; metadata does not enter the prediction.",
    "A decision layer was added between the model and the user: cost-based referral thresholds, an uncertainty flag, a "
    "borderline flag, conformal prediction sets and a melanoma safety rule.",
    "A patient mode for single smartphone photos was added, with a calm result view, a scope warning and a mandatory "
    "centred square crop.",
    "Persistence was added: anonymous results, physician reviews, an append-only audit log and survey answers are stored only "
    "with consent; images and personal data are never stored.",
])
c = section(101, "Figure 6 shows the current architecture. The user interacts with a Streamlit application that performs input "
                 "checks, prediction, decision support and reporting. Model weights and their configuration are downloaded once "
                 "from a private Hugging Face repository; anonymous results are written to a Supabase PostgreSQL database. "
                 "Training and evaluation run offline on Kaggle GPUs and never use the official test labels.")
c.figure(f"{FIG}/fig_architecture.png", "Current system architecture", 13.5)
c = section(103, "Table 16 lists the tools in use; the main changes are the switch to PyTorch and Kaggle and the addition of "
                 "deployment and database services.")
c.table([
    ["Category", "Capstone I plan", "Current"],
    ["Deep learning", "TensorFlow 2 / Keras", "PyTorch 2.5 (CPU build in deployment), timm (EfficientNet-B3, EfficientNetV2-S)"],
    ["Training platform", "Google Colab / Kaggle", "Kaggle (NVIDIA T4, free tier)"],
    ["Data processing and analysis", "NumPy, Pandas, OpenCV, Albumentations", "NumPy, Pandas, Pillow, scikit-learn"],
    ["Interface", "Streamlit", "Streamlit 1.50 on Streamlit Community Cloud; Altair charts"],
    ["Model hosting", "–", "Hugging Face Hub (private model repository)"],
    ["Database", "–", "Supabase (PostgreSQL, row-level security, service-role access only)"],
    ["Reports and explainability", "Matplotlib, Seaborn", "fpdf2 (PDF), segno (QR codes), Grad-CAM (own implementation), Matplotlib"],
    ["Collaboration", "Git / GitHub, Google Drive", "Git / GitHub, GitHub Desktop, private Kaggle dataset, WhatsApp"],
    ["AI assistance", "ChatGPT / Claude for ideation, summaries and proofreading", "Claude Code (Anthropic) for most code, analyses and report drafts; ChatGPT for occasional questions"],
    ["Standards and metrics", "Macro F1, IEEE citations", "Macro F1, balanced accuracy, ROC-AUC, ECE, SUS, Wilson and bootstrap intervals, IEEE citations"],
], widths=[3.8, 4.4, 8.0], caption="Tools and technologies (Capstone I plan versus current)")
c.para("**Development approach and use of AI.** Without software engineers in the fall team, the three IE members built the "
       "system with an AI coding assistant (Claude Code) as an implementation tool. The ideas, the methods and the design "
       "decisions came from the team, which also ran all training on Kaggle, executed and checked every analysis, and tested "
       "the application; the assistant wrote most of the code, the evaluation scripts and first drafts of the report text. Because AI output can be wrong in convincing ways, "
       "results were accepted only after independent checks: re-computation from the saved prediction files, automated page "
       "tests of every screen in both languages, comparison with Kaggle outputs, and live checks on the deployed site. Several "
       "errors were caught this way, for example an apparent threshold-tuning gain that disappeared under nested "
       "cross-validation and a reversed reading of the skin-tone scale. The same approach is declared on the AI-usage page.")

# ------------------------------------------------------------------ 8 risks
c = section(106, "Table 17 lists the Capstone I risks that materialised and how they were handled.")
c.table([
    ["Risk (Capstone I)", "P / I", "What happened", "Action taken"],
    ["Team member unavailable", "M / H", "The three SE members did not continue in Capstone II.",
     "SE tasks (interface, deployment, database) redistributed within the IE team; scope kept."],
    ["Leaderboard format error", "M / M", "Two submissions were rejected because the scorer expected a lesion_id column.",
     "Format corrected and documented; two valid submissions."],
    ["Severe imbalance harms rare classes", "H / M", "Other malignant (recall 0.00) and other benign (0.22) remain weak.",
     "Class weights and label smoothing; per-class reporting; melanoma safety rule."],
    ["Metadata-fusion problems", "M / M", "The fusion network did not outperform the image ensemble.",
     "Kept as an evaluated alternative; decision supported by the ablation."],
    ["GPU session limits", "H / M", "Weekly Kaggle quota constrained the number of runs.",
     "Fixed settings, background runs, image caching; no large hyperparameter search."],
    ["Macro F1 below target", "M / H", "Not realised (hidden test 0.483).", "–"],
], widths=[3.6, 1.3, 5.6, 5.7], caption="Capstone I risks that materialised")
c = section(108, "New risks identified during the project are shown in Table 18 (P = probability, I = impact).")
c.table([
    ["New risk", "P / I", "Description"],
    ["Validation results do not transfer", "H / H", "The hidden-test score did not follow the validation gain."],
    ["Referral overload", "H / M", "At clinic prevalence, 54% of patients are referred and 38% flagged as priority."],
    ["Missed melanomas resembling nevi", "M / H", "Twelve of 14 missed malignant lesions were melanomas predicted as nevi."],
    ["Framing and background sensitivity", "H / M", "Small lesions on fabric change the patient-mode prediction in about half of photos."],
    ["Health anxiety in patient mode", "H / M", "About half of benign lesions receive a referral recommendation."],
    ["Out-of-scope inputs", "M / M", "Rashes, eczema or non-skin photos are forced into one of the eleven classes."],
    ["Under-representation of dark skin", "H / M", "Only 19 validation lesions in skin-tone groups 0–1."],
    ["Overfitting decisions to the validation set", "M / M", "Threshold and model choices made on the same 1,048 lesions."],
    ["Changes to free-tier services", "M / M", "Hugging Face Docker became paid; Supabase pauses inactive projects."],
    ["Reliance on AI-generated code", "M / M", "The team may not know every implementation detail; AI output can contain subtle errors."],
], widths=[5.2, 1.3, 9.7], caption="New risks")
c = section(110, "Table 19 gives the updated mitigation plan for the most important open risks.")
c.table([
    ["Risk", "Mitigation", "Contingency"],
    ["Results do not transfer", "Hidden test as the main claim; validation figures labelled as supporting", "External validation (future work)"],
    ["Referral overload", "Cost and sensitivity analysis; operating points prepared for the advisor decision", "Raise the priority threshold (Option B)"],
    ["Missed melanomas", "Melanoma safety rule; Grad-CAM and possible-diagnosis list shown to physicians", "Review the rule threshold with the advisor"],
    ["Framing sensitivity", "Mandatory centred crop; borderline flag; photo guidance", "Zoom-out augmentation in retraining"],
    ["Health anxiety", "Action-first patient view; absolute numbers; details behind an expander", "Raise patient-mode thresholds if feedback shows alarm"],
    ["Out-of-scope inputs", "Scope question and warning; skin-likeness check", "Out-of-scope detector (future work)"],
    ["Dark-skin under-representation", "Transparent reporting; low-data warning for skin tones 0–1", "Evaluation on DDI / Fitzpatrick17k (future work)"],
    ["Validation overfitting", "Pre-registered adoption rules; nested CV for tuning", "Use cross-validated estimates only"],
    ["Free-tier changes", "Weights in a private repository; check database and model status before presentations", "Demo mode without database"],
    ["AI-generated code", "Independent checks of every result; automated page tests; written hand-over notes for the code", "Simplify features that cannot be verified"],
], widths=[3.8, 7.6, 4.8], caption="Revised risk mitigation plan")

# ------------------------------------------------------------------ 9 challenges
set_text(P[113], "The main technical challenges and their solutions were:")
c = Cursor(P[113])
c.bullets([
    "**Deployment limits.** The free cloud tier offers about 1 GB of memory. CPU builds of PyTorch, loading only the models in "
    "use and resizing uploads to 1,600 px kept peak memory below 800 MB.",
    "**Data leakage between datasets.** ISIC 2019 shares images with MILK10k; identifier matching plus a calibrated perceptual "
    "hash removed 33 near-duplicates before training.",
    "**Comparing models fairly.** Every experiment reused the same split file and validation lesions, and adoption rules were "
    "fixed before results were seen.",
    "**Real-world photos.** Smartphone photos differ from the professional close-ups of MILK10k in framing and background; the "
    "robustness test turned this into concrete interface rules.",
    "**Benchmark format.** The official scorer accepted only a lesion_id column, which was not clear from the documentation.",
])
set_text(P[115], "The main management challenge was team capacity. The joint IE–SE plan of Capstone I assumed six students, while "
                 "Capstone II is completed by three in a compressed irregular term. The interface and "
                 "deployment work was therefore taken on by the IE team with an AI coding assistant, and the scope was protected by reusing free services "
                 "and fixed training settings. Work developed in parallel on separate branches had to be integrated carefully. "
                 "Communication was mainly asynchronous (WhatsApp), which was efficient for small decisions but slower for "
                 "larger ones; written result summaries after each experiment kept everyone aligned.")
set_text(P[117], "Key lessons so far:")
c = Cursor(P[117])
c.bullets([
    "Validation results must be confirmed on independent data; the hidden test showed that a +0.06 validation gain can disappear.",
    "Accuracy can hide weak classes in imbalanced data; balanced accuracy and per-class recall tell the real story.",
    "A threshold is a value judgement: making the cost ratio explicit turns an argument into a decision that can be discussed.",
    "Tuning on the evaluation set produces convincing but false gains; nested cross-validation is necessary.",
    "How a photo is taken matters as much as the model; interface rules are a cheap and effective robustness measure.",
    "An AI coding assistant let a three-person IE team build and deploy a full application, but only because every output "
    "was checked; verification, not code writing, became the main engineering task.",
    "A written task plan with named deliverables, dates and fallback options kept the project on schedule.",
])

# ------------------------------------------------------------------ 10 budget
c = section(119, "No money has been spent. All mandatory resources remained free, and the optional Colab Pro subscription was "
                 "not needed because Kaggle's free GPU quota was sufficient.")
c.table([
    ["Item", "Capstone I estimate (TRY)", "Actual (TRY)"],
    ["Software, datasets, GitHub, Kaggle (free tier)", "0", "0"],
    ["Streamlit Community Cloud, Hugging Face Hub, Supabase (free tiers)", "– (not planned)", "0"],
    ["Google Colab Pro (optional)", "≈ 390 / month", "0 (not needed)"],
    ["Report printing and binding", "≈ 150", "0 (digital submission only)"],
    ["Total", "≈ 0 (≈ 540 maximum)", "0 so far"],
], widths=[8.2, 4.0, 4.0], caption="Budget: Capstone I estimate versus actual")

# ------------------------------------------------------------------ 11 advisor
set_text(P[121], "Communication with the advisor has mainly taken place through WhatsApp messages, with an in-person meeting in "
                 "November 2026. On 27 November 2026 the advisor reviewed the deployed system and sent written feedback. She "
                 "found the system transparent end to end and highlighted the 'not a diagnosis' banner, the out-of-scope warning, "
                 "the threshold simulator, the calibration analysis and the equity analysis, and concluded that the system exceeds its "
                 "target and that the remaining work is mainly for the report to reflect this maturity. She will also try the PDF "
                 "report and the physician mode separately. Her requests and how they were addressed are listed below.")
c = Cursor(P[121])
c.table([
    ["Advisor request", "How it is addressed", "Status"],
    ["Keep the additional data, but report the move away from a single dataset as an explicit design decision change, consistent with the website",
     "Section 2.2 and the website 'Data and limitations' page use the same justification", "Done"],
    ["Add a model × dataset table and confirm that ISIC 2019 does not overlap with the 1,048 validation lesions",
     "Table 2; duplicate check against all MILK10k images, including the validation set", "Done"],
    ["Add licence and attribution requirements of PAD-UFES-20 (CC BY 4.0) and ISIC 2019 (CC BY-NC 4.0)",
     "Section 2.2 and references [26]–[27]", "Done"],
    ["Use the numbers of one version consistently in report, website and presentation",
     "All figures stated 'as of v3'; old figures removed from the website", "Done"],
    ["Do not lead with accuracy; lead with the hidden-test Macro F1, balanced accuracy and per-class sensitivity, and discuss weak rare classes",
     "Sections 1 and 6.1–6.2 (Tables 8–9)", "Done"],
    ["Make the benign referral rate the main discussion and justify the thresholds with the cost curve and sensitivity analysis; finalise them together",
     "Section 6.4 (Tables 11–14, Figure 4); operating points prepared", "Meeting pending"],
    ["Keep the patient-identifier limitation; make the hidden test the main claim and cross-validation supporting",
     "Sections 3.2 and 6.1", "Done"],
    ["Check that the English version of the website is complete; add trusted sources (e.g. WHO, AAD) to the skin-health page",
     "All pages checked in English; AAD and WHO sources added", "Done"],
    ["Explain how the team worked together", "Sections 3.3, 4 and 9.2 describe the actual division of work; reply to the advisor in preparation", "Done"],
], widths=[6.6, 6.6, 3.0], caption="Advisor feedback (27 November 2026) and actions",
    shade_col=(2, {"Done": "D9EFE0", "Meeting": "FBEFC9"}))
c.para("Earlier guidance: the advisor defined the project as AI-supported early detection of skin cancer, provided MILK10k as "
       "the primary dataset and approved the use of additional public datasets in November 2026. The team previously agreed to "
       "keep the current thresholds and the melanoma rule; following the advisor's feedback this is now treated as a proposal "
       "to be finalised with her on the basis of Section 6.4. No ethics-committee application was made because no personal, "
       "health or identifiable data is collected; an informed-consent statement with a mandatory agreement box was added to "
       "the usability survey. A communication log is given in Appendix A.")

# ------------------------------------------------------------------ 12 conclusion
set_text(P[123], "The project has met its main technical objectives: a reproducible pipeline, a multimodal system that exceeds "
                 "the Macro F1 target on the official hidden test, a deployed decision-support application and a transparent "
                 "equity analysis. The additional datasets made the system clearly better for smartphone photos, while the "
                 "hidden test shows that generalisation gains are smaller than validation figures suggest. The most important "
                 "remaining question is not accuracy but the operating point: how many unnecessary referrals are acceptable to "
                 "avoid one missed cancer. Next steps:")
c = Cursor(P[123])
c.bullets([
    "Agree the final referral and priority thresholds and the melanoma rule with the advisor (Section 6.4).",
    "Collect physician feedback and complete the usability analysis.",
    "Add the full robustness tables and complete the reproducibility run.",
    "Prepare the final report and the 15–30 minute presentation with a live demonstration.",
    "Future work: external validation (DDI, Derm7pt), more diverse skin tones (Fitzpatrick17k), zoom-out augmentation and an "
    "out-of-scope detector.",
])

# ------------------------------------------------------------------ references
refs = [
    "P. Tschandl et al., “MILK10k: a hierarchical multimodal imaging-learning toolkit for diagnosing pigmented and nonpigmented skin "
    "cancer and its simulators,” J. Invest. Dermatol., vol. 146, no. 2, pp. 357–364.e7, 2026.",
    "MILK study team, “MILK10k,” ISIC Archive, 2025, doi: 10.34970/648456. Licence: CC BY-NC 4.0.",
    "MILK study team, “MILK10k Benchmark,” ISIC Challenge, 2025, doi: 10.34970/262082.",
    "A. G. C. Pacheco et al., “PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from "
    "smartphones,” Data in Brief, vol. 32, art. 106221, 2020.",
    "P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of "
    "common pigmented skin lesions,” Sci. Data, vol. 5, art. 180161, 2018.",
    "N. C. F. Codella et al., “Skin lesion analysis toward melanoma detection: a challenge at the 2017 ISBI, hosted by ISIC,” in Proc. "
    "IEEE ISBI, 2018, pp. 168–172.",
    "M. Combalia et al., “BCN20000: dermoscopic lesions in the wild,” Sci. Data, vol. 11, art. 641, 2024.",
    "M. Tan and Q. V. Le, “EfficientNet: rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, pp. 6105–6114.",
    "M. Tan and Q. V. Le, “EfficientNetV2: smaller models and faster training,” in Proc. ICML, 2021, pp. 10096–10106.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.",
    "A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and distribution-free uncertainty "
    "quantification,” arXiv:2107.07511, 2021.",
    "R. R. Selvaraju et al., “Grad-CAM: visual explanations from deep networks via gradient-based localization,” in Proc. IEEE ICCV, "
    "2017, pp. 618–626.",
    "A. J. Vickers and E. B. Elkin, “Decision curve analysis: a novel method for evaluating prediction models,” Med. Decis. Making, "
    "vol. 26, no. 6, pp. 565–574, 2006.",
    "J. Brooke, “SUS: a ‘quick and dirty’ usability scale,” in Usability Evaluation in Industry, P. W. Jordan et al., Eds. London, "
    "U.K.: Taylor & Francis, 1996, pp. 189–194.",
    "A. Bangor, P. Kortum, and J. Miller, “Determining what individual SUS scores mean: adding an adjective rating scale,” J. "
    "Usability Stud., vol. 4, no. 3, pp. 114–123, 2009.",
    "G. Gigerenzer et al., “Helping doctors and patients make sense of health statistics,” Psychol. Sci. Public Interest, vol. 8, "
    "no. 2, pp. 53–96, 2007.",
    "R. Daneshjou et al., “Disparities in dermatology AI performance on a diverse, curated clinical image set,” Sci. Adv., vol. 8, "
    "no. 32, eabq6147, 2022.",
    "M. Groh et al., “Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset,” "
    "in Proc. IEEE/CVF CVPR Workshops, 2021, pp. 1820–1828.",
    "J. Kawahara, S. Daneshvar, G. Argenziano, and G. Hamarneh, “Seven-point checklist and skin lesion classification using "
    "multitask multimodal neural nets,” IEEE J. Biomed. Health Inform., vol. 23, no. 2, pp. 538–546, 2019.",
    "V. Rotemberg et al., “A patient-centric dataset of images and metadata for identifying melanomas using clinical context,” Sci. "
    "Data, vol. 8, art. 34, 2021.",
    "A. Paszke et al., “PyTorch: an imperative style, high-performance deep learning library,” in Proc. NeurIPS, 2019, pp. 8024–8035.",
    "R. Wightman, “PyTorch image models (timm),” GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models",
    "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” J. Amer. Statist. Assoc., vol. 22, no. "
    "158, pp. 209–212, 1927.",
    "S. G. Pauker and J. P. Kassirer, “The threshold approach to clinical decision making,” N. Engl. J. Med., vol. 302, no. 20, "
    "pp. 1109–1117, 1980.",
    "M. E. Aksu, D. Y. Denizci, K. Develioğlu, T. Ata, B. Sevencan, and A. T. Yücesoy, “Design of a multimodal clinical decision-support system for skin lesion classification using the "
    "MILK10k dataset,” Capstone I project report, Dept. of Industrial Eng. and Dept. of Software Eng., Bahçeşehir Univ., Istanbul, "
    "June 2026.",
    "A. G. C. Pacheco et al., “PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from "
    "smartphones,” Mendeley Data, v1, 2020. Licence: CC BY 4.0 (attribution required).",
    "International Skin Imaging Collaboration, “ISIC 2019 Challenge training data,” ISIC Archive, 2019. Licence: CC BY-NC 4.0; "
    "users must cite [5], [6] and [7].",
]
c = section(125, f"[1] {refs[0]}")
for i, r in enumerate(refs[1:], start=2):
    c.para(f"[{i}] {r}", size=Pt(10))

# ------------------------------------------------------------------ appendices
c = section(130, "The team has coordinated through a WhatsApp group since the first week of November 2026, and communication "
                 "with the advisor took place mainly through WhatsApp. The log below lists the main events of the fall term.")
c.table([
    ["Date", "Channel", "Topic", "Outcome"],
    ["June 2026", "Submission", "Capstone I project report (joint IE–SE team)", "Report submitted; project plan, objectives and success criteria set"],
    ["First week of Nov 2026", "Team (WhatsApp)", "Start of Capstone II work by the three IE members", "Work started; division of work discussed"],
    ["14 Nov 2026", "Team (WhatsApp, shared document)", "Written task plan with work packages, deliverables and dates", "Accepted by all three members"],
    ["18–24 Nov 2026", "Team (WhatsApp, GitHub)", "Prototype, real models, database, ISIC submissions, additional-data models", "Results shared after each experiment (Section 3.2)"],
    ["November 2026", "In person", "Use of additional datasets; next steps", "Additional public datasets approved"],
    ["November 2026", "Team (WhatsApp)", "Thresholds, ethics, presentation", "Thresholds kept as a proposal; consent statement added to the survey; 15–30 min presentation with live demo"],
    ["27 Nov 2026", "WhatsApp (written)", "Advisor review of the deployed system", "Nine requests (Table 21); threshold decision to be taken jointly"],
    ["28 Nov 2026", "Team", "Report and website updated to the advisor's requests; robustness fixes deployed", "Progress report v2"],
], widths=[2.6, 2.8, 5.0, 5.8], caption="Advisor communication log")
c.para("Team coordination took place continuously in a WhatsApp group; decisions and results were also recorded in the GitHub "
       "repository history.")

c = section(133, "Code repository: https://github.com/ahmetycsy/Skin-analysis (MILK10k data, model weights and lesion-level "
                 "predictions are not included because of the CC BY-NC 4.0 licence and privacy considerations).")
c.para("Live application: https://bau-skin-cds.streamlit.app")
c.para("Main components of the repository:")
c.bullets([
    "app.py – Streamlit application (workflow, result view, case tracking, reports); i18n.py – Turkish and English texts.",
    "predictor.py, gradcam.py – model loading, calibrated ensemble prediction, conformal sets, referral thresholds and Grad-CAM.",
    "eval_page.py, stats_page.py, survey_page.py, health_page.py, about.py – model analysis, statistics, SUS survey, skin-health "
    "and model-card pages.",
    "report.py – PDF report; db.py, supabase_ek.sql – anonymous storage (PostgreSQL via Supabase).",
    "kaggle/ – training notebooks (MILK10k models, PAD-UFES-20 clinical models, ISIC 2019 dermoscopic models with duplicate check).",
    "analysis/ – evaluation scripts (thresholds, threshold sensitivity, comparisons, ablation, ROC-AUC, threshold tuning, "
    "evaluation summary).",
])
c.para("Model weights are kept in a private Hugging Face repository and can be shared with the advisor and examiners on request. "
       "The robustness-test scripts and tables are kept by Tuğalp Ata and will be added to the final report.")

c = section(136, "Class distribution of the MILK10k training data (5,240 lesions) and per-class validation results of the "
                 "deployed physician mode:")
counts = {"BCC": 2522, "NV": 746, "BKL": 544, "SCCKA": 473, "MEL": 450, "AKIEC": 303, "DF": 52, "INF": 50, "VASC": 47,
          "BEN_OTH": 44, "MAL_OTH": 9}
groups = {"BCC": "Malignant", "MEL": "Malignant", "SCCKA": "Malignant", "MAL_OTH": "Malignant", "AKIEC": "Pre-malignant"}
rec = {"AKIEC": 0.52, "BCC": 0.95, "BEN_OTH": 0.22, "BKL": 0.48, "DF": 0.90, "INF": 0.30, "MAL_OTH": 0.00, "MEL": 0.66, "NV": 0.87,
       "SCCKA": 0.69, "VASC": 0.78}
f1 = {"AKIEC": 0.62, "BCC": 0.91, "BEN_OTH": 0.29, "BKL": 0.54, "DF": 0.90, "INF": 0.33, "MAL_OTH": 0.00, "MEL": 0.69, "NV": 0.85,
      "SCCKA": 0.68, "VASC": 0.74}
auc = {"AKIEC": 0.947, "BCC": 0.965, "BEN_OTH": 0.754, "BKL": 0.879, "DF": 0.999, "INF": 0.981, "MAL_OTH": 0.844, "MEL": 0.948,
       "NV": 0.983, "SCCKA": 0.957, "VASC": 0.998}
rows = [["Class", "Group", "Lesions", "Share", "Val. recall", "Val. F1", "Val. ROC-AUC"]]
for k, n in counts.items():
    rows.append([k, groups.get(k, "Benign"), f"{n:,}", f"{n / 5240 * 100:.1f}%", f"{rec[k]:.2f}", f"{f1[k]:.2f}", f"{auc[k]:.3f}"])
c.table(rows, widths=[2.2, 2.8, 2.0, 1.8, 2.4, 2.2, 2.8], caption="MILK10k class distribution and per-class validation results (deployed physician mode)")

# ------------------------------------------------------------------ page breaks: no stand-alone break paragraphs
def only_break(el):
    txt = "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
    return txt == "" and any(b.get(qn("w:type")) == "page" for b in el.iter(qn("w:br")))


def is_empty(el):
    return el.tag == qn("w:p") and not "".join(t.text or "" for t in el.iter(qn("w:t"))).strip() \
        and not list(el.iter(qn("w:drawing"))) and not list(el.iter(qn("w:br")))


NO_BREAK = {"LIST OF TABLES", "1. Executive Summary"}  # flow on (avoid near-empty pages)
els = list(d.element.body.iterchildren())
for k, el in enumerate(els):
    if el.tag == qn("w:p") and only_break(el):
        nxt = next((e for e in els[k + 1:] if e.tag == qn("w:p") and "".join(t.text or "" for t in e.iter(qn("w:t"))).strip()), None)
        el.getparent().remove(el)
        if nxt is not None:
            np_ = Paragraph(nxt, d._body)
            title = np_.text.strip()
            if title not in NO_BREAK:
                np_.paragraph_format.page_break_before = True
# empty paragraphs left in front of appendix headings / after references
els = list(d.element.body.iterchildren())
for k, el in enumerate(els[:-1]):
    nxt = els[k + 1]
    if is_empty(el) and nxt.tag == qn("w:p"):
        ppr = nxt.find(qn("w:pPr"))
        if ppr is not None and ppr.find(qn("w:pageBreakBefore")) is not None:
            el.getparent().remove(el)

# table rows never split across pages; header rows repeat
for tbl in d.element.body.iter(qn("w:tbl")):
    for k, tr in enumerate(tbl.findall(qn("w:tr"))):
        trPr = tr.find(qn("w:trPr"))
        if trPr is None:
            trPr = OxmlElement("w:trPr")
            tr.insert(0, trPr)
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(OxmlElement("w:cantSplit"))
        if k == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(OxmlElement("w:tblHeader"))
# "1. Executive Summary" now follows the abbreviations: give it some space above
for par in d.paragraphs:
    if par.text.strip() == "1. Executive Summary":
        par.paragraph_format.space_before = Pt(24)

# ------------------------------------------------------------------ settings: update fields when opened in Word
settings = d.settings.element
upd = OxmlElement("w:updateFields")
upd.set(qn("w:val"), "true")
# schema order: updateFields must precede these elements (Word rejects misordered settings)
after = ["hdrShapeDefaults", "footnotePr", "endnotePr", "compat", "docVars", "rsids", "attachedSchema", "themeFontLang",
         "clrSchemeMapping", "doNotIncludeSubdocsInStats", "doNotAutoCompressPictures", "forceUpgrade", "captions",
         "readModeInkLockDown", "smartTagType", "shapeDefaults", "doNotEmbedSmartTags", "decimalSymbol", "listSeparator"]
nxt = next((ch for ch in settings if ch.tag.split("}")[-1] in after or ch.tag.endswith("mathPr") or ch.tag.endswith("schemaLibrary")), None)
if nxt is not None:
    nxt.addprevious(upd)
else:
    settings.append(upd)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
d.save(OUT)
print("saved", OUT, COUNT)
