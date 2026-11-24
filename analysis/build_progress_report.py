"""Fills the 4992 Capstone Progress Report template with the project's current status (draft for the team to revise)."""
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
SRC = os.path.join(HERE, "4992_ProgressReportTemplate.docx")
OUT = "/Users/ahmet/Desktop/vlaude code/raporlar/Capstone_Progress_Report_DRAFT.docx"
FIG = os.path.join(HERE, "figs")

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
set_keep_format(P[17], "Advisor: Asst. Prof. Zuhal Özcan Yavuz")
set_keep_format(P[18], "Department of Industrial Engineering")
set_keep_format(P[21], "November, 2026")

# ------------------------------------------------------------------ AI declaration
for i, used in zip(range(31, 36), (True, True, True, True, True)):
    p = P[i]
    p.runs[0].text = ("☒ " if used else "☐ ") + p.runs[0].text
set_text(P[36], "☒ Other (please specify): drafting of report text, tables and figures from the team's own results; all drafts "
                "were reviewed, verified against the project files and revised by the team.")

# project team table
rows = ["Department\t: Industrial Engineering", "Member 1\t: Tuğalp Ata", "Member 2\t: Bedirhan Sevencan",
        "Member 3\t: Ahmet Talat Yücesoy", "Advisor \t: Asst. Prof. Zuhal Özcan Yavuz (Industrial Engineering)",
        "Department 2\t: – (single-department project)", "", "", ""]
for cell, txt in zip(T_TEAM.columns[0].cells, rows):
    p = cell.paragraphs[0]
    runs = p.runs
    if runs:
        runs[0].text = txt
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(txt)

# front matter: student declaration, AI declaration and project team share one page (template put each on its own page)
def drop_page_break(par_el):
    for br in list(par_el.iter(qn("w:br"))):
        if br.get(qn("w:type")) == "page":
            br.getparent().remove(br)


body_els = list(d.element.body.iterchildren())
decl_idx = body_els.index(P[23]._p)  # "STUDENT DECLARATION"
team_idx = body_els.index(P[43]._p)  # "Project Team"
for el in body_els[decl_idx:team_idx]:
    if el.tag == qn("w:p"):
        drop_page_break(el)
# lists of tables, figures and abbreviations flow on (template gave each its own page)
lot_idx = body_els.index(P[53]._p)   # "LIST OF TABLES"
first_h1 = body_els.index(P[64]._p)  # "1. Executive Summary" (keeps its page break before)
abbr_idx = body_els.index(P[61]._p)
for el in body_els[lot_idx:abbr_idx]:
    if el.tag == qn("w:p"):
        drop_page_break(el)
for hp in (P[57], P[61]):
    hp.paragraph_format.space_before = Pt(18)
for hp in (P[29], P[43]):  # "DECLARATION OF AI USAGE", "Project Team": smaller gap above
    hp.paragraph_format.space_before = Pt(14)
for tr in list(T_TEAM.rows)[6:]:  # empty rows of the project-team table
    T_TEAM._tbl.remove(tr._tr)

# ------------------------------------------------------------------ abstract & keywords
set_text(P[46],
         "This capstone project designs, builds and evaluates a multimodal clinical decision-support (CDS) prototype that classifies "
         "skin lesions into the eleven diagnostic categories of the MILK10k dataset. The proposal set six objectives: a reproducible "
         "data pipeline, a dual-image model with patient metadata, systematic optimisation, a validation Macro F1 of at least 0.45, a "
         "web-based interface returning results within ten seconds, and a skin-tone equity analysis. At this stage all core technical "
         "work packages are complete and overall progress is estimated at about 80%. A stratified lesion-level split (4,192 training "
         "and 1,048 validation lesions) was created; four EfficientNet image models (clinical and dermoscopic, B3 and V2-S) were "
         "combined into a calibrated ensemble; a metadata fusion network was evaluated as an alternative; and a bilingual Streamlit "
         "application with case tracking, PDF reports, Grad-CAM explanations and an anonymous database was deployed online. Trained "
         "on MILK10k only, the system reached a validation Macro F1 of 0.546 (target 0.45) and 0.485 on the official ISIC hidden test "
         "set. With additional public data approved by the advisor (PAD-UFES-20 smartphone photos and ISIC 2019 dermoscopy), "
         "validation Macro F1 rose to 0.594 and phone-photo performance from 0.399 to 0.668, while the hidden-test score stayed at "
         "0.483, showing that validation gains do not automatically transfer. Per-class threshold tuning and metadata fusion gave no "
         "gain under cross-validation and were therefore not deployed. A melanoma safety rule reduced missed melanomas from 12 to 7 "
         "of 90. Remaining work covers usability testing (6 SUS responses so far, mean 89.2), a reproducibility run, external "
         "validation and the final report.", size=BODY)
set_text(P[48], "**Key Words:** clinical decision support, skin lesion classification, MILK10k, multimodal deep learning, model calibration")

# ------------------------------------------------------------------ abbreviations
abbr = ['AI – Artificial intelligence', 'AKIEC – Actinic keratosis / intraepidermal carcinoma', 'AUC – Area under the ROC curve', 'BCC – Basal cell carcinoma', 'BEN_OTH – Other benign proliferations', 'BKL – Benign keratinocytic lesion', 'CDS – Clinical decision support', 'CV – Cross-validation', 'DCA – Decision curve analysis', 'DF – Dermatofibroma', 'ECE – Expected calibration error', 'Grad-CAM – Gradient-weighted class activation mapping', 'INF – Inflammatory and infectious conditions', 'ISIC – International Skin Imaging Collaboration', 'MAL_OTH – Other malignant proliferations', 'MEL – Melanoma', 'NV – Melanocytic nevus', 'PAD-UFES-20 – Smartphone skin-lesion dataset of the Federal University of Espírito Santo', 'ROC – Receiver operating characteristic', 'SCCKA – Squamous cell carcinoma / keratoacanthoma', 'SUS – System Usability Scale', 'TTA – Test-time augmentation', 'VASC – Vascular lesions and haemorrhage', 'WP – Work package']
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
            "The project is on schedule and, for its technical deliverables, ahead of the proposal plan. Overall progress is estimated "
            "at **about 80%**: data engineering (WP2), baseline modelling (WP3) and model development (WP4) are complete; the "
            "decision-support interface (WP5) is deployed and publicly accessible, with usability testing in progress; evaluation and "
            "reporting (WP6) are about 60% complete.")
c.para("The main quantitative target of the proposal is met. Using MILK10k only, the deployed ensemble reached a validation Macro "
       "F1 of **0.546** (target ≥ 0.45) and **0.485** on the official ISIC MILK10k hidden test. After the advisor approved the use "
       "of additional public datasets, the clinical models were retrained with smartphone photos (PAD-UFES-20) and the dermoscopic "
       "models with ISIC 2019. This raised validation Macro F1 to **0.594**, reduced missed malignant lesions from 18 to 14 (out "
       "of 691) and raised phone-photo Macro F1 from 0.399 to 0.668, but did not change the hidden-test score (0.483). Both result "
       "sets are reported separately throughout this document.")
c.para("Two open decisions were settled by the team in November 2026: the current referral thresholds and the melanoma safety "
       "rule are kept as the final operating point (prioritising not missing cancer over fewer referrals), and, because no "
       "personal or identifiable data is collected, an informed-consent statement was added to the usability survey instead of "
       "a formal ethics-committee application. No critical issues are blocking the remaining work.")

# ------------------------------------------------------------------ 2 objectives
set_text(P[67], "The objectives below are restated from the approved proposal (January 2026). Section 2.2 explains the changes "
                "made during the project and why.")
c = section(69, "The proposal defined the following specific, measurable objectives:")
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
c = section(71, "All six objectives are retained. The following changes were made to how they are achieved, and two objectives "
                "were extended:")
c.table([
    ["Area", "Proposal", "Current approach", "Reason"],
    ["Data scope", "MILK10k only; other datasets out of scope",
     "MILK10k remains the primary dataset; PAD-UFES-20 and ISIC 2019 added for training (advisor approval, November 2026)",
     "Smartphone-photo performance was poor (Macro F1 0.399). MILK10k-only results are kept as the primary comparison."],
    ["Model architecture", "End-to-end late-fusion network (two images + metadata)",
     "Calibrated probability-level ensemble of four image models; metadata network kept as an evaluated alternative",
     "The fusion network (0.497) and metadata stacking did not outperform the image ensemble (0.546) under the same split."],
    ["Optimisation", "Hyperparameter search and per-class threshold tuning",
     "Temperature calibration, conformal prediction sets and cost-sensitive referral thresholds; per-class tuning tested",
     "Per-class tuning looked better on the validation set but lost 0.01–0.015 Macro F1 under nested cross-validation."],
    ["Output layer", "Per-class sigmoid with weighted binary cross-entropy",
     "Softmax with square-root class weights and label smoothing", "One diagnosis per lesion in MILK10k; better calibration."],
    ["Framework", "TensorFlow/Keras on Colab", "PyTorch + timm on Kaggle", "Pretrained EfficientNetV2 weights and longer free GPU sessions."],
    ["Interface (extended)", "Upload, metadata entry, probability display",
     "Deployed web application with patient mode, case tracking, physician review, PDF/QR reports, Grad-CAM and anonymous database",
     "Turns the model into a usable workflow tool, the industrial-engineering focus of the project."],
    ["Leaderboard", "One submission", "Two valid submissions (MILK10k-only and extended data)", "To report both result sets honestly."],
], widths=[2.6, 3.6, 5.0, 5.0], caption="Changes to the objectives and approach since the proposal")

# ------------------------------------------------------------------ 3.1 WP table
set_text(P[74], "Table 2 summarises the progress of each work package against the proposal's work breakdown structure "
                "(Table 5 of the proposal) and fifteen-week Gantt chart.")
wp = [["WP1", "Project management: coordination, milestone and risk tracking, GitHub, advisor communication", "Week 15", "75", "All members"],
      ["WP2", "Data engineering: download, EDA, preprocessing, imbalance handling, stratified split", "Week 4", "100", "Tuğalp Ata, all members"],
      ["WP3", "Baseline models: single-modality baselines, training loop, Macro F1 evaluation", "Week 6", "100", "Ahmet Talat Yücesoy, Tuğalp Ata"],
      ["WP4", "Model development: EfficientNet ensemble, metadata fusion network, ablation, calibration, extra-data models", "Week 11", "100",
       "Tuğalp Ata, Bedirhan Sevencan, Ahmet Talat Yücesoy"],
      ["WP5", "Decision-support interface: application, model integration, deployment, usability testing", "Week 13", "90", "Ahmet Talat Yücesoy, all members"],
      ["WP6", "Evaluation and reporting: metrics, equity analysis, ISIC submission, reproducibility, report and presentation", "Week 15", "60", "All members"]]
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
# caption above the template table
cap = Cursor(P[74])
cap.caption("Table", "Task-by-task progress against the proposal work packages")
# the template table carries number 3 in reading order: renumber below via field update; text reference uses 3
c = Cursor(P[75])
c.para("Actual status is the share of each work package's planned scope that is finished. WP5 is 90% because the usability "
       "study is still collecting responses; WP6 is 60% because the reproducibility test, external validation and final report "
       "remain.")

# ------------------------------------------------------------------ 3.2 detailed description
c = section(77, "**WP2 – Data engineering.** MILK10k provides 5,240 lesions, each with a clinical close-up and a dermoscopic "
                "image (10,480 images), and metadata on age (5-year bands), sex, skin tone (0 = very dark to 5 = very light) and "
                "anatomical site. The class distribution is highly imbalanced: basal cell carcinoma accounts for 48% of lesions, while "
                "other malignant proliferations have only 9 (Appendix C). Because the dataset has no patient identifier, a lesion-level "
                "stratified 80/20 split with seed 42 was created (4,192 training and 1,048 validation lesions) and frozen in a split "
                "file that every later experiment reuses. The column names were detected automatically by the loading code, and the "
                "official test files were never used for training.")
c.para("**WP3 – Baselines.** A first dermoscopic EfficientNet-B3 baseline trained on MILK10k reached a validation Macro F1 of "
       "0.477 (accuracy 71.9%) and established the training and evaluation loop. Single-modality ensembles (dermoscopic only and "
       "clinical only) serve as the baselines of the ablation study in Section 6.")
c.para("**WP4 – Model development.** Four ImageNet-pretrained models were trained with identical settings (12 epochs, learning "
       "rate 3×10⁻⁴, batch size 24, square-root class weights, label smoothing 0.05, flip test-time augmentation): EfficientNet-B3 "
       "(300 px) and EfficientNetV2-S (384 px) for each image type. Five-fold cross-validation gave a physician-mode Macro F1 of "
       "0.542 and a clinical-only (patient-mode) Macro F1 of 0.463. Probabilities are averaged within each image type and then "
       "across the two types, and temperature scaling fitted on out-of-fold predictions calibrates the output (T = 0.60 for "
       "physician mode, 0.76 for patient mode). Split-conformal prediction provides a list of possible diagnoses that contains the "
       "true class about 90% of the time (validation coverage 90.5%, mean set size 1.46).")
c.para("In parallel, a metadata model and a multimodal fusion network that combines image features with age, sex, skin tone "
       "and site were developed. On the same validation fold the fusion network reached 0.497, compared with 0.547 for the image "
       "ensemble. A cross-validated stacking experiment confirmed the pattern: metadata adds a small signal (+0.007 to +0.009 Macro "
       "F1, +0.013 to +0.018 AUC over the same model without metadata), but not enough to beat simple averaging of the image "
       "models. This comparison was essential for the final design decision; metadata is still collected in the interface for "
       "reporting, low-data warnings and the equity analysis.")
c.para("**Additional data (advisor-approved).** Two pre-registered experiments tested whether public data could strengthen the "
       "weakest parts of the system, using the same settings and the same 1,048 MILK10k validation lesions. A model was adopted "
       "only if it met rules fixed before training (for ISIC 2019: physician Macro F1 at least +0.01, missed malignant lesions not "
       "increased, calibration error ≤ 0.03).")
c.bullets([
    "**PAD-UFES-20** (2,298 smartphone photos, six classes mapped to MILK10k categories) was added to the clinical models, with "
    "20% of its patients held out as an independent phone-photo test set and phone-style augmentation (blur, JPEG compression, "
    "wider crops).",
    "**ISIC 2019** (25,331 dermoscopic images; unknown class excluded, nevi capped at 4,000) was added to the dermoscopic models. "
    "To prevent leakage into the validation set, images duplicated in MILK10k were removed by ISIC identifier and by a 256-bit "
    "perceptual hash (distance ≤ 22, calibrated on MILK10k): 33 near-duplicates were removed and 16,423 images used.",
])
c.para("**Decision layer.** The interface turns probabilities into a recommendation. A lesion is flagged for priority review when "
       "the total malignant probability is at least 30%, and for dermatology review when malignant plus pre-malignant probability "
       "is at least 20% or the model is uncertain (top probability below 40% or a gap below 10 points). These values were chosen "
       "on the validation set with a cost analysis (missed cancers weighted far more heavily than unnecessary referrals) and a "
       "decision-curve analysis re-weighted to a typical outpatient case mix. After the advisor meeting a melanoma safety rule was "
       "added (Section 6).")
c.para("**WP5 – Decision-support interface.** The Streamlit application is deployed at bau-skin-cds.streamlit.app in Turkish and "
       "English. Its main features are listed below.")
c.bullets([
    "A three-step workflow (images, patient information, result) with crop, rotate and image-quality checks (resolution, "
    "sharpness, lighting, skin-likeness).",
    "Two modes: physician mode with both images, and patient mode with one smartphone photo.",
    "Results with a recommendation, an uncertainty warning, possible diagnoses, a 'How reliable is this result?' box based on "
    "validation precision, and a Grad-CAM heat map.",
    "Case tracking with physician review (PIN-protected role), an ABCDE checklist, follow-up dates, lesion timelines and an audit "
    "log.",
    "PDF reports with QR codes, calendar reminders and a model-card section documenting performance and limitations.",
    "Anonymous storage of results in a PostgreSQL database (Supabase), only with user consent and never images or personal data.",
])
c.para("Patient mode was redesigned for safe risk communication: it leads with an action ('show it to a dermatologist within "
       "one to two weeks') rather than cancer percentages, explains in absolute numbers that most referred patients do not have "
       "cancer, asks five symptom questions the photo cannot answer, and warns that the tool is not designed for rashes or eczema. "
       "Detailed probabilities remain available behind a 'Details' section.")
c.para("**WP6 – Evaluation.** Completed: validation metrics, ablation, ROC-AUC, calibration, decision-curve and threshold "
       "analyses, skin-tone, sex and age subgroup analysis, and two valid ISIC benchmark submissions. In progress: usability "
       "testing and a reproducibility test, in which a team member re-runs a training notebook in a fresh Kaggle environment.")

# ------------------------------------------------------------------ 3.3 contributions
set_text(P[79], "All three members contributed comparable effort during this period. Each member led the parts listed below "
                "and supported the others through shared reviews and decisions, so effort is reported as equal.")
contrib = [
    ["Tuğalp Ata", "Industrial Eng.",
     "Exploratory data analysis and verification of the stratified split; training of the four image models with five-fold "
     "cross-validation; temperature calibration, conformal prediction sets and cost-sensitive decision analysis; data, literature "
     "and baseline sections of the reports.", "33"],
    ["Bedirhan Sevencan", "Industrial Eng.",
     "Metadata model and the multimodal fusion network combining image features with patient information; confusion-matrix and "
     "error analysis of the fusion approach; the fusion-versus-ensemble comparison that shaped the final architecture; method, "
     "results and equity sections of the reports.", "33"],
    ["Ahmet Talat Yücesoy", "Industrial Eng.",
     "First dermoscopic baseline; decision-support web application and its deployment (Streamlit Cloud, Hugging Face, Supabase); "
     "Grad-CAM integration; additional-data experiments (PAD-UFES-20, ISIC 2019) and ISIC benchmark submissions; evaluation "
     "dashboards; system design, interface and ethics sections of the reports.", "33"],
]
t = T_CONTRIB
for i, row in enumerate(contrib, start=1):
    for j, val in enumerate(row):
        clear(t.cell(i, j).paragraphs[0])
        add_runs(t.cell(i, j).paragraphs[0], val, TBL)
for tr in list(t.rows)[4:]:
    t._tbl.remove(tr._tr)
Cursor(P[79]).caption("Table", "Individual contributions during the reporting period")

# ------------------------------------------------------------------ 4 collaboration
set_text(P[81], "This is a single-department project: all three members and the advisor belong to the Department of "
                "Industrial Engineering. This section therefore describes collaboration within the team.")
set_text(P[83], "The team coordinated mainly through a WhatsApp group for day-to-day communication, complemented by short calls "
                "when decisions were needed. Technical work was shared through a GitHub repository with one branch per member, a "
                "private Kaggle dataset holding MILK10k and the fixed split file so that every member trained on identical data, "
                "and a private Hugging Face repository for model weights. Written summaries of results were shared after each "
                "experiment so that decisions could be taken asynchronously.")
set_text(P[85], "The main integration challenge was combining work developed in parallel. Branches created from different "
                "versions of the code could not be merged directly without overwriting newer work, so results were integrated "
                "selectively: model weights, prediction files and metrics were imported instead of whole branches. To decouple "
                "training from the application, the team agreed on a single model-configuration file (model list, input sizes, "
                "temperatures, conformal thresholds) that the interface reads, so new models can be deployed without code changes. "
                "A further challenge was the change of Hugging Face Docker Spaces to a paid service, which was resolved by keeping "
                "the site on Streamlit Community Cloud and loading the weights from a free private model repository.")
set_text(P[87], "Deliverables that required input from all members are the deployed model package (four image models with "
                "calibration and conformal settings), the validation prediction file used by every evaluation, the evaluation "
                "dashboards, and this report. All are complete except the final report.")

# ------------------------------------------------------------------ 5 schedule
c = section(90, "The project is being completed in an irregular term, so the calendar dates differ from the proposal; Table 4 "
                "compares progress against the proposal's milestones using its fifteen-week structure.")
c.table([
    ["Milestone", "Planned", "Status", "Comment"],
    ["M1 Preprocessing pipeline validated", "Week 3", "Completed", "Split file fixed (seed 42) and reused by all experiments."],
    ["M2 Baseline Macro F1 recorded", "Week 6", "Completed", "Dermoscopic baseline 0.477; single-modality baselines in ablation."],
    ["M3 Fusion model and ablation complete", "Week 11", "Completed", "Ensemble selected; fusion network and metadata evaluated; ablation table finalised."],
    ["M4 Interface functional", "Week 13", "Completed (ahead)", "Deployed online with real models and database."],
    ["M5 Leaderboard submission and final report", "Week 15", "In progress", "Leaderboard done (two valid submissions); final report pending."],
], widths=[4.6, 1.8, 2.8, 7.0], caption="Proposal milestones and current status",
    shade_col=(2, {"Completed": "D9EFE0", "In": "FBEFC9"}))
c = section(92, "Figure 1 shows the updated schedule mapped onto the proposal's fifteen weeks. Green bars are completed, yellow "
                "in progress, red delayed (none) and blue planned.")
c.figure(f"{FIG}/fig_gantt.png", "Updated Gantt chart (weeks of the proposal plan)", 14.5)
c = section(94, "The remaining work is listed below in order of priority. The final report is due on 6 December 2026 (23:59), "
                "so all remaining tasks are scheduled before that date.")
c.table([
    ["Task", "Description", "Target", "Owner"],
    ["Usability study", "Reach at least 8–12 SUS responses, including physicians or medical students; summarise feedback", "by 2 Dec 2026", "All"],
    ["Reproducibility test", "Fresh Kaggle re-run of a training notebook; compare Macro F1 and run time (target ≤ 4 h)", "by 26 Nov 2026", "Ahmet Talat Yücesoy"],
    ["Live demonstration", "Prepare and rehearse the 15–30 minute final presentation with a live demo of the deployed system", "by 6 Dec 2026", "All"],
    ["External validation", "Test the system on independent datasets (Derm7pt, DDI) without retraining, subject to data access", "by 3 Dec 2026", "All"],
    ["Optional model work", "Clinical model with more diverse skin tones (Fitzpatrick17k); more melanoma data (ISIC 2020) — only if time allows", "by 3 Dec 2026", "All"],
    ["Final deliverables", "Final report (due 6 Dec 2026, 23:59), presentation and short demo video", "by 6 Dec 2026", "All"],
], widths=[3.2, 8.0, 2.2, 2.8], caption="Remaining work plan")

# ------------------------------------------------------------------ 6 results
c = section(96, "All results below were measured on the same 1,048 MILK10k validation lesions unless stated otherwise. Results "
                "obtained with MILK10k only and with the additional datasets are shown separately.")
c.table([
    ["Version", "Physician Macro F1", "Physician accuracy", "Missed malignant (of 691)", "Patient Macro F1", "Phone-photo test", "ISIC hidden test"],
    ["v1 MILK10k only", "0.546", "76.8%", "19", "0.484", "0.399", "0.485"],
    ["v2 + PAD-UFES-20 (clinical)", "0.535", "77.1%", "18", "0.496", "0.668", "–"],
    ["v3 + ISIC 2019 (dermoscopic), deployed", "0.594", "80.1%", "14", "0.496", "0.668", "0.483"],
], widths=[4.4, 2.0, 2.0, 2.2, 1.9, 1.9, 1.8], caption="Performance of the three model versions")
c.figure(f"{FIG}/fig_versions.png", "Macro F1 of the three model versions on validation, phone-photo and hidden test sets", 12)
c.para("The extended-data version improved every validation measure: accuracy rose from 77.1% to 80.1%, missed malignant "
       "lesions fell from 18 to 14, and unnecessary referral of benign lesions fell from 49.0% to 44.6%, with unchanged "
       "calibration (ECE 0.020 → 0.021). On the independent phone-photo test (433 photos of 277 held-out patients), Macro F1 rose "
       "from 0.399 to 0.668 (95% bootstrap CI of the difference +0.18 to +0.34), accuracy from 46% to 78% and calibration error "
       "fell from 0.123 to 0.046. However, the official ISIC hidden-test score did not improve (0.485 → 0.483). Likely reasons are "
       "a different class distribution in the hidden test (the model predicts far more squamous cell carcinoma and actinic "
       "keratosis there), optimistic validation estimates because the best epoch was chosen on the validation set, and the strong "
       "influence of a few rare-class lesions on Macro F1 with only 479 lesions. The validation gain should therefore be read as "
       "an upper bound.")
c.table([
    ["Input", "Macro F1 (MILK10k only)", "Macro F1 (extended data)", "Macro ROC-AUC (MILK10k only / extended)"],
    ["Dermoscopic image only", "0.477", "0.583", "0.904 / 0.931"],
    ["Clinical image only", "0.484", "0.496", "0.882 / 0.862"],
    ["Both images (deployed ensemble)", "0.546", "0.594", "0.922 / 0.932"],
    ["Both images, logistic stacking (5-fold CV)", "0.438", "0.501", "0.865 / 0.906"],
    ["Both images + metadata, logistic stacking (5-fold CV)", "0.447", "0.508", "0.883 / 0.919"],
], widths=[6.2, 3.0, 3.0, 4.0], caption="Ablation study: contribution of each input")
c.para("The ablation confirms the multimodal design: combining both images improves Macro F1 by 0.06–0.07 over the best single "
       "image (proposal criterion ≥ +0.05). Adding metadata improves a like-for-like stacked model slightly, but the simple "
       "average of the image models remains best on this data size.")
c.para("**Threshold tuning.** Tuning one offset per class on the validation set appeared to raise Macro F1 from 0.594 to 0.630. "
       "Under nested five-fold cross-validation (tuning on four folds, measuring on the fifth, three repetitions), the change was "
       "−0.009 to −0.015, so the apparent gain was overfitting and the method was not deployed.")
c.figure(f"{FIG}/fig_confusion.png", "Confusion matrix of the deployed physician mode (counts; colour = share of each true class)", 10)
c.para("Per-class F1 is high for basal cell carcinoma (0.91), dermatofibroma (0.90) and nevi (0.85), moderate for melanoma (0.69), "
       "squamous cell carcinoma (0.68) and actinic keratosis (0.62), and low for the rarest groups: other benign (0.29), "
       "inflammatory (0.33) and other malignant proliferations (0.00, two validation lesions). The most frequent errors are "
       "between visually similar classes: squamous cell carcinoma and actinic keratosis, melanoma and nevus, and benign "
       "keratinocytic lesions and basal cell carcinoma.")
c.para("**Referral performance and melanoma safety rule.** With the current thresholds the physician mode refers 98.0% of "
       "malignant lesions for dermatology review (96.4% as priority). Twelve of the 14 missed malignant lesions were melanomas "
       "that the model mistook for nevi. A rule that also refers a lesion when its melanoma probability exceeds a safety threshold "
       "was therefore evaluated (Table 8) and adopted, with a lower threshold in physician mode and a more cautious one in "
       "patient mode to limit false alarms for anxious users; the team confirmed these values as the final operating point. "
       "Because the thresholds were chosen on the validation set, the figures are somewhat optimistic.")
c.table([
    ["Setting", "Missed melanoma (of 90)", "Missed malignant (of 691)", "Benign lesions referred"],
    ["Physician mode, no rule", "12", "14", "44.6%"],
    ["Physician mode, melanoma ≥ 10% (adopted)", "7", "9", "51.0%"],
    ["Physician mode, melanoma ≥ 5%", "4", "6", "57.1%"],
    ["Patient mode, no rule", "13", "20", "58.1%"],
    ["Patient mode, melanoma ≥ 15% (adopted)", "11", "18", "59.5%"],
], widths=[6.4, 3.2, 3.2, 3.4], caption="Trade-off of the melanoma safety rule on the validation set")
c.para("**Skin-tone equity.** Figure 4 reports performance by MILK10k skin-tone group (0 = very dark, 5 = very light). "
       "Malignant lesions were referred in 95–100% of cases in every measurable group, but Macro F1 varied from 0.37 to 0.62. "
       "Macro F1 within small groups is unstable because rare classes are missing or represented by one or two lesions; accuracy "
       "(76–82%) is more stable. The darkest groups (0–1) contain only 19 validation lesions, so performance on dark skin cannot "
       "be established with this dataset. This is the most important equity limitation and motivates the planned external "
       "validation on DDI, a dataset with many dark-skin images.")
c.figure(f"{FIG}/fig_equity.png", "Physician-mode performance by skin-tone group (validation set)", 12)
c.para("**Usability and interface performance.** The preliminary SUS survey has 6 responses with a mean score of 89.2; this is "
       "too few to draw conclusions, and the target is 8–12 responses. On the free cloud tier, the models load in about 6 s once "
       "per session and an analysis takes about 2 s (Grad-CAM about 1 s), well within the ten-second target, with peak memory below "
       "800 MB.")
c.table([
    ["Success criterion (proposal)", "Target", "Current value", "Status"],
    ["Validation Macro F1", "≥ 0.45", "0.546 (MILK10k only); 0.594 (extended)", "Met"],
    ["Improvement over single modality", "≥ +0.05", "+0.069 (MILK10k only)", "Met"],
    ["Optimisation gain over untuned baseline", "Improvement", "Calibration and referral thresholds; per-class tuning tested, no gain", "Partly met"],
    ["Melanoma recall (top-1)", "≥ 0.65", "0.533 (MILK10k only); 0.656 (extended, 95% CI 0.55–0.75)", "Borderline"],
    ["Inference time", "≤ 10 s", "≈ 2 s", "Met"],
    ["Equity analysis", "Macro F1 per skin-tone band", "Reported; groups 0–1 too small", "Met"],
    ["Leaderboard submission", "≥ 1", "2 valid submissions (0.485, 0.483)", "Met"],
    ["Reproducibility", "≤ 4 h end-to-end", "Fresh re-run in progress", "In progress"],
], widths=[4.6, 2.8, 6.0, 2.8], caption="Status of the proposal's success criteria",
    shade_col=(3, {"Met": "D9EFE0", "Partly": "FBEFC9", "Borderline": "FBEFC9", "In": "FBEFC9"}))

# ------------------------------------------------------------------ 7 design
c = section(99, "The main design changes since the proposal are listed below; Table 1 gives the reasons.")
c.bullets([
    "The deployed model is a calibrated ensemble of four separately trained image models (probability-level late fusion) instead "
    "of one end-to-end network; metadata does not enter the prediction.",
    "A decision layer was added between the model and the user: calibrated referral thresholds, an uncertainty flag, conformal "
    "prediction sets and a melanoma safety rule.",
    "A patient mode for single smartphone photos was added, with its own calm result view and scope warning.",
    "Persistence was added: anonymous results, physician reviews, an append-only audit log and survey answers are stored only "
    "with consent; images and personal data are never stored.",
])
c = section(101, "Figure 5 shows the current architecture. The user interacts with a Streamlit application that performs input "
                 "checks, prediction, decision support and reporting. Model weights and their configuration are downloaded once "
                 "from a private Hugging Face repository; anonymous results are written to a Supabase PostgreSQL database. Training "
                 "and evaluation run offline on Kaggle GPUs and never use the official test labels.")
c.figure(f"{FIG}/fig_architecture.png", "Current system architecture", 13.5)
c = section(103, "Table 10 lists the tools in use; the main changes from the proposal are the switch to PyTorch and Kaggle and the "
                 "addition of deployment and database services.")
c.table([
    ["Category", "Proposal", "Current"],
    ["Deep learning", "TensorFlow 2 / Keras", "PyTorch 2.5 (CPU build in deployment), timm (EfficientNet-B3, EfficientNetV2-S)"],
    ["Training platform", "Google Colab / Kaggle", "Kaggle (NVIDIA T4, free tier)"],
    ["Data processing and analysis", "NumPy, Pandas, OpenCV, Albumentations", "NumPy, Pandas, Pillow, scikit-learn"],
    ["Interface", "Streamlit", "Streamlit 1.50 on Streamlit Community Cloud; Altair charts"],
    ["Model hosting", "–", "Hugging Face Hub (private model repository)"],
    ["Database", "–", "Supabase (PostgreSQL, row-level security, service-role access only)"],
    ["Reports and explainability", "Matplotlib, Seaborn", "fpdf2 (PDF), segno (QR codes), Grad-CAM (own implementation), Matplotlib"],
    ["Collaboration", "Git / GitHub", "Git / GitHub, GitHub Desktop, WhatsApp"],
    ["Standards and metrics", "Macro F1, IEEE citations", "Macro F1, ROC-AUC, ECE, SUS, Wilson and bootstrap intervals, IEEE citations"],
], widths=[3.8, 4.4, 8.0], caption="Tools and technologies (proposal versus current)")

# ------------------------------------------------------------------ 8 risks
c = section(106, "Table 11 lists the proposal risks that materialised and how they were handled.")
c.table([
    ["Risk (proposal)", "P / I", "What happened", "Action taken"],
    ["Leaderboard format error", "M / M", "Two submissions were rejected because the scorer expected a lesion_id column.",
     "Format corrected and documented; two valid submissions."],
    ["Severe imbalance harms rare classes", "H / M", "Other malignant (F1 0.00) and other benign (0.29) remain weak.",
     "Class weights and label smoothing; per-class reporting; melanoma safety rule."],
    ["Metadata-fusion problems", "M / M", "The fusion network did not outperform the image ensemble.",
     "Kept as an evaluated alternative; decision supported by the ablation."],
    ["GPU session limits", "H / M", "Weekly Kaggle quota constrained the number of runs.",
     "Background runs, image caching, a fixed number of epochs."],
    ["Macro F1 below target", "M / H", "Not realised (0.546 with MILK10k only).", "–"],
], widths=[3.6, 1.3, 5.6, 5.7], caption="Proposal risks that materialised")
c = section(108, "New risks identified during the project are shown in Table 12 (P = probability, I = impact).")
c.table([
    ["New risk", "P / I", "Description"],
    ["Validation results do not transfer", "H / H", "The hidden-test score did not follow the validation gain."],
    ["Missed melanomas resembling nevi", "M / H", "Twelve of 14 missed malignant lesions were melanomas predicted as nevi."],
    ["False alarms and health anxiety in patient mode", "H / M", "44% of benign lesions receive a priority recommendation in patient mode."],
    ["Out-of-scope inputs", "M / M", "Rashes, eczema or non-skin photos are forced into one of the eleven classes."],
    ["Under-representation of dark skin", "H / M", "Only 19 validation lesions in skin-tone groups 0–1."],
    ["Overfitting decisions to the validation set", "M / M", "Threshold and model choices made on the same 1,048 lesions."],
    ["Changes to free-tier services", "M / M", "Hugging Face Docker became paid; Supabase pauses inactive projects."],
    ["Privacy of stored results", "L / H", "The database could expose information if misconfigured."],
], widths=[5.2, 1.3, 9.7], caption="New risks")
c = section(110, "Table 13 gives the updated mitigation plan for the most important open risks.")
c.table([
    ["Risk", "Mitigation", "Contingency"],
    ["Results do not transfer", "Report hidden-test and validation results side by side; external validation on Derm7pt and DDI",
     "Present validation figures as upper bounds"],
    ["Missed melanomas", "Melanoma safety rule; Grad-CAM and possible-diagnosis list shown to physicians", "Review the threshold after external validation"],
    ["Health anxiety / false alarms", "Calm, action-first patient view; absolute-number explanation; details behind an expander",
     "Raise patient-mode thresholds if user feedback shows undue alarm"],
    ["Out-of-scope inputs", "Scope question and warning in patient mode; skin-likeness check", "Train an out-of-scope detector (future work)"],
    ["Dark-skin under-representation", "Transparent reporting; low-data warning for skin tones 0–1", "External test on DDI; Fitzpatrick17k training"],
    ["Validation overfitting", "Pre-registered decision rules; nested cross-validation for tuning", "Use cross-validated estimates only"],
    ["Free-tier changes", "Weights in a private model repository; keep the database active before presentations", "Fall back to demo mode without database"],
    ["Privacy", "No images or personal data stored; consent checkboxes for storage and for the survey; service-role access only; append-only audit log", "Disable storage"],
], widths=[3.8, 7.6, 4.8], caption="Revised risk mitigation plan")

# ------------------------------------------------------------------ 9 challenges
set_text(P[113], "The main technical challenges and their solutions were:")
c = Cursor(P[113])
c.bullets([
    "**Deployment limits.** The free cloud tier offers about 1 GB of memory. CPU builds of PyTorch, loading only the models in use "
    "and resizing uploaded images to 1,600 px kept peak memory below 800 MB.",
    "**Data leakage between datasets.** ISIC 2019 shares images with MILK10k. Identifier matching plus a perceptual hash calibrated "
    "on MILK10k removed 33 near-duplicates before training.",
    "**Comparing models fairly.** Every experiment reused the same split file and validation lesions, and decision rules were fixed "
    "before results were seen.",
    "**Benchmark format.** The official scorer accepted only a lesion_id column, which was not clear from the documentation.",
    "**Interface robustness.** Widget state, language switching, dark theme and mobile layout required several iterations; the app "
    "was tested with automated page tests and on a phone-sized viewport.",
])
set_text(P[115], "The project is being completed in an irregular term, which compressed the calendar. Work was developed in "
                 "parallel on separate branches, and integrating it required care to avoid overwriting newer work. Communication was "
                 "mainly asynchronous (WhatsApp), which was efficient for small decisions but made larger design decisions slower; "
                 "written result summaries after each experiment helped to keep everyone aligned.")
set_text(P[117], "Key lessons so far:")
c = Cursor(P[117])
c.bullets([
    "Validation results must be confirmed on independent data; the hidden test showed that a +0.06 validation gain can disappear.",
    "Tuning on the evaluation set produces convincing but false gains; nested cross-validation is necessary.",
    "With limited data, a simple calibrated ensemble can beat a more complex fusion architecture.",
    "Deciding adoption rules before running an experiment keeps decisions objective.",
    "How results are communicated matters as much as accuracy, especially for non-expert users.",
])

# ------------------------------------------------------------------ 10 budget
c = section(119, "No money has been spent. All mandatory resources remained free, and the optional Colab Pro subscription was not "
                 "needed because Kaggle's free GPU quota was sufficient.")
c.table([
    ["Item", "Proposal estimate (TRY)", "Actual (TRY)"],
    ["Software, datasets, GitHub, Kaggle (free tier)", "0", "0"],
    ["Streamlit Community Cloud, Hugging Face Hub, Supabase (free tiers)", "– (not planned)", "0"],
    ["Google Colab Pro (optional)", "≈ 390 / month", "0 (not needed)"],
    ["Report printing and binding", "≈ 150", "0 (digital submission only)"],
    ["Total", "≈ 0 (≈ 540 maximum)", "0 so far"],
], widths=[8.2, 4.0, 4.0], caption="Budget: proposal versus actual")

# ------------------------------------------------------------------ 11 advisor
set_text(P[121], "Communication with the advisor has mainly taken place through WhatsApp messages rather than formal progress "
                 "meetings: the team shared written status updates and results, and the advisor responded with guidance. An "
                 "in-person meeting was held in November 2026. The main feedback and how it was used:")
c = Cursor(P[121])
c.bullets([
    "**Project framing:** the advisor defined the project as AI-supported early detection of skin cancer and provided MILK10k as "
    "the primary dataset. The system is therefore positioned as a triage and decision-support tool, not a diagnostic device.",
    "**Use of additional data (November 2026):** the advisor approved the use of additional public datasets. The extended models "
    "were adopted, and MILK10k-only results are kept as the primary comparison.",
    "**Decisions taken by the team (November 2026):** (1) the current referral thresholds and the melanoma safety rule are "
    "kept as the final operating point, because missing a cancer is far more costly than an unnecessary referral; (2) no "
    "ethics-committee application is needed because no personal, health or identifiable data is collected, and an "
    "informed-consent statement with an explicit agreement checkbox was added to the usability survey; (3) the final "
    "presentation will last 15–30 minutes and include a live demonstration of the deployed system. The final report is due "
    "on 6 December 2026.",
])
c.para("A log of advisor communication is given in Appendix A.")

# ------------------------------------------------------------------ 12 conclusion
set_text(P[123], "The project has met its main technical objectives: a reproducible pipeline, a multimodal model that exceeds the "
                 "Macro F1 target using MILK10k alone, a deployed decision-support application and a transparent equity analysis. "
                 "The additional datasets made the system clearly better for smartphone photos and on the validation set, but the "
                 "official hidden test shows that the generalisation gain is smaller than the validation figures suggest. Next steps "
                 "and priorities for the remaining period:")
c = Cursor(P[123])
c.bullets([
    "Complete the usability study (at least 8–12 SUS responses) and the reproducibility test.",
    "Agree the final referral thresholds and the melanoma rule with the advisor and clarify ethics requirements.",
    "Run external validation on independent datasets (Derm7pt, DDI), subject to data access.",
    "If time allows, improve the clinical model on diverse skin tones (Fitzpatrick17k) and melanoma detection (ISIC 2020).",
    "Prepare the final report, poster, presentation and a short demonstration video.",
])

# ------------------------------------------------------------------ references
refs = [
    "P. Tschandl et al., “MILK10k: a hierarchical multimodal imaging-learning toolkit for diagnosing pigmented and nonpigmented skin "
    "cancer and its simulators,” J. Invest. Dermatol., vol. 146, no. 2, pp. 357–364.e7, 2026.",
    "MILK study team, “MILK10k,” ISIC Archive, 2025, doi: 10.34970/648456.",
    "MILK study team, “MILK10k Benchmark,” ISIC Challenge, 2025, doi: 10.34970/262082.",
    "A. G. C. Pacheco et al., “PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from "
    "smartphones,” Data in Brief, vol. 32, art. 106221, 2020.",
    "P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of "
    "common pigmented skin lesions,” Sci. Data, vol. 5, art. 180161, 2018.",
    "N. C. F. Codella et al., “Skin lesion analysis toward melanoma detection: a challenge at the 2017 ISBI, hosted by ISIC,” in Proc. "
    "IEEE ISBI, 2018, pp. 168–172.",
    "M. Combalia et al., “BCN20000: dermoscopic lesions in the wild,” arXiv:1908.02288, 2019.",
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
]
c = section(125, f"[1] {refs[0]}")
for i, r in enumerate(refs[1:], start=2):
    c.para(f"[{i}] {r}", size=Pt(10))

# ------------------------------------------------------------------ appendices
c = section(130, "Communication with the advisor took place mainly through WhatsApp. The log below lists the main exchanges; the "
                 "team should complete the dates from the message history before submission.")
c.table([
    ["Date", "Channel", "Topic", "Outcome"],
    ["Start of project", "In person / WhatsApp", "Project topic and dataset", "Topic set as AI-supported early skin-cancer detection; MILK10k provided"],
    ["January 2026", "Submission", "Project proposal", "Proposal submitted"],
    ["[date]", "WhatsApp", "Progress update: models, deployed prototype", "[advisor response]"],
    ["[date]", "WhatsApp", "ISIC benchmark result and validation analysis", "[advisor response]"],
    ["November 2026", "In person", "Use of additional datasets; next steps", "Additional public datasets approved"],
    ["November 2026", "Team (WhatsApp)", "Thresholds, ethics, presentation", "Thresholds kept; consent statement added to survey; 15–30 min presentation with live demo"],
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
    "eval_page.py, stats_page.py, survey_page.py, health_page.py – model analysis, statistics, SUS survey and skin-health pages.",
    "db.py, supabase_ek.sql – anonymous storage (PostgreSQL via Supabase).",
    "kaggle/ – training notebooks (MILK10k baseline, PAD-UFES-20 clinical models, ISIC 2019 dermoscopic models).",
    "analysis/ – evaluation scripts (thresholds, comparisons, ablation, ROC-AUC, threshold tuning, evaluation summary).",
])
c.para("Model weights are kept in a private Hugging Face repository and can be shared with the advisor and examiners on request.")

c = section(136, "Class distribution of the MILK10k training data (5,240 lesions):")
counts = {"BCC": 2522, "NV": 746, "BKL": 544, "SCCKA": 473, "MEL": 450, "AKIEC": 303, "DF": 52, "INF": 50, "VASC": 47,
          "BEN_OTH": 44, "MAL_OTH": 9}
groups = {"BCC": "Malignant", "MEL": "Malignant", "SCCKA": "Malignant", "MAL_OTH": "Malignant", "AKIEC": "Pre-malignant"}
f1 = {"AKIEC": 0.62, "BCC": 0.91, "BEN_OTH": 0.29, "BKL": 0.54, "DF": 0.90, "INF": 0.33, "MAL_OTH": 0.00, "MEL": 0.69, "NV": 0.85,
      "SCCKA": 0.68, "VASC": 0.74}
auc = {"AKIEC": 0.947, "BCC": 0.965, "BEN_OTH": 0.754, "BKL": 0.879, "DF": 0.999, "INF": 0.981, "MAL_OTH": 0.844, "MEL": 0.948,
       "NV": 0.983, "SCCKA": 0.957, "VASC": 0.998}
rows = [["Class", "Group", "Lesions", "Share", "Validation F1", "Validation ROC-AUC"]]
for k, n in counts.items():
    rows.append([k, groups.get(k, "Benign"), f"{n:,}", f"{n / 5240 * 100:.1f}%", f"{f1[k]:.2f}", f"{auc[k]:.3f}"])
c.table(rows, widths=[2.4, 3.0, 2.2, 2.0, 3.0, 3.6], caption="MILK10k class distribution and per-class validation results (deployed physician mode)")

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
