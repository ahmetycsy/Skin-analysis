"""'Model analizi' sayfası: doğrulama kümesine (1.048 lezyon, modelin eğitimde görmediği) göre
(1) performans panosu, (2) eşik / maliyet simülatörü, (3) eşitlik analizi, (4) poliklinik senaryosu.

Veri: assets/eval_summary.json (analysis/build_eval_summary.py üretir). Yalnızca toplu sayılar vardır;
lezyon bazlı veri yoktur. Sayfa model yüklü olmasa da (demo modunda da) çalışır, çünkü sayılar önceden hesaplanmıştır.
"""
from __future__ import annotations

import json
import math
import os

import altair as alt
import pandas as pd
import streamlit as st

import i18n
from i18n import t
from predictor import BENIGN, GROUPS, MALIGN, PREMALIGN
from stats_page import _kpis, _label, _num

PATH = os.path.join(os.path.dirname(__file__), "assets", "eval_summary.json")
C_URGENT, C_REFER, C_BENIGN = "#A63D57", "#B0701A", "#2F7F78"
TEAL, MUTED = "#0F766E", "#5F747C"
LEVEL_COL = {"urgent": C_URGENT, "refer": C_REFER, "benign": C_BENIGN}


def L(tr: str, en: str) -> str:
    return tr if i18n.get_lang() == "tr" else en


@st.cache_data
def _load():
    with open(PATH) as f:
        return json.load(f)


def _pc(x: float, d: int = 1) -> str:
    return "%" + _num(100 * x, d) if i18n.get_lang() == "tr" else _num(100 * x, d) + "%"


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _note(text: str):
    st.markdown(f'<div class="ev-note">{text}</div>', unsafe_allow_html=True)


def _level_name(k: str) -> str:
    return t({"urgent": "lvl_urgent", "refer": "lvl_refer", "benign": "lvl_benign"}[k])


def _tg_name(g: str) -> str:
    return i18n.group({"mal": MALIGN, "pre": PREMALIGN, "ben": BENIGN}[g])


def _counts(cells):
    """9 sayı → {gerçek grup: {seviye: sayı}}"""
    return {g: {k: cells[i * 3 + j] for j, k in enumerate(("urgent", "refer", "benign"))}
            for i, g in enumerate(("mal", "pre", "ben"))}


def _rates(c):
    mal, pre, ben = c["mal"], c["pre"], c["ben"]
    nm, nr, nb = sum(mal.values()), sum(mal.values()) + sum(pre.values()), sum(ben.values())
    return {
        "mal_caught": (mal["urgent"] + mal["refer"]) / nm if nm else 0,
        "mal_urgent": mal["urgent"] / nm if nm else 0,
        "mal_missed": mal["benign"],
        "risk_caught": (nr - mal["benign"] - pre["benign"]) / nr if nr else 0,
        "pre_missed": pre["benign"],
        "ben_ref": (ben["urgent"] + ben["refer"]) / nb if nb else 0,
        "ben_urgent": ben["urgent"] / nb if nb else 0,
        "n_ref": sum(c[g]["urgent"] + c[g]["refer"] for g in c),
    }


def _per_class(cm, classes):
    n = sum(map(sum, cm))
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i][i]
        sup = sum(cm[i])
        pred = sum(r[i] for r in cm)
        fn, fp = sup - tp, pred - tp
        tn = n - tp - fn - fp
        prec = tp / pred if pred else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"code": c, "sup": sup, "prec": prec, "rec": rec, "spec": tn / (tn + fp) if tn + fp else 0.0, "f1": f1})
    return rows


def _macro_f1(cm, classes, present_only=False):
    rows = _per_class(cm, classes)
    if present_only:
        rows = [r for r in rows if r["sup"] > 0]
    return sum(r["f1"] for r in rows) / len(rows) if rows else 0.0


# ------------------------------------------------------------------ 1. performans
def _tab_performance(D, M, mode):
    cm, classes = M["cm"], D["classes"]
    n = D["n"]
    acc = sum(cm[i][i] for i in range(len(classes))) / n
    rows = _per_class(cm, classes)
    bal = sum(r["rec"] for r in rows) / len(rows)
    g = D["grid"]
    cur = _rates(_counts(M["grid"][g.index(D["current"]["U"])][g.index(D["current"]["R"])][1]))
    _kpis([(_pc(acc), L("Doğruluk", "Accuracy")), (_num(_macro_f1(cm, classes), 3), "Macro F1"),
           (_pc(bal), L("Dengeli doğruluk", "Balanced accuracy")),
           (_pc(cur["mal_caught"]), L("Kötü huylu yakalama", "Malignant caught"))])
    _note(L(f"{n} lezyonluk doğrulama kümesi (modeller bu lezyonları eğitimde görmedi). Tek bölünme olduğu için "
            f"5 katlı çapraz doğrulama sonucundan (Model hakkında sayfası) biraz farklıdır. \"Kötü huylu yakalama\": "
            f"kötü huylu lezyonların şu anki eşiklerle en az \"dermatolog değerlendirmesi\" alan payı.",
            f"Validation set of {n} lesions (not seen in training). A single split, so it differs slightly from the "
            f"5-fold cross-validation result on the About page. \"Malignant caught\": share of malignant lesions that "
            f"get at least \"dermatologist review\" with the current thresholds."))

    _label(L("Karışıklık matrisi", "Confusion matrix"))
    norm = st.toggle(L("Satıra göre yüzde göster", "Show row percentages"), value=True, key=f"ev_norm_{mode}")
    recs = []
    for i, a in enumerate(classes):
        tot = sum(cm[i]) or 1
        for j, b in enumerate(classes):
            v = cm[i][j]
            recs.append({"true": a, "pred": b, "n": v, "p": v / tot, "true_name": i18n.plain(a), "pred_name": i18n.plain(b),
                         "lab": ((f"{round(100 * v / tot)}" if 100 * v / tot >= 0.5 else "<1") if v else "") if norm else (str(v) if v else "")})
    df = pd.DataFrame(recs)
    val = "p:Q" if norm else "n:Q"
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=classes, title=L("Modelin tahmini", "Predicted"), axis=alt.Axis(labelAngle=-45, ticks=False)),
        y=alt.Y("true:N", sort=classes, title=L("Gerçek tür", "True class"), axis=alt.Axis(ticks=False)))
    heat = base.mark_rect(stroke="#fff", strokeWidth=1.5).encode(
        color=alt.condition(alt.datum.n > 0, alt.Color(val, legend=None, scale=alt.Scale(scheme="tealblues", type="sqrt")),
                            alt.value("#1C222B" if st.session_state.get("dark") else "#F4F7F7")),
        tooltip=[alt.Tooltip("true_name:N", title=L("Gerçek", "True")), alt.Tooltip("pred_name:N", title=L("Tahmin", "Predicted")),
                 alt.Tooltip("n:Q", title=L("Lezyon", "Lesions")), alt.Tooltip("p:Q", title=L("Satır payı", "Row share"), format=".1%")])
    txt = base.mark_text(fontSize=10.5).encode(
        text="lab:N", color=alt.condition(alt.datum.p > 0.45, alt.value("#fff"), alt.value("#1F3A44")))
    st.altair_chart((heat + txt).properties(height=430).configure_view(strokeWidth=0), use_container_width=True)
    _note(L("Satırlar gerçek türü, sütunlar modelin en olası tahminini gösterir. Köşegen (sol üstten sağ alta) doğru "
            "tahminlerdir. Satır yüzdesi o türün duyarlılığıdır. En sık karışma: BCC ↔ SCCKA/AKIEC ve NV ↔ BKL/MEL.",
            "Rows are the true class, columns the model's top prediction; the diagonal is correct. A row percentage is "
            "that class's sensitivity. Most common confusions: BCC ↔ SCCKA/AKIEC and NV ↔ BKL/MEL."))

    _label(L("Tür bazında başarı", "Per-class performance"))
    tab = pd.DataFrame([{
        L("Tür", "Class"): f"{i18n.plain(r['code'])} ({r['code']})",
        L("Grup", "Group"): i18n.group(GROUPS[r["code"]]),
        L("Lezyon", "Lesions"): r["sup"],
        L("Duyarlılık", "Sensitivity"): round(100 * r["rec"], 1),
        L("Özgüllük", "Specificity"): round(100 * r["spec"], 1),
        L("Kesinlik", "Precision"): round(100 * r["prec"], 1),
        "F1": round(r["f1"], 3)} for r in sorted(rows, key=lambda r: -r["sup"])])
    pc = {"format": "%.1f%%"}
    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        # koyu temada ilerleme çubuğu temanın kırmızı vurgusuyla çizilir (tuval, CSS'le değişmez) → düz sayı
        L("Duyarlılık", "Sensitivity"): (st.column_config.NumberColumn(**pc) if st.session_state.get("dark") else
                                         st.column_config.ProgressColumn(min_value=0, max_value=100, **pc)),
        L("Özgüllük", "Specificity"): st.column_config.NumberColumn(**pc),
        L("Kesinlik", "Precision"): st.column_config.NumberColumn(**pc),
        "F1": st.column_config.NumberColumn(format="%.3f")})
    small = [r["code"] for r in rows if r["sup"] < 15]
    _note(L(f"Az örnekli türler ({', '.join(small)}: her biri 15'ten az lezyon) için sonuçlar güvenilir değildir; "
            f"birkaç lezyonun doğru ya da yanlış olması oranı çok değiştirir. Macro F1 tüm türlere eşit ağırlık verdiği "
            f"için bu türler genel skoru aşağı çeker.",
            f"Results for rare classes ({', '.join(small)}: fewer than 15 lesions each) are not reliable; a few lesions "
            f"change the rate a lot. Macro F1 weights all classes equally, so these classes pull the score down."))
    u = M["uncertain"]
    _note(L(f"\"Model kararsız\" uyarısı vakaların {_pc(u['n'] / n)}'inde çıkıyor. Bu vakalarda doğruluk {_pc(u['acc_unc'])}, "
            f"diğerlerinde {_pc(u['acc_rest'])}: uyarı gerçekten zor vakaları işaretliyor.",
            f"The \"model uncertain\" warning appears in {_pc(u['n'] / n)} of cases. Accuracy there is {_pc(u['acc_unc'])} "
            f"versus {_pc(u['acc_rest'])} elsewhere: the warning does flag the hard cases."))


# ------------------------------------------------------------------ 2. simülatör
def _thr_inputs(D, mode):
    g, cur = D["grid"], D["current"]
    ss = st.session_state
    c1, c2, c3 = st.columns([1, 1, 0.8], vertical_alignment="bottom")
    with c1:
        U = st.select_slider(L("Öncelikli eşiği (kötü huylu toplam ≥)", "Priority threshold (malignant total ≥)"),
                             options=g, format_func=_pc0, key=_wk("ev_U", cur["U"], mode))
    with c2:
        R = st.select_slider(L("Sevk eşiği (kötü huylu + kansere dönüşebilir ≥)", "Referral threshold (malignant + premalignant ≥)"),
                             options=g, format_func=_pc0, key=_wk("ev_R", cur["R"], mode))
    with c3:
        unc = st.toggle(L("Kararsız vakaları da sevk et", "Also refer uncertain cases"), key=_wk("ev_unc", True, mode))
    ss.ev_U, ss.ev_R, ss.ev_unc = U, R, unc
    return U, R, unc


def _wk(store: str, default, suffix: str = "") -> str:
    """Kalıcı widget anahtarı: değer `store` altında saklanır, dil/mod değişince yeni widget aynı değerle başlar.
    (value/default parametresini her çalıştırmada değiştirmek widget kimliğini değiştirip tıklamayı kaybettiriyordu.)"""
    ss = st.session_state
    wk = f"w_{store}_{suffix}_{i18n.get_lang()}"
    if wk not in ss:
        ss[wk] = ss.get(store, default)
    return wk


def _pc0(x):
    return _pc(x, 0)


def _tab_simulator(D, M, mode):
    g, cur = D["grid"], D["current"]
    _note(L("Eşikleri değiştirip doğrulama kümesinde sonucun nasıl değiştiğini görün. Kural sitedeki ile aynıdır: "
            "kötü huylu toplam olasılık öncelikli eşiğini geçerse <b>öncelikli değerlendirme</b>, kötü huylu + kansere "
            "dönüşebilir toplamı sevk eşiğini geçerse (ya da model kararsızsa) <b>dermatolog değerlendirmesi</b>, "
            "aksi hâlde <b>iyi huylu görünüm</b>. Burada seçilen eşikler yalnızca bu sayfadaki hesaplarda kullanılır, "
            "analiz ekranını değiştirmez.",
            "Change the thresholds and see how the validation results change. The rule is the same as the app's: malignant "
            "total above the priority threshold → <b>priority review</b>; malignant + premalignant total above the referral "
            "threshold (or model uncertain) → <b>dermatologist review</b>; otherwise <b>benign appearance</b>. Thresholds "
            "chosen here only affect this page, not the analysis screen."))
    U, R, unc = _thr_inputs(D, mode)
    c = _counts(M["grid"][g.index(U)][g.index(R)][int(unc)])
    r = _rates(c)
    r0 = _rates(_counts(M["grid"][g.index(cur["U"])][g.index(cur["R"])][1]))
    nm, nb = D["true_group_n"]["mal"], D["true_group_n"]["ben"]

    def delta(a, b, pct=True, good_up=True):
        d = a - b
        if abs(d) < 1e-9:
            return ""
        s = ("+" if d > 0 else "−") + (_num(abs(100 * d), 1) + L(" puan", " pts") if pct else str(abs(int(d))))
        col = "#2F7F78" if (d > 0) == good_up else "#A63D57"
        return f'<div class="ev-d" style="color:{col}">{s} {L("şu ankine göre", "vs current")}</div>'

    items = [
        (_pc(r["mal_caught"]), L("Kötü huylu: en az değerlendirme alan", "Malignant: at least review"), delta(r["mal_caught"], r0["mal_caught"])),
        (f'{r["mal_missed"]} / {nm}', L("Kaçırılan kötü huylu", "Missed malignant"), delta(r["mal_missed"], r0["mal_missed"], False, False)),
        (_pc(r["mal_urgent"]), L("Kötü huylu: öncelikli", "Malignant: priority"), delta(r["mal_urgent"], r0["mal_urgent"])),
        (_pc(r["ben_ref"]), L("İyi huylu ama sevk edilen", "Benign but referred"), delta(r["ben_ref"], r0["ben_ref"], True, False)),
    ]
    for col, (v, lab, d) in zip(st.columns(4), items):
        col.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{lab}</div>{d}</div>', unsafe_allow_html=True)

    _label(L("Gerçek gruba göre yönlendirme", "Routing by true group"))
    recs = []
    for tg in ("mal", "pre", "ben"):
        tot = sum(c[tg].values()) or 1
        for k in ("urgent", "refer", "benign"):
            recs.append({"grp": f"{_tg_name(tg)} (n={sum(c[tg].values())})", "k": k, "name": _level_name(k), "n": c[tg][k],
                         "p": c[tg][k] / tot, "o": ("urgent", "refer", "benign").index(k)})
    df = pd.DataFrame(recs)
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            x=alt.X("p:Q", stack="normalize", title=None, axis=alt.Axis(format="%", grid=False)),
            y=alt.Y("grp:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=260)),
            color=alt.Color("name:N", title=None, sort=[_level_name(k) for k in LEVEL_COL],
                            scale=alt.Scale(domain=[_level_name(k) for k in LEVEL_COL], range=list(LEVEL_COL.values())),
                            legend=alt.Legend(orient="bottom")),
            order="o:Q",
            tooltip=[alt.Tooltip("grp:N", title=L("Gerçek grup", "True group")), alt.Tooltip("name:N", title=L("Yönlendirme", "Routing")),
                     alt.Tooltip("n:Q", title=L("Lezyon", "Lesions")), alt.Tooltip("p:Q", title=L("Pay", "Share"), format=".1%")])
        .properties(height=alt.Step(36)).configure_scale(bandPaddingInner=0.3).configure_view(strokeWidth=0),
        use_container_width=True)

    _label(L("Denge eğrisi: kaçırılan kanser ↔ gereksiz sevk", "Trade-off: missed cancer ↔ unnecessary referral"))
    pts = []
    for Rv in g:
        rr = _rates(_counts(M["grid"][g.index(U)][g.index(Rv)][int(unc)]))
        pts.append({"R": Rv, "Rl": _pc0(Rv), "missed": rr["mal_missed"], "benref": rr["ben_ref"], "sel": Rv == R})
    dp = pd.DataFrame(pts)
    line = alt.Chart(dp).mark_line(color=TEAL, point=alt.OverlayMarkDef(color=TEAL, size=40)).encode(
        x=alt.X("benref:Q", title=L("İyi huylu ama sevk edilen", "Benign but referred"), axis=alt.Axis(format="%")),
        y=alt.Y("missed:Q", title=L("Kaçırılan kötü huylu (lezyon)", "Missed malignant (lesions)")),
        tooltip=[alt.Tooltip("Rl:N", title=L("Sevk eşiği", "Referral threshold")), alt.Tooltip("missed:Q", title=L("Kaçırılan", "Missed")),
                 alt.Tooltip("benref:Q", title=L("İyi huylu sevk", "Benign referred"), format=".1%")])
    sel = alt.Chart(dp[dp.sel]).mark_point(size=220, color=C_URGENT, filled=True)
    lab = alt.Chart(dp[dp.sel]).mark_text(dx=12, dy=-12, align="left", color=C_URGENT, fontWeight="bold").encode(
        x="benref:Q", y="missed:Q", text="Rl:N")
    st.altair_chart((line + sel.encode(x="benref:Q", y="missed:Q") + lab).properties(height=280).configure_view(strokeWidth=0),
                    use_container_width=True)
    _note(L("Her nokta farklı bir sevk eşiğidir (öncelikli eşiği sabit). Sola gidildikçe daha az gereksiz sevk ama "
            "daha çok kaçırılan kanser olur. Kırmızı nokta seçili eşik.",
            "Each point is a referral threshold (priority threshold fixed). Moving left means fewer unnecessary referrals "
            "but more missed cancers. The red point is the selected threshold."))

    _label(L("Maliyete göre en iyi eşik", "Best thresholds by cost"))
    ca, cb = st.columns([1.2, 1], vertical_alignment="center")
    with ca:
        cost = st.slider(L("Bir kanseri kaçırmak, bir gereksiz sevkten kaç kat kötü?",
                           "How many times worse is missing a cancer than one unnecessary referral?"),
                         1, 100, key=_wk("ev_cost", 50, mode))
        st.session_state.ev_cost = cost
    with ca:
        mixes = {"clinic": L("Poliklinik varsayımı", "Clinic assumption"), "val": L("Doğrulama kümesi", "Validation set")}
        mk = st.segmented_control(L("Hasta karışımı", "Patient mix"), list(mixes), format_func=mixes.get,
                                  key=_wk("ev_mix", "clinic", mode)) or "clinic"
        st.session_state.ev_mix = mk
    N = D["true_group_n"]
    if mk == "clinic":
        pm, pp = st.session_state.get("ev_pm", 8.0) / 100, st.session_state.get("ev_pp", 10.0) / 100
        mix = {"mal": pm, "pre": pp, "ben": max(0.0, 1 - pm - pp)}
    else:
        mix = {k: N[k] / D["n"] for k in N}
    w = {k: mix[k] / N[k] for k in N}  # lezyon sayısını hasta karışımına göre yeniden ağırlıklandırır
    best = None
    for Ui, Uv in enumerate(g):
        for Ri, Rv in enumerate(g):
            cc = _counts(M["grid"][Ui][Ri][int(unc)])
            # kaçırılan kötü huylu: c; kaçırılan kansere dönüşebilir: c/4; öncelikli yerine rutin kötü huylu: c/20;
            # iyi huyluya gereksiz öncelikli sevk: 2 (öncelikli randevu daha pahalı); gereksiz rutin sevk: 1
            score = (w["mal"] * (cost * cc["mal"]["benign"] + cost / 20 * cc["mal"]["refer"]) + w["pre"] * cost / 4 * cc["pre"]["benign"]
                     + w["ben"] * (2 * cc["ben"]["urgent"] + cc["ben"]["refer"]))
            if best is None or score < best[0]:
                best = (score, Uv, Rv, cc)
    br = _rates(best[3])
    with cb:
        st.markdown(
            f'<div class="ev-best"><div class="ev-best-t">{L("Önerilen", "Suggested")}: '
            f'{L("öncelikli", "priority")} {_pc0(best[1])} · {L("sevk", "referral")} {_pc0(best[2])}</div>'
            f'{L("Kaçırılan kötü huylu", "Missed malignant")}: <b>{br["mal_missed"]}</b> · '
            f'{L("iyi huylu sevk", "benign referred")}: <b>{_pc(br["ben_ref"])}</b> · '
            f'{L("kötü huylu öncelikli", "malignant priority")}: <b>{_pc(br["mal_urgent"])}</b>'
            f'<div class="ev-d" style="color:#5F747C">{L("Hasta karışımı", "Patient mix")}: {L("kötü huylu", "malignant")} {_pc0(mix["mal"])}, '
            f'{L("kansere dönüşebilir", "premalignant")} {_pc0(mix["pre"])}</div></div>', unsafe_allow_html=True)
    _note(L("Maliyet = c × kaçırılan kötü huylu + c/4 × kaçırılan kansere dönüşebilir + c/20 × öncelikli yerine rutin sevk edilen "
            "kötü huylu + 2 × iyi huyluya gereksiz öncelikli sevk + 1 × iyi huyluya gereksiz rutin sevk. c değeri klinik bir karardır; ekip ve danışmanla belirlenmelidir. "
            "Doğrulama kümesinin %66'sı kötü huylu olduğu için oranlar seçilen hasta karışımına göre yeniden ağırlıklandırılır; "
            "poliklinik varsayımı Poliklinik senaryosu sekmesinden değiştirilebilir. "
            "Eşikler aynı veride seçildiğinden sonuçlar biraz iyimserdir.",
            "Cost = c × missed malignant + c/4 × missed premalignant + c/20 × malignant routed as routine instead of priority + "
            "2 × unnecessary priority referral of a benign lesion + 1 × unnecessary routine referral. c is a clinical decision for the team and advisor. 66% of the validation set is malignant, "
            "so the best thresholds in a real clinic may differ (see the clinic scenario tab). Thresholds are chosen on the "
            "same data, so results are slightly optimistic."))


# ------------------------------------------------------------------ 3. eşitlik
def _tab_fairness(D, M, mode):
    dims = {"tone": L("Ten rengi (Fitzpatrick benzeri, 0 açık – 5 koyu)", "Skin tone (0 light – 5 dark)"),
            "sex": L("Cinsiyet", "Sex"), "age": L("Yaş grubu", "Age group")}
    dim = st.segmented_control(L("Karşılaştır", "Compare by"), list(dims), format_func=lambda k: dims[k].split(" (")[0],
                               default="tone", key=f"evDim_{mode}_{i18n.get_lang()}") or "tone"

    def name(k):
        if dim == "tone":
            return L("Ten 0–2", "Tone 0–2") if k == "0-2" else L(f"Ten {k}", f"Tone {k}")
        if dim == "sex":
            return i18n.sex(k)
        return L(f"{k} yaş", f"age {k}")
    rows, chart = [], []
    for s in M["subgroups"][dim]:
        c = _counts(s["levels"])
        nm = sum(c["mal"].values())
        nb = sum(c["ben"].values())
        mc = c["mal"]["urgent"] + c["mal"]["refer"]
        br = c["ben"]["urgent"] + c["ben"]["refer"]
        lo, hi = _wilson(s["correct"], s["n"])
        rows.append({L("Grup", "Group"): name(s["key"]), L("Lezyon", "Lesions"): s["n"],
                     L("Doğruluk", "Accuracy"): f'{_pc(s["correct"] / s["n"])}  ({_pc0(lo)}–{_pc0(hi)})',
                     "Macro F1": round(_macro_f1(s["cm"], D["classes"], present_only=True), 3),
                     L("Kötü huylu yakalama", "Malignant caught"): f"{_pc(mc / nm)} ({mc}/{nm})" if nm else "–",
                     L("İyi huylu sevk", "Benign referred"): f"{_pc(br / nb)} ({br}/{nb})" if nb else "–"})
        for metric, k, n in ((L("Doğruluk", "Accuracy"), s["correct"], s["n"]), (L("Kötü huylu yakalama", "Malignant caught"), mc, nm)):
            if n:
                a, b = _wilson(k, n)
                chart.append({"grp": name(s["key"]), "m": metric, "p": k / n, "lo": a, "hi": b, "n": n})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    dc = pd.DataFrame(chart)
    order = [name(s["key"]) for s in M["subgroups"][dim]]
    base = alt.Chart(dc).encode(y=alt.Y("grp:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False)))
    bars = base.mark_rule(strokeWidth=2.5, color="#9DB5B2").encode(
        x=alt.X("lo:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")), x2="hi:Q")
    pts = base.mark_point(filled=True, size=90, color=TEAL).encode(
        x="p:Q", tooltip=[alt.Tooltip("grp:N", title=L("Grup", "Group")), alt.Tooltip("p:Q", format=".1%", title=L("Oran", "Rate")),
                          alt.Tooltip("lo:Q", format=".1%", title=L("Alt sınır", "Lower")),
                          alt.Tooltip("hi:Q", format=".1%", title=L("Üst sınır", "Upper")), alt.Tooltip("n:Q", title="n")])
    st.altair_chart((bars + pts).properties(height=34 * len(order) + 20).facet(
        column=alt.Column("m:N", title=None, sort=[L("Doğruluk", "Accuracy"), L("Kötü huylu yakalama", "Malignant caught")],
                          header=alt.Header(labelFontWeight="bold")))
        .configure_view(strokeWidth=0), use_container_width=True)

    accs = [(s["correct"] / s["n"], name(s["key"])) for s in M["subgroups"][dim]]
    hi_, lo_ = max(accs), min(accs)
    overlap = all(_wilson(s["correct"], s["n"])[0] <= max(a for a, _ in accs) and
                  _wilson(s["correct"], s["n"])[1] >= min(a for a, _ in accs) for s in M["subgroups"][dim])
    _note(L(f"Doğrulukta en yüksek grup {hi_[1]} ({_pc(hi_[0])}), en düşük {lo_[1]} ({_pc(lo_[0])}); fark "
            f"{_num(100 * (hi_[0] - lo_[0]), 1)} puan. Çizgiler %95 güven aralığıdır (Wilson). "
            + ("Aralıklar örtüştüğü için bu fark istatistiksel olarak anlamlı görünmüyor. " if overlap else
               "Bazı aralıklar örtüşmüyor; bu grup farkı dikkatle incelenmeli. ")
            + ("Ten rengi 0–1 olan çok az lezyon olduğu için 0, 1 ve 2 birleştirildi. Grupların tür dağılımı farklıdır "
               "(ör. koyu tenlerde farklı lezyon türleri), bu yüzden fark yalnızca modelden kaynaklanmayabilir."
               if dim == "tone" else "Grupların tür dağılımı farklı olduğu için fark yalnızca modelden kaynaklanmayabilir."),
            f"Highest accuracy: {hi_[1]} ({_pc(hi_[0])}), lowest: {lo_[1]} ({_pc(lo_[0])}); gap "
            f"{_num(100 * (hi_[0] - lo_[0]), 1)} pts. Lines are 95% confidence intervals (Wilson). "
            + ("The intervals overlap, so the gap does not look statistically significant. " if overlap else
               "Some intervals do not overlap; this gap deserves attention. ")
            + ("Tones 0–1 have very few lesions, so 0, 1 and 2 are merged. Groups differ in class mix, so a gap is not "
               "necessarily due to the model alone." if dim == "tone" else
               "Groups differ in class mix, so a gap is not necessarily due to the model alone.")))
    _note(L("Kötü huylu yakalama ve iyi huylu sevk şu anki site eşikleriyle hesaplanır. Macro F1 yalnızca o grupta bulunan "
            "türler üzerinden alınır ve küçük gruplarda dalgalıdır.",
            "Malignant caught and benign referred use the app's current thresholds. Macro F1 is over the classes present in "
            "each group and is noisy in small groups."))


# ------------------------------------------------------------------ 4. poliklinik
def _tab_clinic(D, M, mode):
    ss = st.session_state
    g, cur = D["grid"], D["current"]
    U, R, unc = ss.get("ev_U", cur["U"]), ss.get("ev_R", cur["R"]), ss.get("ev_unc", True)
    _note(L(f"Bir aile hekimliği / dermatoloji polikliniğinde sistemin iş yüküne etkisini tahmin eder. Gerçek oranlar "
            f"doğrulama kümesinden, hasta karışımı ise sizin varsayımınızdan gelir. Eşikler: öncelikli {_pc0(U)}, sevk "
            f"{_pc0(R)} (Eşik simülatörü sekmesinden değiştirilebilir).",
            f"Estimates the system's effect on workload in a primary-care / dermatology clinic. Per-group rates come from "
            f"the validation set; the patient mix is your assumption. Thresholds: priority {_pc0(U)}, referral {_pc0(R)} "
            f"(change them in the threshold simulator tab)."))
    lg = i18n.get_lang()
    c1, c2, c3 = st.columns(3)
    daily = c1.number_input(L("Günlük lezyon başvurusu", "Lesion visits per day"), 5, 2000, step=5, key=_wk("ev_daily", 60))
    pm = c2.number_input(L("Kötü huylu oranı (%)", "Malignant share (%)"), 0.5, 80.0, step=0.5, key=_wk("ev_pm", 8.0))
    pp = c3.number_input(L("Kansere dönüşebilir oranı (%)", "Premalignant share (%)"), 0.0, 60.0, step=0.5, key=_wk("ev_pp", 10.0))
    c4, c5, c6 = st.columns(3)
    cap = c4.number_input(L("Dermatolog kapasitesi (hasta/gün)", "Dermatologist capacity (patients/day)"), 1, 2000,
                          step=1, key=_wk("ev_cap", 40))
    days = c5.number_input(L("Simülasyon süresi (iş günü)", "Simulation length (working days)"), 5, 120, step=1, key=_wk("ev_days", 22))
    minutes = c6.number_input(L("Muayene süresi (dk/hasta)", "Visit length (min/patient)"), 5, 60, step=1, key=_wk("ev_min", 15))
    ss.ev_daily, ss.ev_pm, ss.ev_pp, ss.ev_cap, ss.ev_days, ss.ev_min = daily, pm, pp, cap, days, minutes
    if pm + pp >= 100:
        st.warning(L("Kötü huylu + kansere dönüşebilir oranı %100'den küçük olmalı.", "Malignant + premalignant must be below 100%."))
        return
    mix = {"mal": pm / 100, "pre": pp / 100, "ben": 1 - (pm + pp) / 100}
    c = _counts(M["grid"][g.index(U)][g.index(R)][int(unc)])
    rate = {tg: {k: c[tg][k] / max(1, sum(c[tg].values())) for k in ("urgent", "refer", "benign")} for tg in c}
    per_day = {k: daily * sum(mix[tg] * rate[tg][k] for tg in mix) for k in ("urgent", "refer", "benign")}
    missed_day = daily * mix["mal"] * rate["mal"]["benign"]
    caught_day = daily * mix["mal"] * (rate["mal"]["urgent"] + rate["mal"]["refer"])
    ref_day = per_day["urgent"] + per_day["refer"]

    _label(L("Günlük beklenen dağılım", "Expected per day"))
    _kpis([(_num(per_day["urgent"], 1), _level_name("urgent")), (_num(per_day["refer"], 1), _level_name("refer")),
           (_num(per_day["benign"], 1), _level_name("benign")),
           (_num(missed_day * days, 1), L(f"{days} günde kaçırılan kötü huylu", f"Missed malignant in {days} days"))])

    # öncelikli kuyruk: kapasite önce öncelikli vakalara, kalanı rutin sevklere (akışkan, deterministik model)
    def simulate(urg, rut):
        bu = br_ = 0.0
        out = []
        for d in range(1, days + 1):
            bu += urg
            br_ += rut
            s_u = min(bu, cap)
            bu -= s_u
            s_r = min(br_, cap - s_u)
            br_ -= s_r
            out.append((d, bu, br_))
        return out
    sys_ = simulate(per_day["urgent"], per_day["refer"])
    allr = simulate(0, daily)
    util = ref_day / cap
    # kuyruğu eritme süresi: biriken hastaların tamamı kapasiteyle kaç günde görülür (yeni gelişler hariç)
    wait_r = (sys_[-1][1] + sys_[-1][2]) / cap
    wait_all = allr[-1][2] / cap
    nnr = ref_day / caught_day if caught_day else float("inf")

    _label(L("Dermatolog iş yükü", "Dermatologist workload"))
    _kpis([(_pc(util, 0), L("Kapasite kullanımı (sistemle)", "Capacity use (with system)")),
           (_pc(daily / cap, 0), L("Herkes sevk edilseydi", "If everyone were referred")),
           (_num(ref_day * minutes / 60, 1) + L(" sa", " h"), L("Günlük muayene süresi", "Daily visit time")),
           (_num(nnr, 1) if nnr != float("inf") else "–", L("Yakalanan 1 kanser için sevk", "Referrals per cancer caught"))])
    recs = []
    for (d, bu, brr), (_, _, ba) in zip(sys_, allr):
        recs += [{"d": d, "q": brr + bu, "s": L("Sistemle: bekleyen sevk", "With system: waiting")},
                 {"d": d, "q": ba, "s": L("Herkes sevk: bekleyen", "Refer everyone: waiting")}]
    dq = pd.DataFrame(recs)
    st.altair_chart(alt.Chart(dq).mark_line(strokeWidth=2.5).encode(
        x=alt.X("d:Q", title=L("İş günü", "Working day"), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("q:Q", title=L("Bekleyen hasta", "Patients waiting")),
        color=alt.Color("s:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=[L("Sistemle: bekleyen sevk", "With system: waiting"),
                                                L("Herkes sevk: bekleyen", "Refer everyone: waiting")], range=[TEAL, "#B0701A"])),
        tooltip=[alt.Tooltip("d:Q", title=L("Gün", "Day")), alt.Tooltip("q:Q", format=".0f", title=L("Bekleyen", "Waiting"))])
        .properties(height=240).configure_view(strokeWidth=0), use_container_width=True)
    if util <= 1:
        msg = L(f"Sistemle günde yaklaşık {_num(ref_day, 1)} hasta dermatoloğa gider; kapasite ({cap}) yetiyor, kuyruk birikmez. "
                f"Herkes sevk edilseydi günde {daily} hasta gerekirdi" + (f" ve {days} günün sonunda yaklaşık {_num(allr[-1][2], 0)} hasta "
                f"beklerdi (rutin bekleme ≈ {_num(wait_all, 1)} gün)." if daily > cap else "."),
                f"With the system about {_num(ref_day, 1)} patients a day go to the dermatologist; capacity ({cap}) is enough and no "
                f"queue builds up. Referring everyone would need {daily} a day" + (f" and about {_num(allr[-1][2], 0)} patients would be "
                f"waiting after {days} days (routine wait ≈ {_num(wait_all, 1)} days)." if daily > cap else "."))
    else:
        urg_ok = per_day["urgent"] <= cap
        msg = L(f"Sevk sayısı kapasiteyi aşıyor (%{round(100 * util)}): {days} günün sonunda yaklaşık "
                f"{_num(sys_[-1][1] + sys_[-1][2], 0)} hasta bekler; bu kuyruğu eritmek ≈ {_num(wait_r, 1)} iş günü sürer. "
                + ("Kapasite önce öncelikli vakalara ayrıldığı için öncelikli vakalar beklemez, rutin sevkler birikir. " if urg_ok else
                   "Öncelikli vakalar bile kapasiteyi aşıyor; öncelikli eşiği çok düşük ya da kapasite çok az. ")
                + "Kapasite artırılmalı ya da eşik yükseltilmeli (daha az sevk, daha çok risk).",
                f"Referrals exceed capacity ({round(100 * util)}%): after {days} days about "
                f"{_num(sys_[-1][1] + sys_[-1][2], 0)} patients wait; clearing this queue takes ≈ {_num(wait_r, 1)} working days. "
                + ("Capacity goes to priority cases first, so they do not wait and routine referrals pile up. " if urg_ok else
                   "Even priority cases exceed capacity; the priority threshold is too low or capacity too small. ")
                + "Add capacity or raise the threshold (fewer referrals, more risk).")
    _note(msg)
    _note(L("Basit akışkan kuyruk modeli: her gün beklenen sayıda hasta gelir, kapasite önce öncelikli vakalara ayrılır; "
            "rastgele dalgalanma, randevuya gelmeme ve tekrar başvuru dahil değildir. Hasta karışımı varsayımdır; gerçek "
            "poliklinik verisiyle güncellenmelidir. Doğrulama kümesindeki oranlar MILK10k'nin (çoğunluğu biyopsi yapılmış "
            "lezyonlar) özelliklerini taşır.",
            "Simple fluid queue model: the expected number of patients arrives each day and capacity goes to priority cases "
            "first; random variation, no-shows and repeat visits are not included. The patient mix is an assumption and should "
            "be updated with real clinic data. Validation rates reflect MILK10k (mostly biopsied lesions)."))


# ------------------------------------------------------------------ sayfa
def render_eval():
    if not os.path.exists(PATH):
        st.info(L("Analiz özeti bulunamadı (assets/eval_summary.json).", "Analysis summary not found (assets/eval_summary.json)."))
        return
    D = _load()
    st.markdown("""<style>
    .ev-note { color:#5F747C; font-size:.84rem; line-height:1.5; margin:.35rem 0 .9rem; }
    .ev-d { font-size:.72rem; margin-top:.2rem; }
    .ev-best { background:#EAF4F2; border:1px solid #CFE4E0; border-radius:10px; padding:.7rem .9rem; font-size:.86rem; color:#1F3A44; }
    .ev-best-t { font-weight:600; color:#0F766E; margin-bottom:.25rem; }
    </style>""", unsafe_allow_html=True)
    modes = {"h": t("mode_hekim"), "p": t("mode_hasta")}
    mode = st.segmented_control(L("Analiz modu", "Analysis mode"), list(modes), format_func=modes.get,
                                key=_wk("ev_mode", "h")) or "h"
    st.session_state.ev_mode = mode
    M = D["modes"][mode]
    tabs = st.tabs([L("Performans", "Performance"), L("Eşik simülatörü", "Threshold simulator"),
                    L("Eşitlik analizi", "Fairness"), L("Poliklinik senaryosu", "Clinic scenario")])
    with tabs[0]:
        _tab_performance(D, M, mode)
    with tabs[1]:
        _tab_simulator(D, M, mode)
    with tabs[2]:
        _tab_fairness(D, M, mode)
    with tabs[3]:
        _tab_clinic(D, M, mode)
