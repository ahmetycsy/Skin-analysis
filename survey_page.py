"""'Kullanılabilirlik anketi' sayfası: Sistem Kullanılabilirlik Ölçeği (SUS, Brooke 1996), 10 madde, 1–5 Likert.

Puan: tek numaralı maddeler (olumlu) için (yanıt − 1), çift numaralılar (olumsuz) için (5 − yanıt); toplam × 2,5 → 0–100.
Yorum (Bangor vd. 2009; Sauro 2011): ~68 ortalama; ≥ 85,5 mükemmel, ≥ 71,4 iyi, ≥ 50,9 kabul edilebilir sınırda, altı zayıf.
Yanıtlar anonimdir: ad, e-posta, IP kaydedilmez. Veritabanı varsa `sus` tablosuna (supabase_ek.sql), yoksa yalnızca oturuma yazılır.
"""
from __future__ import annotations

import pandas as pd
import altair as alt
import streamlit as st

import i18n

TEAL, MUTED = "#0F766E", "#5F747C"

ITEMS = [
    ("Bu sistemi sık sık kullanmak isteyeceğimi düşünüyorum.", "I think that I would like to use this system frequently."),
    ("Sistemi gereksiz yere karmaşık buldum.", "I found the system unnecessarily complex."),
    ("Sistemin kullanımının kolay olduğunu düşündüm.", "I thought the system was easy to use."),
    ("Bu sistemi kullanabilmek için teknik bilgisi olan birinin desteğine ihtiyaç duyacağımı düşünüyorum.",
     "I think that I would need the support of a technical person to be able to use this system."),
    ("Sistemdeki işlevlerin birbiriyle iyi bütünleştirildiğini düşündüm.", "I found the various functions in this system were well integrated."),
    ("Sistemde çok fazla tutarsızlık olduğunu düşündüm.", "I thought there was too much inconsistency in this system."),
    ("Çoğu insanın bu sistemi kullanmayı çok çabuk öğreneceğini düşünüyorum.",
     "I would imagine that most people would learn to use this system very quickly."),
    ("Sistemi kullanmayı çok zahmetli buldum.", "I found the system very cumbersome to use."),
    ("Sistemi kullanırken kendimden çok emin hissettim.", "I felt very confident using the system."),
    ("Bu sistemi kullanmaya başlamadan önce birçok şey öğrenmem gerekti.",
     "I needed to learn a lot of things before I could get going with this system."),
]
ROLES = {"physician": ("Hekim", "Physician"), "student": ("Tıp öğrencisi", "Medical student"),
         "health": ("Diğer sağlık çalışanı", "Other healthcare worker"), "other": ("Diğer", "Other")}


def L(tr: str, en: str) -> str:
    return tr if i18n.get_lang() == "tr" else en


def sus_score(ans: list) -> float:
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(ans))


def grade(score: float) -> str:
    if score >= 85.5:
        return L("Mükemmel", "Excellent")
    if score >= 71.4:
        return L("İyi", "Good")
    if score >= 50.9:
        return L("Kabul edilebilir (sınırda)", "OK (marginal)")
    return L("Zayıf", "Poor")


def _num(x: float, d: int = 1) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if i18n.get_lang() == "tr" else s


def _form(db, version: str, log):
    ss = st.session_state
    lg = i18n.get_lang()
    with st.form(key=f"sus_form_{ss.get('sus_round', 0)}_{lg}", border=True):
        st.markdown(f'<div class="card-title">{L("Anket", "Survey")}</div>', unsafe_allow_html=True)
        st.caption(L("1 = Kesinlikle katılmıyorum · 5 = Kesinlikle katılıyorum. Her maddeyi işaretleyin; çok düşünmeden, "
                     "ilk aklınıza gelen yanıtı verin.",
                     "1 = Strongly disagree · 5 = Strongly agree. Answer every item; go with your first impression."))
        role = st.radio(L("Rolünüz", "Your role"), list(ROLES), index=None, horizontal=True,
                        format_func=lambda k: L(*ROLES[k]))
        answers = []
        for i, (tr, en) in enumerate(ITEMS, 1):
            answers.append(st.radio(f"{i}. {L(tr, en)}", [1, 2, 3, 4, 5], index=None, horizontal=True, key=f"sus_q{i}"))
        comment = st.text_area(L("Eklemek istediğiniz bir şey var mı? (isteğe bağlı)", "Anything to add? (optional)"),
                               max_chars=400, height=80,
                               placeholder=L("Ad, telefon, hasta bilgisi gibi kişisel bilgi yazmayın.",
                                             "Do not write personal information such as names or patient details."))
        sent = st.form_submit_button(L("Gönder", "Submit"), type="primary", icon=":material/send:")
    if not sent:
        return
    if role is None or any(a is None for a in answers):
        empty = [str(i) for i, a in enumerate(answers, 1) if a is None]
        st.warning(L("Lütfen rolünüzü ve tüm maddeleri işaretleyin.", "Please choose your role and answer every item.")
                   + (f" ({L('boş maddeler', 'unanswered')}: {', '.join(empty)})" if empty else ""))
        return
    score = sus_score(answers)
    row = {"role": role, "lang": lg, "app_version": version, "answers": {"items": answers, "comment": comment.strip()},
           "score": score}
    ok = db.add_sus(row) if db is not None else False
    if not ok:
        ss.setdefault("sus_local", []).append(row)
    ss.sus_last = (score, ok, db is not None)
    ss.sus_round = ss.get("sus_round", 0) + 1
    log("sus", None, score=score)
    st.rerun()


def _results(db):
    ss = st.session_state
    rows, err = [], None
    if db is not None:
        rows = db.list_sus()
        err = db.last_error
    rows = rows + [r for r in ss.get("sus_local", [])]
    st.markdown(f'<div class="sec-label">{L("Sonuçlar", "Results")}</div>', unsafe_allow_html=True)
    if err:
        st.caption(L("Anket tablosuna ulaşılamadı; veritabanında `sus` tablosu yoksa supabase_ek.sql çalıştırılmalı. "
                     "Bu oturumdaki yanıtlar gösteriliyor.",
                     "Could not reach the survey table; if the `sus` table is missing, run supabase_ek.sql. "
                     "Showing this session's answers.") + f" ({err[:80]})")
    if not rows:
        st.markdown(f'<div class="empty">{L("Henüz yanıt yok.", "No answers yet.")}</div>', unsafe_allow_html=True)
        return
    scores = pd.Series([float(r["score"]) for r in rows])
    mean = scores.mean()
    cols = st.columns(4)
    for col, (v, lab) in zip(cols, [(str(len(rows)), L("Yanıt", "Responses")), (_num(mean), L("Ortalama SUS", "Mean SUS")),
                                    (_num(scores.median()), L("Ortanca", "Median")), (grade(mean), L("Değerlendirme", "Rating"))]):
        col.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{lab}</div></div>', unsafe_allow_html=True)
    if len(rows) >= 2:
        sd = scores.std()
        half = 1.96 * sd / len(rows) ** 0.5
        st.caption(L(f"%95 güven aralığı ≈ {_num(max(0, mean - half))} – {_num(min(100, mean + half))} "
                     f"(az yanıtla geniştir; kullanılabilirlik çalışmalarında genelde 8–12 katılımcı önerilir).",
                     f"95% CI ≈ {_num(max(0, mean - half))} – {_num(min(100, mean + half))} "
                     f"(wide with few answers; usability studies usually aim for 8–12 participants)."))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div class="sec-label">{L("Puan dağılımı", "Score distribution")}</div>', unsafe_allow_html=True)
        bins = [f"{b}–{b + 10}" for b in range(0, 100, 10)]
        cnt = [int(((scores >= b) & ((scores < b + 10) if b < 90 else (scores <= 100))).sum()) for b in range(0, 100, 10)]
        df = pd.DataFrame({"bin": bins, "n": cnt, "lo": list(range(0, 100, 10))})
        bars = alt.Chart(df).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(
            x=alt.X("bin:N", sort=bins, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("n:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("bin:N", title="SUS"), alt.Tooltip("n:Q", title=L("Yanıt", "Responses"))])
        rule = alt.Chart(pd.DataFrame({"x": ["60–70"]})).mark_rule(color="#B0701A", strokeDash=[4, 3]).encode(x="x:N")
        st.altair_chart((bars + rule).properties(height=200).configure_view(strokeWidth=0), use_container_width=True)
        st.caption(L("Kesikli çizgi: 68 civarı = sektör ortalaması.", "Dashed line: around 68 = industry average."))
    with c2:
        st.markdown(f'<div class="sec-label">{L("Madde ortalamaları (1–5)", "Item means (1–5)")}</div>', unsafe_allow_html=True)
        items = [r["answers"]["items"] if isinstance(r["answers"], dict) else r["answers"] for r in rows]
        means = [sum(a[i] for a in items) / len(items) for i in range(10)]
        di = pd.DataFrame({"q": [f"{i + 1}" for i in range(10)], "m": means,
                           "kind": [L("Olumlu madde (yüksek iyi)", "Positive item (high is good)") if i % 2 == 0 else
                                    L("Olumsuz madde (düşük iyi)", "Negative item (low is good)") for i in range(10)],
                           "text": [L(*ITEMS[i]) for i in range(10)]})
        st.altair_chart(alt.Chart(di).mark_bar(cornerRadiusEnd=3, size=12).encode(
            y=alt.Y("q:N", sort=None, title=None), x=alt.X("m:Q", scale=alt.Scale(domain=[1, 5]), title=None),
            color=alt.Color("kind:N", title=None, scale=alt.Scale(range=[TEAL, "#A63D57"]), legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("text:N", title=L("Madde", "Item")), alt.Tooltip("m:Q", format=".2f", title=L("Ortalama", "Mean"))])
            .properties(height=230).configure_view(strokeWidth=0), use_container_width=True)
    by_role = pd.DataFrame({"role": [L(*ROLES.get(r.get("role"), ("—", "—"))) for r in rows], "score": scores})
    tab = by_role.groupby("role")["score"].agg(["count", "mean"]).reset_index()
    tab.columns = [L("Rol", "Role"), L("Yanıt", "Responses"), L("Ortalama SUS", "Mean SUS")]
    st.dataframe(tab.round(1), hide_index=True, width="stretch")
    out = pd.DataFrame([{"score": r["score"], "role": r.get("role"), "lang": r.get("lang"),
                         **{f"q{i + 1}": a for i, a in enumerate(items[k])}} for k, r in enumerate(rows)])
    st.download_button(L("Sonuçları indir (CSV)", "Download results (CSV)"), data=out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sus_sonuclari.csv", mime="text/csv", icon=":material/download:")


def render_survey(db, version: str, log):
    ss = st.session_state
    lead = L("Sistemi denedikten sonra bu kısa anketi doldurun (yaklaşık 2 dakika). Uluslararası standart Sistem Kullanılabilirlik "
             "Ölçeği (SUS) kullanılır. Yanıtlar <b>anonimdir</b>: ad, e-posta ya da IP adresi kaydedilmez.",
             "After trying the system, please fill in this short survey (about 2 minutes). It uses the standard System Usability "
             "Scale (SUS). Answers are <b>anonymous</b>: no name, e-mail or IP address is stored.")
    st.markdown(f'<div class="lead">{lead}</div>', unsafe_allow_html=True)
    last = ss.get("sus_last")
    if last:
        score, ok, has_db = last
        st.success(L(f"Teşekkürler! Bu yanıtın SUS puanı: {_num(score)} ({grade(score)}).",
                     f"Thank you! SUS score of this answer: {_num(score)} ({grade(score)}).")
                   + ("" if ok else " " + L("Veritabanına yazılamadı; yalnızca bu oturumda tutuluyor.",
                                            "Could not be saved to the database; kept in this session only.") if has_db else
                      " " + L("Veritabanı bağlı değil; yalnızca bu oturumda tutuluyor.", "No database; kept in this session only.")))
        st.button(L("Yeni katılımcı için anketi aç", "Open the survey for a new participant"), icon=":material/person_add:",
                  on_click=lambda: ss.pop("sus_last", None))
    else:
        _form(db, version, log)
    _results(db)
