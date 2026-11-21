"""'İstatistikler' sekmesi: bu oturumdaki analizlerin ve hekim değerlendirmelerinin özeti.

Şimdilik veri yalnızca oturum belleğinden (st.session_state) gelir. Veritabanı eklenince aynı fonksiyon
veritabanından okunan kayıtlarla çağrılabilir: history = [{case, time, meta, top, level, elapsed}, ...],
feedback = {case: {dx, agree, action, note, time}}.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import i18n
from i18n import t
from predictor import GROUPS, MALIGN

LEVELS = ["urgent", "refer", "benign"]
LEVEL_KEYS = {"urgent": "lvl_urgent", "refer": "lvl_refer", "benign": "lvl_benign"}
AGREE_KEYS = {"yes": "fb_yes", "partial": "fb_partial", "no": "fb_no"}
ACTION_KEYS = {"follow": "act_follow", "biopsy": "act_biopsy", "excision": "act_excision", "treat": "act_treat", "none": "act_none"}


def _label(text: str):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


def _kpis(items):
    for col, (val, lab) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="kpi-v">{val}</div><div class="kpi-l">{lab}</div></div>',
                     unsafe_allow_html=True)


def _num(x: float, digits: int = 2) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if i18n.get_lang() == "tr" else s


def _int_axis(max_n: int):
    """Vaka sayıları tam sayı olduğu için eksende yalnızca tam sayılar gösterilir."""
    step = max(1, -(-max_n // 5))
    return alt.Axis(values=list(range(0, max_n + 1, step)), format="d", grid=False)


def cohen_kappa(a: list, b: list):
    """İki değerlendirici arası uyum: (gözlenen uyum − şans eseri beklenen uyum) / (1 − beklenen)."""
    n = len(a)
    if n == 0:
        return None, None
    cats = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return po, (1.0 if pe >= 1 else (po - pe) / (1 - pe))


def kappa_label(k: float) -> str:
    """Landis ve Koch (1977) yorumu."""
    for lim, key in ((0.0, "kap_poor"), (0.20, "kap_slight"), (0.40, "kap_fair"), (0.60, "kap_moderate"), (0.80, "kap_substantial")):
        if k <= lim:
            return t(key)
    return t("kap_perfect")


def render_stats(history: list, feedback: dict, is_mock: bool, group_colors: dict, finals: list | None = None, db_on: bool = False):
    st.markdown(f'<p class="lead">{t("st_lead")}</p>', unsafe_allow_html=True)
    if is_mock:
        st.markdown(f'<div class="notice demo">{t("st_demo")}</div>', unsafe_allow_html=True)
    if not history:
        st.markdown(f'<div class="empty">{t("st_empty")}</div>', unsafe_allow_html=True)
        st.caption(t("st_scope_db" if db_on else "st_scope"))
        return

    with_fb = [h for h in history if h["case"] in feedback]
    known = [h for h in with_fb if feedback[h["case"]]["dx"] != "unknown"]
    matches = sum(1 for h in known if feedback[h["case"]]["dx"] == h["top"][0])
    times = [h["elapsed"] for h in history if "elapsed" in h]
    _kpis([
        (str(len(history)), t("st_total")),
        (f"{_num(sum(times) / len(times))} {'sn' if i18n.get_lang() == 'tr' else 's'}" if times else "–", t("st_time")),
        (str(len(with_fb)), t("st_fb")),
        (f"{matches}/{len(known)}" if known else "–", t("st_match")),
    ])

    max_n = max(1, max(sum(1 for h in history if h["level"] == k) for k in LEVELS),
                max(sum(1 for h in history if h["top"][0] == c) for c in {h["top"][0] for h in history}))
    c1, c2 = st.columns(2, gap="large")
    lvl_col = {"urgent": "#A63D57", "refer": "#B0701A", "benign": "#2F7F78"}
    with c1:
        _label(t("st_levels"))
        df = pd.DataFrame({
            "k": LEVELS,
            "name": [t(LEVEL_KEYS[k]) for k in LEVELS],
            "n": [sum(1 for h in history if h["level"] == k) for k in LEVELS],
        })
        chart = (
            alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=18)
            .encode(
                x=alt.X("n:Q", title=None, axis=_int_axis(max_n)),
                y=alt.Y("name:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=220)),
                color=alt.Color("k:N", legend=None, scale=alt.Scale(domain=LEVELS, range=[lvl_col[k] for k in LEVELS])),
                tooltip=[alt.Tooltip("name:N", title=t("h_level")), alt.Tooltip("n:Q", title=t("st_count"))],
            )
            .properties(height=130).configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        _label(t("st_types"))
        counts = {}
        for h in history:
            counts[h["top"][0]] = counts.get(h["top"][0], 0) + 1
        df2 = pd.DataFrame({
            "name": [i18n.plain(c) for c in counts],
            "grp": [i18n.group(GROUPS[c]) for c in counts],
            "color": [group_colors[GROUPS[c]] for c in counts],
            "n": list(counts.values()),
        }).sort_values("n", ascending=False)
        chart2 = (
            alt.Chart(df2).mark_bar(cornerRadiusEnd=3, size=14)
            .encode(
                x=alt.X("n:Q", title=None, axis=_int_axis(max_n)),
                y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=240)),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("name:N", title=t("h_top")), alt.Tooltip("grp:N", title=t("col_group")),
                         alt.Tooltip("n:Q", title=t("st_count"))],
            )
            .properties(height=max(130, 26 * len(df2))).configure_view(strokeWidth=0)
        )
        st.altair_chart(chart2, use_container_width=True)

    _label(t("st_fb_table"))
    if not with_fb:
        st.caption(t("st_no_fb"))
    else:
        rows = []
        for h in with_fb:
            fb = feedback[h["case"]]
            dx = fb["dx"]
            rows.append({
                t("h_case"): h["case"],
                t("st_model_top"): f"{i18n.plain(h['top'][0])} ({i18n.pct(h['top'][1])})",
                t("fb_dx"): t("fb_dx_unknown") if dx == "unknown" else i18n.plain(dx),
                t("st_match_col"): "–" if dx == "unknown" else ("✓" if dx == h["top"][0] else "✗"),
                t("fb_agree"): t(AGREE_KEYS[fb["agree"]]),
                t("fb_action"): t(ACTION_KEYS[fb["action"]]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(t("st_match_note"))

    _label(t("kap_title"))
    pairs = [(feedback[h["case"]]["dx"], h["top"][0]) for h in known]
    if len(pairs) < 5:
        st.caption(t("kap_need", n=len(pairs)))
    else:
        po, k = cohen_kappa([p for p, _ in pairs], [m for _, m in pairs])
        pog, kg = cohen_kappa([GROUPS[p] for p, _ in pairs], [GROUPS[m] for _, m in pairs])
        _kpis([(str(len(pairs)), t("kap_n")), (f"{_num(k)}", t("kap_type", lab=kappa_label(k))),
               (f"{_num(kg)}", t("kap_group", lab=kappa_label(kg))), (i18n.pct(pog), t("kap_group_agree"))])
        st.caption(t("kap_note"))

    _label(t("st_final_title"))
    finals = finals or []
    if not finals:
        st.caption(t("st_final_none"))
    else:
        top1 = sum(1 for f in finals if f["ranked"][0] == f["final"])
        top3 = sum(1 for f in finals if f["final"] in f["ranked"][:3])
        mal = [f for f in finals if GROUPS[f["final"]] == MALIGN]
        caught = sum(1 for f in mal if f["level"] in ("urgent", "refer"))
        _kpis([
            (str(len(finals)), t("st_closed")),
            (f"{top1}/{len(finals)}", t("st_top1")),
            (f"{top3}/{len(finals)}", t("st_top3")),
            (f"{caught}/{len(mal)}" if mal else "–", t("st_sens")),
        ])
        st.caption(t("st_final_note"))
    st.caption(t("st_scope_db" if db_on else "st_scope"))
