"""Eşik kararı için duyarlılık analizi (rapor için): maliyet oranı c ve hasta karışımı değişince en iyi eşikler.

Model analizi sayfasındaki (eval_page.py) maliyet modelinin aynısı, assets/eval_summary.json ızgarası üzerinde:
  kaçırılan kötü huylu = c, kaçırılan kansere dönüşebilir = c/4, öncelikli yerine rutin sevk edilen kötü huylu = c/20,
  iyi huyluya gereksiz öncelikli sevk = 2, gereksiz rutin sevk = 1.
Lezyon sayıları seçilen hasta karışımına göre yeniden ağırlıklandırılır (doğrulama kümesinin %66'sı kötü huylu).
Melanom güvenlik kuralı ızgarada YOKTUR (eval_summary bu kuralı içermez).

Çıktı: analysis/threshold_sensitivity.json ve raporlar/sekiller/fig_cost_curve.png
Kullanım: python3 analysis/threshold_sensitivity.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "assets", "eval_summary.json")))
G = D["grid"]
N = D["true_group_n"]


def counts(cells):
    return {g: {k: cells[i * 3 + j] for j, k in enumerate(("urgent", "refer", "benign"))}
            for i, g in enumerate(("mal", "pre", "ben"))}


def per1000(cc, mix):
    """1.000 başvuruda beklenen sayılar (karışıma göre)."""
    r = {g: {k: v / N[g] for k, v in cc[g].items()} for g in cc}
    ref = sum(mix[g] * (r[g]["urgent"] + r[g]["refer"]) for g in mix)
    return {"missed_mal": 1000 * mix["mal"] * r["mal"]["benign"],
            "missed_pre": 1000 * mix["pre"] * r["pre"]["benign"],
            "referred": 1000 * ref,
            "priority": 1000 * sum(mix[g] * r[g]["urgent"] for g in mix),
            "benign_referred": 1000 * mix["ben"] * (r["ben"]["urgent"] + r["ben"]["refer"]),
            "mal_sens": 1 - r["mal"]["benign"], "ben_ref_rate": r["ben"]["urgent"] + r["ben"]["refer"]}


def cost(cc, mix, c):
    w = {k: mix[k] / N[k] for k in N}
    return (w["mal"] * (c * cc["mal"]["benign"] + c / 20 * cc["mal"]["refer"]) + w["pre"] * c / 4 * cc["pre"]["benign"]
            + w["ben"] * (2 * cc["ben"]["urgent"] + cc["ben"]["refer"]))


def best(mode, mix, c, unc=1):
    M = D["modes"][mode]["grid"]
    out = None
    for ui, u in enumerate(G):
        for ri, r in enumerate(G):
            cc = counts(M[ui][ri][unc])
            s = cost(cc, mix, c)
            if out is None or s < out[0]:
                out = (s, u, r, cc)
    return out


MIXES = {"clinic_low": {"mal": .05, "pre": .10}, "clinic": {"mal": .08, "pre": .10}, "referral_clinic": {"mal": .15, "pre": .15},
         "validation": {k: N[k] / D["n"] for k in ("mal", "pre")}}
for m in MIXES.values():
    m["ben"] = 1 - m["mal"] - m["pre"]
CS = [10, 20, 50, 100, 200]

res = {"cost_model": "c*missed_mal + c/4*missed_pre + c/20*mal_routine + 2*benign_priority + 1*benign_routine",
       "current": {"U": D["current"]["U"], "R": D["current"]["R"]}, "table": [], "curve": {}}
ui0, ri0 = G.index(D["current"]["U"]), G.index(D["current"]["R"])
for mode in ("h", "p"):
    for mk, mix in MIXES.items():
        cur = counts(D["modes"][mode]["grid"][ui0][ri0][1])
        for c in CS:
            s, u, r, cc = best(mode, mix, c)
            res["table"].append({"mode": mode, "mix": mk, "c": c, "best_U": u, "best_R": r,
                                 "best": per1000(cc, mix), "current": per1000(cur, mix),
                                 "cost_ratio_current_vs_best": cost(cur, mix, c) / s if s else 1.0})
# maliyet eğrisi: hekim modu, poliklinik karışımı, U=0,30 sabit, R değişir
for c in (20, 50, 100):
    pts = []
    for ri, r in enumerate(G):
        cc = counts(D["modes"]["h"]["grid"][ui0][ri][1])
        pts.append({"R": r, "cost": cost(cc, MIXES["clinic"], c) * 1000, **per1000(cc, MIXES["clinic"])})
    res["curve"][str(c)] = pts
json.dump(res, open(os.path.join(HERE, "threshold_sensitivity.json"), "w"), indent=1)

# özet yazdır
for row in res["table"]:
    if row["mode"] == "h":
        b, cu = row["best"], row["current"]
        print(f'{row["mix"]:15s} c={row["c"]:3d}  best U={row["best_U"]:.2f} R={row["best_R"]:.2f} | best: missed {b["missed_mal"]:.1f}/1000 '
              f'ref {b["referred"]:.0f}/1000 | current: missed {cu["missed_mal"]:.1f} ref {cu["referred"]:.0f} | cost x{row["cost_ratio_current_vs_best"]:.2f}')

# şekil
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = {"20": "#8FB8B2", "50": "#0E5E57", "100": "#1C2B31"}
    for c, pts in res["curve"].items():
        xs = [p["R"] * 100 for p in pts]
        ys = [p["cost"] for p in pts]
        mn = min(ys)
        a1.plot(xs, [y / mn for y in ys], color=cols[c], lw=2, label=f"c = {c}")
    a1.axvline(20, color="#A63D57", ls="--", lw=1)
    a1.text(21, 2.6, "chosen 20%", color="#A63D57", fontsize=8)
    a1.set_ylim(0.95, 3)
    a1.set_xlabel("Referral threshold R (malignant + pre-malignant, %)")
    a1.set_ylabel("Expected cost / minimum")
    a1.set_title("Cost curve (clinic mix 8% / 10%, U = 30%)", fontsize=9)
    a1.legend(frameon=False, fontsize=8)
    pts = res["curve"]["50"]
    xs = [p["R"] * 100 for p in pts]
    a2.plot(xs, [p["missed_mal"] for p in pts], color="#A63D57", lw=2, label="missed malignant / 1,000")
    a2.set_xlabel("Referral threshold R (%)")
    a2.set_ylabel("Missed malignant per 1,000", color="#A63D57")
    a3 = a2.twinx()
    a3.plot(xs, [p["benign_referred"] for p in pts], color="#0E5E57", lw=2, label="benign referred / 1,000")
    a3.set_ylabel("Benign referred per 1,000", color="#0E5E57")
    a2.axvline(20, color="#A63D57", ls="--", lw=1)
    a2.set_title("Trade-off per 1,000 patients (clinic mix)", fontsize=9)
    for ax in (a1, a2, a3):
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "raporlar", "sekiller", "fig_cost_curve.png")
    fig.savefig(out, dpi=200)
    print("figure", out)
except ImportError:
    print("matplotlib yok; şekil üretilmedi")
