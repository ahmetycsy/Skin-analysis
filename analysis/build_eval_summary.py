"""Doğrulama tahminlerinden (val_predictions.py çıktısı) sitedeki "Model analizi" sayfası için ÖZET üretir.

Kullanım: python3 analysis/build_eval_summary.py ~/Downloads/sonuclar/val_ensemble_predictions.csv
Çıktı: assets/eval_summary.json — yalnızca toplu sayılar (karışıklık matrisi, eşik ızgarası, alt grup sayıları).
Lezyon bazlı veri, görüntü ya da kimlik İÇERMEZ (MILK10k lisansı: veri herkese açık repoya konmaz).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from predictor import (CLASSES, GROUPS, MALIGN, MEL_SAFETY, PREMALIGN, PRIORITY_TYPE_T, PRIORITY_TYPES,  # noqa: E402
                       REFER_MALIGN_OR_PRE, REFER_URGENT_MALIGN, UNCERTAIN_GAP, UNCERTAIN_TOP)

GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
LEVELS = ["urgent", "refer", "benign"]
TRUE_GROUPS = ["mal", "pre", "ben"]

df = pd.read_csv(sys.argv[1])
y = df.label.values
yi = np.array([CLASSES.index(c) for c in y])
tg = np.array(["mal" if GROUPS[c] == MALIGN else "pre" if GROUPS[c] == PREMALIGN else "ben" for c in y])
MAL_IDX = [CLASSES.index(c) for c in CLASSES if GROUPS[c] == MALIGN]
RISK_IDX = [CLASSES.index(c) for c in CLASSES if GROUPS[c] in (MALIGN, PREMALIGN)]


def levels(P, U, R, use_unc=True):
    mal, risk = P[:, MAL_IDX].sum(1), P[:, RISK_IDX].sum(1)
    s = np.sort(P, 1)
    unc = (s[:, -1] < UNCERTAIN_TOP) | ((s[:, -1] - s[:, -2]) < UNCERTAIN_GAP)
    lv = np.where(mal >= U, 0, np.where((risk >= R) | (unc & use_unc), 1, 2))
    return lv, unc


def type_levels(P, T, R, mel_t, use_unc=True):
    """Sitedeki türe göre kural (melanom güvenlik kuralı dahil): 0 öncelikli, 1 sevk, 2 iyi huylu görünüm."""
    risk = P[:, RISK_IDX].sum(1)
    s = np.sort(P, 1)
    unc = ((s[:, -1] < UNCERTAIN_TOP) | ((s[:, -1] - s[:, -2]) < UNCERTAIN_GAP)) & use_unc
    refer = (risk >= R) | unc | (P[:, CLASSES.index("MEL")] >= mel_t)
    fast = P[:, [CLASSES.index(c) for c in PRIORITY_TYPES]].sum(1)
    return np.where(refer & (fast >= T), 0, np.where(refer, 1, 2))


def level_counts(lv, mask=None):
    m = np.ones(len(lv), bool) if mask is None else mask
    return [int(((tg == g) & (lv == k) & m).sum()) for g in TRUE_GROUPS for k in range(3)]


def cm(pred, mask):
    M = np.zeros((len(CLASSES), len(CLASSES)), int)
    np.add.at(M, (yi[mask], pred[mask]), 1)
    return M.tolist()


def age_band(a):
    if pd.isna(a):
        return "unknown"
    return "<40" if a < 40 else "40-59" if a < 60 else "60-74" if a < 75 else "75+"


SUB = {
    "tone": df.skin_tone.map(lambda v: "0-2" if v <= 2 else str(int(v))).values,
    "sex": df.sex.fillna("unknown").values,
    "age": df.age.map(age_band).values,
}
SUB_ORDER = {"tone": ["0-2", "3", "4", "5"], "sex": ["female", "male"], "age": ["<40", "40-59", "60-74", "75+"]}

out = {"dca_thresholds": [round(x, 2) for x in np.arange(0.01, 0.801, 0.01)], "n": int(len(df)), "classes": CLASSES, "grid": GRID, "levels": LEVELS, "true_groups": TRUE_GROUPS,
       "true_group_n": {g: int((tg == g).sum()) for g in TRUE_GROUPS},
       "current": {"U": REFER_URGENT_MALIGN, "T": PRIORITY_TYPE_T, "R": REFER_MALIGN_OR_PRE, "top": UNCERTAIN_TOP, "gap": UNCERTAIN_GAP},
       "modes": {}}
for mode in ("h", "p"):
    P = df[[f"{mode}_{c}" for c in CLASSES]].values
    pred = P.argmax(1)
    allm = np.ones(len(df), bool)
    lv_cur, unc = levels(P, REFER_URGENT_MALIGN, REFER_MALIGN_OR_PRE)
    grid = [[[level_counts(levels(P, U, R, bool(u))[0]) for u in (0, 1)] for R in GRID] for U in GRID]
    subs = {}
    for dim, vals in SUB.items():
        subs[dim] = []
        for key in SUB_ORDER[dim]:
            m = vals == key
            if m.sum() == 0:
                continue
            subs[dim].append({"key": key, "n": int(m.sum()), "correct": int((pred[m] == yi[m]).sum()),
                              "cm": cm(pred, m), "levels": level_counts(lv_cur, m)})
    correct = pred == yi
    # kalibrasyon: en olası türün olasılığı (güven) ile gerçekten doğru çıkma oranı, 10 eşit aralık
    conf = P.max(1)
    cal = []
    for lo in np.arange(0, 1, 0.1):
        m = (conf >= lo) & ((conf < lo + 0.1) if lo < 0.9 else (conf <= 1))
        cal.append({"lo": round(float(lo), 1), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None, "acc": float(correct[m].mean()) if m.any() else None})
    ece = float(sum(c["n"] * abs(c["conf"] - c["acc"]) for c in cal if c["n"]) / len(df))
    # kötü huylu toplam olasılık ile gerçekten kötü huylu olma oranı (ikili kalibrasyon)
    mal_score = P[:, MAL_IDX].sum(1)
    is_mal = tg == "mal"
    cal_mal = []
    for lo in np.arange(0, 1, 0.1):
        m = (mal_score >= lo) & ((mal_score < lo + 0.1) if lo < 0.9 else (mal_score <= 1))
        cal_mal.append({"lo": round(float(lo), 1), "n": int(m.sum()),
                        "pred": float(mal_score[m].mean()) if m.any() else None, "obs": float(is_mal[m].mean()) if m.any() else None})
    # karar eğrisi (net fayda) için: eşik pt'de "pozitif" denen gerçek pozitif / yanlış pozitif sayıları
    risk_score = P[:, RISK_IDX].sum(1)
    is_risk = tg != "ben"
    pts = [round(x, 2) for x in np.arange(0.01, 0.801, 0.01)]
    dca = {
        "mal": {"npos": int(is_mal.sum()), "nneg": int((~is_mal).sum()),
                "tp": [int(((mal_score >= t) & is_mal).sum()) for t in pts], "fp": [int(((mal_score >= t) & ~is_mal).sum()) for t in pts]},
        "risk": {"npos": int(is_risk.sum()), "nneg": int((~is_risk).sum()),
                 "tp": [int(((risk_score >= t) & is_risk).sum()) for t in pts], "fp": [int(((risk_score >= t) & ~is_risk).sum()) for t in pts]},
    }
    mel_t = MEL_SAFETY["hekim" if mode == "h" else "hasta"]
    type_grid = [[[level_counts(type_levels(P, T, R, mel_t, bool(u))) for u in (0, 1)] for R in GRID] for T in GRID]
    out["modes"][mode] = {
        "type_grid": type_grid, "mel_t": mel_t,
        "calibration": {"bins": cal, "ece": ece, "mal_bins": cal_mal}, "dca": dca,
        "cm": cm(pred, allm), "grid": grid, "subgroups": subs,
        "uncertain": {"n": int(unc.sum()), "acc_unc": float(correct[unc].mean()), "acc_rest": float(correct[~unc].mean())},
    }

path = os.path.join("assets", "eval_summary.json")
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(path, os.path.getsize(path) // 1024, "KB")
