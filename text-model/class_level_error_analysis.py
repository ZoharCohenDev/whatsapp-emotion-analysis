"""class_level_error_analysis.py

Breaks down E2-vs-E3 outcome (fixes/regressions/both-right/both-wrong) by
TRUE CLASS (interested vs losing_interest) at each observation prefix
(25/50/75/100%). Also computes mean predicted-class confidence for fixed
vs broken conversations. Read-only: loads existing prediction CSVs, no
retraining, no model changes.

Source files (per prefix, 459 paired conversations each):
  text-model/outputs/early_detection_e2/predictions_{tag}.csv
  text-model/outputs/early_detection_e3/predictions_{tag}.csv
Column loading/pairing logic reused from text-model/statistical_analysis.py
(load_predictions / verify_predictions / error_analysis).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
E2_DIR = BASE_DIR / "outputs" / "early_detection_e2"
E3_DIR = BASE_DIR / "outputs" / "early_detection_e3"

FRACTIONS = [0.25, 0.50, 0.75, 1.00]

KNOWN_TOTALS = {
    0.25: {"fixes": 44, "regressions": 37},
    0.50: {"fixes": 23, "regressions": 23},
    0.75: {"fixes": 17, "regressions": 7},
    1.00: {"fixes": 1, "regressions": 5},
}

CLASSES = ["interested", "losing_interest"]


def load_predictions(frac: float):
    tag = f"{int(frac * 100):03d}pct"
    e2 = pd.read_csv(E2_DIR / f"predictions_{tag}.csv")
    e3 = pd.read_csv(E3_DIR / f"predictions_{tag}.csv")
    return e2, e3


def verify_paired(e2: pd.DataFrame, e3: pd.DataFrame, tag: str) -> None:
    assert len(e2) == 459 and len(e3) == 459, f"{tag}: expected 459 rows each, got e2={len(e2)} e3={len(e3)}"
    assert not e2["conversation_id"].duplicated().any(), f"{tag}: dup IDs in E2"
    assert not e3["conversation_id"].duplicated().any(), f"{tag}: dup IDs in E3"
    assert (e2["conversation_id"].values == e3["conversation_id"].values).all(), f"{tag}: ID order mismatch"
    assert (e2["actual_label"].values == e3["actual_label"].values).all(), f"{tag}: true label mismatch"


def max_conf(df: pd.DataFrame) -> pd.Series:
    return df[["probability_interested", "probability_losing_interest"]].max(axis=1)


def analyze_prefix(frac: float) -> dict:
    tag = f"{int(frac * 100)}pct"
    e2, e3 = load_predictions(frac)
    verify_paired(e2, e3, tag)

    e2_correct = e2["correct"].astype(int).values
    e3_correct = e3["correct"].astype(int).values
    y_true = e2["actual_label"].values
    cids = e2["conversation_id"].values
    e2_conf = max_conf(e2).values
    e3_conf = max_conf(e3).values

    n = len(e2)
    outcome = []  # per-row outcome label
    for i in range(n):
        if e2_correct[i] and e3_correct[i]:
            outcome.append("both_right")
        elif (not e2_correct[i]) and e3_correct[i]:
            outcome.append("fix")
        elif e2_correct[i] and (not e3_correct[i]):
            outcome.append("regression")
        else:
            outcome.append("both_wrong")

    df = pd.DataFrame({
        "conversation_id": cids,
        "true_label": y_true,
        "outcome": outcome,
        "e2_conf": e2_conf,
        "e3_conf": e3_conf,
    })

    # --- class-level fix/regression/net breakdown ---
    class_breakdown = {}
    total_fixes = 0
    total_regressions = 0
    for cls in CLASSES:
        sub = df[df["true_label"] == cls]
        n_fix = int((sub["outcome"] == "fix").sum())
        n_reg = int((sub["outcome"] == "regression").sum())
        n_both_right = int((sub["outcome"] == "both_right").sum())
        n_both_wrong = int((sub["outcome"] == "both_wrong").sum())
        class_breakdown[cls] = {
            "n_true": int(len(sub)),
            "fixes": n_fix,
            "regressions": n_reg,
            "net": n_fix - n_reg,
            "both_right": n_both_right,
            "both_wrong": n_both_wrong,
        }
        total_fixes += n_fix
        total_regressions += n_reg

    known = KNOWN_TOTALS[frac]
    sanity_ok = (total_fixes == known["fixes"]) and (total_regressions == known["regressions"])

    # --- confidence for fixes vs regressions (overall, using E3's confidence
    #     since E3 is the model whose predicted-class confidence changed the
    #     outcome; also report E2's confidence on those same conversations) ---
    fix_ids = df[df["outcome"] == "fix"]
    reg_ids = df[df["outcome"] == "regression"]
    conf_summary = {
        "fixes": {
            "n": int(len(fix_ids)),
            "mean_e3_conf": round(float(fix_ids["e3_conf"].mean()), 4) if len(fix_ids) else None,
            "mean_e2_conf": round(float(fix_ids["e2_conf"].mean()), 4) if len(fix_ids) else None,
        },
        "regressions": {
            "n": int(len(reg_ids)),
            "mean_e3_conf": round(float(reg_ids["e3_conf"].mean()), 4) if len(reg_ids) else None,
            "mean_e2_conf": round(float(reg_ids["e2_conf"].mean()), 4) if len(reg_ids) else None,
        },
    }
    # per-class confidence too
    conf_by_class = {}
    for cls in CLASSES:
        f = fix_ids[fix_ids["true_label"] == cls]
        r = reg_ids[reg_ids["true_label"] == cls]
        conf_by_class[cls] = {
            "fixes_mean_e3_conf": round(float(f["e3_conf"].mean()), 4) if len(f) else None,
            "fixes_mean_e2_conf": round(float(f["e2_conf"].mean()), 4) if len(f) else None,
            "regressions_mean_e3_conf": round(float(r["e3_conf"].mean()), 4) if len(r) else None,
            "regressions_mean_e2_conf": round(float(r["e2_conf"].mean()), 4) if len(r) else None,
        }

    # --- contingency: both-right / both-wrong by true class ---
    contingency = {}
    for cls in CLASSES:
        sub = df[df["true_label"] == cls]
        contingency[cls] = {
            "both_right": int((sub["outcome"] == "both_right").sum()),
            "both_wrong": int((sub["outcome"] == "both_wrong").sum()),
            "fix": int((sub["outcome"] == "fix").sum()),
            "regression": int((sub["outcome"] == "regression").sum()),
            "n_true": int(len(sub)),
        }

    return {
        "prefix": tag,
        "total_fixes": total_fixes,
        "total_regressions": total_regressions,
        "known_fixes": known["fixes"],
        "known_regressions": known["regressions"],
        "sanity_ok": sanity_ok,
        "class_breakdown": class_breakdown,
        "confidence_summary": conf_summary,
        "confidence_by_class": conf_by_class,
        "contingency_by_class": contingency,
    }


def main():
    results = {}
    for frac in FRACTIONS:
        results[frac] = analyze_prefix(frac)

    print("=" * 90)
    print("CLASS-LEVEL BREAKDOWN: E2 vs E3 outcomes by TRUE CLASS, per prefix")
    print("=" * 90)
    for frac in FRACTIONS:
        r = results[frac]
        print(f"\n--- Prefix {r['prefix']} ---")
        print(f"  Sanity check: computed fixes={r['total_fixes']} (known={r['known_fixes']}), "
              f"regressions={r['total_regressions']} (known={r['known_regressions']}) "
              f"-> {'MATCH' if r['sanity_ok'] else 'MISMATCH !!!'}")
        for cls in CLASSES:
            cb = r["class_breakdown"][cls]
            print(f"  true={cls:<16} n={cb['n_true']:3d}  fixes={cb['fixes']:3d}  "
                  f"regressions={cb['regressions']:3d}  net={cb['net']:+3d}  "
                  f"both_right={cb['both_right']:3d}  both_wrong={cb['both_wrong']:3d}")
        cs = r["confidence_summary"]
        print(f"  Confidence (max predicted-class prob), overall:")
        print(f"    fixes (n={cs['fixes']['n']}): mean E3 conf={cs['fixes']['mean_e3_conf']}  "
              f"mean E2 conf (on same convos)={cs['fixes']['mean_e2_conf']}")
        print(f"    regressions (n={cs['regressions']['n']}): mean E3 conf={cs['regressions']['mean_e3_conf']}  "
              f"mean E2 conf (on same convos)={cs['regressions']['mean_e2_conf']}")
        for cls in CLASSES:
            cc = r["confidence_by_class"][cls]
            print(f"    [{cls}] fixes E3conf={cc['fixes_mean_e3_conf']} E2conf={cc['fixes_mean_e2_conf']} | "
                  f"regressions E3conf={cc['regressions_mean_e3_conf']} E2conf={cc['regressions_mean_e2_conf']}")

    # Save machine-readable summary alongside existing statistical_analysis outputs
    out_path = BASE_DIR / "outputs" / "statistical_analysis" / "class_level_error_analysis.json"
    serializable = {f"{int(f*100)}pct": v for f, v in results.items()}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved machine-readable results to: {out_path}")


if __name__ == "__main__":
    main()
