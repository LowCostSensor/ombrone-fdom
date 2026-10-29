"""Fluorescence indices for the Ombrone dataset, plus the comparisons behind the settings.

Usage:
    python scripts/indices.py --dataset outputs/ombrone_raw.nc \
        --config config/processing.toml --out outputs/indices
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from correct import correct_until_raman, load_config

from eemkit import corrections as C
from eemkit import indices as I

META = ("pair_id", "river", "campaign", "site", "replicate", "variant")


def compute(ds: xr.Dataset, ex_method: str, bix_den) -> pd.DataFrame:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = I.compute_indices(ds, ex_method=ex_method, bix={"em_den": bix_den})
    df.attrs["warnings"] = [str(w.message) for w in caught]
    return df


def diff_summary(a: pd.Series, b: pd.Series) -> str:
    rel = ((b - a) / a.abs()).dropna() * 100
    return (f"median {rel.median():+.2f} %, range {rel.min():+.2f} to {rel.max():+.2f} %, "
            f"n = {rel.size}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=Path("outputs/ombrone_raw.nc"), type=Path)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/indices"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    icfg = cfg["indices"]
    den = icfg["bix_em_den"]
    den = tuple(den) if isinstance(den, list) else float(den)

    base = correct_until_raman(xr.load_dataset(args.dataset, engine="scipy"), cfg)
    if icfg["stage"] != "after_raman":
        raise SystemExit("only stage = 'after_raman' is implemented")
    main_df = compute(base, icfg["ex_method"], den)
    csv, side = I.export_indices(main_df, args.out / "indices.csv", base, META)
    print(f"Saved {csv} and {side}")

    # --- comparisons for the choices still to be confirmed -----------------------
    s = cfg["scatter"]
    scat = C.remove_scatter(base, widths={b: tuple(s[b]) for b in C.scatter.BANDS if b in s},
                            raman_shift_cm1=s["raman_shift_cm1"],
                            zero_below_rayleigh=s["zero_below_rayleigh"])
    alt = {
        "nearest excitation": compute(base, "nearest", den),
        "after scatter removal": compute(scat, icfg["ex_method"], den),
        "BIX max 420-435": compute(base, icfg["ex_method"], (420.0, 435.0)),
    }
    lines = ["# Phase 3: index settings compared", "",
             f"Main settings: stage {icfg['stage']}, ex_method {icfg['ex_method']}, "
             f"BIX denominator {den}. Spectral correction: "
             f"{main_df.attrs['settings']['spectral_correction']}.", ""]
    for name, df in alt.items():
        lines.append(f"## {name} vs main")
        for col in main_df.columns:
            n_nan = int(df[col].isna().sum())
            if df[col].equals(main_df[col]):
                continue
            lines.append(f"- {col}: {diff_summary(main_df[col], df[col])}"
                         + (f"; NaN in {n_nan} samples" if n_nan else ""))
        lines.append("")

    # replicate agreement (CV of field triplicates) for the main settings
    tab = main_df.join(pd.DataFrame({m: base[m].to_series() for m in META[1:]}))
    grp = tab.groupby(["river", "campaign", "site", "variant"])
    cv = (grp[list(main_df.columns)].std() / grp[list(main_df.columns)].mean().abs() * 100)
    lines += ["## Replicate CV (%), median over site-dates", "",
              "```", cv.median().round(2).to_string(), "```", ""]
    summ = tab.groupby(["river", "campaign"])[["FI", "HIX", "BIX", "peak_C", "peak_T"]] \
        .median().round(3)
    lines += ["## Median per river and campaign", "", "```", summ.to_string(), "```", ""]
    if main_df.attrs["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in main_df.attrs["warnings"]]
    (args.out / "indices_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col in zip(axes, ("FI", "HIX", "BIX"), strict=True):
        for k, (river, g) in enumerate(tab.groupby("river")):
            camps = sorted(g.campaign.unique())
            for j, c in enumerate(camps):
                v = g[g.campaign == c][col]
                a.scatter(np.full(v.size, k * 6 + j) + np.random.default_rng(0).normal(0, .08,
                          v.size), v, s=8, color=f"C{k}")
        a.set_xticks([k * 6 + 1.5 for k in range(3)], sorted(tab.river.unique()))
        a.set_title(col)
    fig.suptitle("Indices per sample (dots grouped by campaign within each river)")
    fig.tight_layout()
    fig.savefig(args.out / "indices_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
