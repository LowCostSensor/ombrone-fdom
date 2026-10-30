"""Publication figures for the Ombrone fDOM dataset (PDF + 300 dpi PNG).

Main figures
    fig1_processing   one sample through the correction steps
    fig2_rivers       a corrected EEM per river
    fig3_rank         choice of the number of components
    fig4_components   the final PARAFAC components with split-half models
    fig5_fmax         Fmax per campaign and river
    fig6_indices      HIX, BIX and FI per campaign and river
Supplementary
    figS1_outliers    leverage vs residuals, 4 components on all samples
    figS2_leverage    leverage vs residuals, final model

Representative samples are chosen by rule, not by eye: the sample whose total
Fmax is the median of the final model (fig1) and of each river (fig2).

Usage:
    python scripts/figures.py --out outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from correct import load_config, path_length

from eemkit import corrections as C
from eemkit import plot

COMPONENT_NAMES = ["C1\nhumic-like (peaks A + M)", "C2\nterrestrial humic-like (A + C)",
                   "C3\ntryptophan-like (peak T)"]
MONTHS = {f"2024-{m:02d}": name for m, name in
          enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                     "Nov", "Dec"], start=1)}


def median_sample(fmax: pd.DataFrame) -> str:
    total = fmax.sum(axis=1)
    return str((total - total.median()).abs().idxmin())


def processing_steps(raw: xr.Dataset, cfg: dict, sample: str) -> dict[str, xr.DataArray]:
    ds = raw.sel(sample=[sample])
    missing = cfg["missing"]["policy"]
    steps = {"a  Raw sample (a.u.)": ds["intensity"]}
    ds = C.subtract_absorbance_blank(ds, missing=missing)
    ds = C.correct_inner_filter(
        ds, absorbance_path_length_cm=path_length(cfg, "absorbance_path_length_cm"),
        fluorescence_path_length_cm=path_length(cfg, "fluorescence_path_length_cm"),
        missing=missing)
    ds = C.subtract_blank(ds, missing=missing)
    steps["b  Inner filter, blank subtracted (a.u.)"] = ds["intensity"]
    r = cfg["raman"]
    ds = C.normalize_raman(ds, ex=r["ex"], em_range=tuple(r["em_range"]),
                           baseline=r["baseline"], missing=missing)
    steps["c  Raman units (R.U.)"] = ds["intensity"]
    s = cfg["scatter"]
    ds = C.remove_scatter(ds, widths={b: tuple(s[b]) for b in C.scatter.BANDS if b in s},
                          raman_shift_cm1=s["raman_shift_cm1"],
                          zero_below_rayleigh=s["zero_below_rayleigh"])
    ds = C.crop(ds, ex=(cfg["crop"]["ex_min"], None))
    steps["d  Scatter removed, ex ≥ 240 nm (R.U.)"] = ds["intensity"]
    return {k: v.isel(sample=0) for k, v in steps.items()}


def summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="rank")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/figures"), type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out
    rank = cfg["parafac"]["final"]["rank"]

    raw = xr.load_dataset("outputs/ombrone_raw.nc", engine="scipy")
    cor = xr.load_dataset("outputs/ombrone_corrected.nc", engine="scipy")
    model = xr.load_dataset(f"outputs/parafac_final/model_{rank}.nc", engine="scipy")
    split = xr.load_dataset(f"outputs/parafac_final/split_half_{rank}.nc", engine="scipy")
    fmax = pd.read_csv("outputs/parafac_final/fmax.csv", index_col="sample")
    fcols = [c for c in fmax.columns if c.endswith("_Fmax")]
    written = []

    s1 = median_sample(fmax[fcols])
    steps = processing_steps(raw, cfg, s1)
    # colour scale of each step capped at its maximum inside the region kept for PARAFAC,
    # so the fluorescence is readable next to the much brighter scatter
    kept = steps[list(steps)[-1]].notnull()
    vmax = {k: float(v.sel(ex=kept.ex, em=kept.em).where(kept).max()) for k, v in steps.items()}
    fig = plot.eems(steps, ncols=2, vmax=vmax)
    fig.suptitle(f"Processing of {s1}", x=0.01, ha="left", fontsize=8)
    written += plot.save(fig, out / "fig1_processing")

    panels = {}
    for river, sub in fmax.groupby("river"):
        s = median_sample(sub[fcols])
        panels[f"{river}  ({s})"] = cor["intensity"].sel(sample=s)
    written += plot.save(plot.eems(panels, ncols=3, shared_scale=True), out / "fig2_rivers")

    fig = plot.rank_diagnostics(summary(Path("outputs/parafac_excl_orcia05_s56/summary.csv")),
                                threshold=cfg["parafac"]["split_half"]["threshold"],
                                chosen=rank)
    written += plot.save(fig, out / "fig3_rank")

    written += plot.save(plot.components(model, split, names=COMPONENT_NAMES[:rank]),
                         out / "fig4_components")

    fig = plot.scores_over(fmax.reset_index(), fcols, x="campaign", by="river",
                           titles=COMPONENT_NAMES[:rank], ylabel="Fmax (R.U.)",
                           x_labels=MONTHS, xlabel="Campaign (2024)")
    written += plot.save(fig, out / "fig5_fmax")

    idx = pd.read_csv("outputs/indices/indices.csv")
    fig = plot.scores_over(idx, ["HIX", "BIX", "FI"], x="campaign", by="river",
                           titles=["HIX (Ohno 2002)", "BIX (Huguet et al. 2009)",
                                   "FI (Cory & McKnight 2005)"],
                           x_labels=MONTHS, xlabel="Campaign (2024)")
    written += plot.save(fig, out / "fig6_indices")

    all4 = xr.load_dataset("outputs/parafac/model_4.nc", engine="scipy")
    written += plot.save(plot.leverage(all4, by="river", n_label=6), out / "figS1_outliers")
    written += plot.save(plot.leverage(model, by="river", n_label=4), out / "figS2_leverage")

    for p in written:
        print(p)
    excluded = np.setdiff1d(cor.sample.values, model.sample.values)
    print(f"final model: {model.sizes['sample']} samples; not in model: {list(excluded)}")


if __name__ == "__main__":
    main()
