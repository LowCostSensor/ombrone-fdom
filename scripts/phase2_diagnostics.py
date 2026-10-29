"""Diagnostics used to choose the Phase 2 correction parameters for the Ombrone data.

Works on the raw dataset written by build_dataset.py and changes nothing.
Outputs figures and a Markdown summary in --out.

Usage:
    python scripts/phase2_diagnostics.py --dataset outputs/ombrone_raw.nc --out outputs/phase2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from eemkit import corrections as C

RAMAN_EX = 350.0
LS_RANGE = (381.0, 426.0)  # Lawaetz & Stedmon (2009)
PROVISIONAL_WIDTH = 15.0  # nm each side, only to exclude scatter from the noise statistics
# Widths proposed from these diagnostics, shown for review only (not yet confirmed)
PROPOSED_WIDTHS = {"rayleigh1": (10.0, 10.0), "raman1": (8.0, 8.0),
                   "rayleigh2": (12.0, 12.0), "raman2": (10.0, 10.0)}
PROPOSED_ZERO_BELOW = 10.0


def one_sample_per_blank(ds: xr.Dataset) -> np.ndarray:
    _, idx = np.unique(ds["eem_blank_source_file"].values.astype(str), return_index=True)
    return ds.sample.values[np.sort(idx)]


def md(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def raman_peaks(ds: xr.Dataset) -> pd.DataFrame:
    """Peak position, FWHM and area of the water Raman band in each distinct blank."""
    rows = []
    for sample in one_sample_per_blank(ds):
        src = ds["eem_blank_source_file"].sel(sample=sample).item()
        spec = ds["blank_intensity"].sel(sample=sample, ex=RAMAN_EX)
        win = spec.sel(em=slice(360, 440))
        em, y = win["em"].values, win.values
        base = np.linspace(y[0], y[-1], y.size)
        net = y - base
        i = int(np.argmax(net))
        # parabolic refinement of the maximum
        if 0 < i < y.size - 1:
            a, b, c = net[i - 1], net[i], net[i + 1]
            peak = em[i] + 0.5 * (em[1] - em[0]) * (a - c) / (a - 2 * b + c)
        else:
            peak = em[i]
        half = net[i] / 2
        above = em[net >= half]
        rows.append({
            "blank_file": Path(str(src)).parent.name + "/" + Path(str(src)).name,
            "peak_em_nm": round(float(peak), 2),
            "shift_cm1": round(1e7 / RAMAN_EX - 1e7 / peak, 0),
            "fwhm_nm": float(above.max() - above.min()),
            "area_LS": float(C.raman_area(spec.expand_dims(ex=[RAMAN_EX]), ex=RAMAN_EX,
                                          em_range=LS_RANGE).item()),
            "area_LS_baseline": float(C.raman_area(spec.expand_dims(ex=[RAMAN_EX]),
                                                   ex=RAMAN_EX, em_range=LS_RANGE,
                                                   baseline=True).item()),
            "measured": str(ds["eem_blank_collection_start"].sel(sample=sample).item())[:10]
            if "eem_blank_collection_start" in ds.coords else "",
        })
    return pd.DataFrame(rows).sort_values("measured")


def rayleigh_profiles(ds: xr.Dataset, out: Path) -> pd.DataFrame:
    """Width of first-order Rayleigh in raw samples: where it falls to 5 % above the floor."""
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for ex in (260.0, 300.0, 350.0, 400.0, 450.0):
        offs = np.arange(-20, 41, 2.0)
        em = ex + offs
        ok = (em >= ds.em.min().item()) & (em <= ds.em.max().item())
        prof = ds["intensity"].sel(ex=ex).interp(em=em[ok])
        med = prof.median("sample").values
        floor = np.median(med[offs[ok] >= 30])
        rel = (med - floor) / (med.max() - floor)
        ax.semilogy(offs[ok], np.clip(rel, 1e-3, None), label=f"Ex {ex:.0f}")
        above = offs[ok][rel > 0.05]
        below_side = offs[ok][(offs[ok] < 0) & (rel > 0.05)]
        rows.append({"ex": ex, "upper_edge_5pct_nm": float(above.max()),
                     "lower_edge_5pct_nm": float(below_side.min()) if below_side.size else None})
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.set_xlabel("em - ex (nm)")
    ax.set_ylabel("median raw intensity, scaled (peak = 1)")
    ax.set_title("First-order Rayleigh band in the samples")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rayleigh_profile.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def excitation_quality(ds: xr.Dataset, out: Path, shift: float) -> pd.DataFrame:
    """Per-excitation signal, blank fraction, negatives and replicate CV after blank subtraction."""
    sub = C.subtract_blank(ds, missing="drop")
    widths = {b: (PROVISIONAL_WIDTH, PROVISIONAL_WIDTH) for b in C.scatter.BANDS}
    clean = C.remove_scatter(sub, widths=widths, raman_shift_cm1=shift,
                             variables=("intensity", "blank_intensity"))
    keep = clean["em"] >= clean["ex"] + PROVISIONAL_WIDTH  # only above first-order Rayleigh
    sig = clean["intensity"].where(keep)
    blk = clean["blank_intensity"].where(keep)

    rows = []
    integ = sig.sum("em", skipna=True)
    cv = []
    gid = pd.Series([f"{r}|{c}|{s}|{v}" for r, c, s, v in zip(
        sig.river.values, sig.campaign.values, sig.site.values, sig.variant.values,
        strict=True)], index=sig.sample.values)
    df_int = integ.to_pandas()  # sample x ex
    for _, idx in gid.groupby(gid).groups.items():
        block = df_int.loc[list(idx)]
        if len(block) >= 2:
            cv.append(block.std(ddof=1) / block.mean().abs())
    cv = pd.concat(cv, axis=1)

    for ex in sig.ex.values:
        s = sig.sel(ex=ex)
        b = blk.sel(ex=ex)
        vals = s.values[np.isfinite(s.values)]
        rows.append({
            "ex": float(ex),
            "median_signal": float(np.nanmedian(s.sum("em").values)),
            "blank_to_sample": float(np.nanmedian((b.sum("em") / (s + b).sum("em")).values)),
            "frac_negative": float((vals < 0).mean()) if vals.size else np.nan,
            "median_replicate_cv": float(cv.loc[ex].median()),
            "p90_replicate_cv": float(cv.loc[ex].quantile(0.9)),
        })
    q = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].semilogy(q.ex, q.median_signal, "o-")
    axes[0].set_title("median integrated signal (a.u. nm)")
    axes[1].plot(q.ex, q.median_replicate_cv * 100, "o-", label="median")
    axes[1].plot(q.ex, q.p90_replicate_cv * 100, "s--", label="90th pct")
    axes[1].set_title("CV between field replicates (%)")
    axes[1].legend(fontsize=8)
    axes[2].plot(q.ex, q.blank_to_sample * 100, "o-", label="blank / sample")
    axes[2].plot(q.ex, q.frac_negative * 100, "s--", label="negative values")
    axes[2].set_title("%")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.set_xlabel("excitation (nm)")
        a.axvline(240, color="grey", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(out / "excitation_quality.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, river in zip(axes, ("Orcia", "Merse", "Farma"), strict=True):
        m = sig.where(sig.river == river, drop=True).median("sample")
        lv = np.nanpercentile(m.values, [2, 99])
        cs = a.contourf(m.em, m.ex, m.values, levels=np.linspace(*lv, 30), extend="both")
        a.set_title(f"{river}: median, blank-subtracted (a.u.)")
        a.set_xlabel("emission (nm)")
        a.axhline(240, color="w", lw=0.8, ls=":")
        fig.colorbar(cs, ax=a)
    axes[0].set_ylabel("excitation (nm)")
    fig.tight_layout()
    fig.savefig(out / "median_eem_by_river.png", dpi=150)
    plt.close(fig)
    return q


def scatter_preview(ds: xr.Dataset, out: Path, shift: float, samples: list[str]) -> None:
    """Raw vs blank-subtracted + proposed scatter removal for a few samples."""
    sub = C.subtract_blank(ds.sel(sample=samples))
    clean = C.remove_scatter(sub, widths=PROPOSED_WIDTHS, raman_shift_cm1=shift,
                             zero_below_rayleigh=PROPOSED_ZERO_BELOW)
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 3.6 * len(samples)), squeeze=False)
    for row, s in zip(axes, samples, strict=True):
        for a, da, title in ((row[0], ds["intensity"].sel(sample=s), "raw"),
                             (row[1], clean["intensity"].sel(sample=s),
                              "blank-subtracted, proposed scatter removal")):
            v = da.values
            hi = np.nanpercentile(clean["intensity"].sel(sample=s).values, 99.5)
            cs = a.contourf(da.em, da.ex, np.clip(v, 0, hi), levels=np.linspace(0, hi, 30))
            fig.colorbar(cs, ax=a)
            a.set_title(f"{s}: {title}", fontsize=9)
            a.set_xlabel("emission (nm)")
            a.set_ylabel("excitation (nm)")
    fig.tight_layout()
    fig.savefig(out / "scatter_preview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=Path("outputs/ombrone_raw.nc"), type=Path)
    ap.add_argument("--out", default=Path("outputs/phase2"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = xr.load_dataset(args.dataset, engine="scipy")

    raman = raman_peaks(ds)
    shift = float(raman.shift_cm1.median())
    fig, ax = plt.subplots(figsize=(6, 4))
    for sample in one_sample_per_blank(ds):
        sp = ds["blank_intensity"].sel(sample=sample, ex=RAMAN_EX).sel(em=slice(360, 450))
        ax.plot(sp.em, sp, lw=1)
    ax.axvspan(*LS_RANGE, color="grey", alpha=0.2, label="381-426 nm")
    ax.set_xlabel("emission (nm)")
    ax.set_ylabel("blank intensity (a.u.)")
    ax.set_title("Milli-Q blanks, Ex 350 nm (one line per session)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "raman_blanks.png", dpi=150)
    plt.close(fig)

    ray = rayleigh_profiles(ds, args.out)
    q = excitation_quality(ds, args.out, shift)
    preview = [s for s in ("Merse_2024-04_S3_R1", "Farma_2024-11_S5_R2") if s in ds.sample]
    scatter_preview(ds, args.out, shift, preview)

    lines = ["# Phase 2 diagnostics", "",
             "## Water Raman peak in the blanks (Ex 350 nm)", "",
             md(raman.round(2)), "",
             f"Median shift: {shift:.0f} cm-1; area CV between sessions: "
             f"{raman.area_LS.std() / raman.area_LS.mean() * 100:.1f} %", "",
             "## First-order Rayleigh width in samples (median profile)", "",
             md(ray), "",
             "## Quality per excitation (after blank subtraction, scatter excluded)", "",
             md(q.round(4)), ""]
    (args.out / "phase2_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
