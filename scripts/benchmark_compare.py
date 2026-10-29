"""Compare eemkit with staRdom on the Ombrone data (Phase 5 benchmark).

Reads the tables of scripts/benchmark_export.py and scripts/benchmark_stardom.R
and writes ``benchmark_report.md``, ``benchmark_summary.json`` and figures.

Where the two toolboxes use different definitions, eemkit is rerun with the
staRdom definition so that the comparison tests the implementation:

* Raman area: staRdom integrates ``seq(371, 428, by = 2)``, i.e. em 371-427 nm
  (eemkit config: 381-426);
* water Raman shift for scatter removal: staRdom 3600 cm-1 (config: 3400);
* FI: eemR uses em 450/500 nm (eemkit default: 470/520, Cory & McKnight 2005).

Usage:
    python scripts/benchmark_compare.py --out outputs/benchmark
"""

from __future__ import annotations

import argparse
import copy
import json
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
from eemkit.parafac.diagnostics import match_components

TCC_THRESHOLD = 0.95


def read_eems(path: Path) -> xr.DataArray:
    """Inverse of benchmark_export.eem_table -> (sample, ex, em)."""
    d = pd.read_csv(path)
    samples = list(dict.fromkeys(d["sample"]))
    em = d.loc[d["sample"] == samples[0], "em"].to_numpy(float)
    ex = np.array([float(c) for c in d.columns[2:]])
    vals = d.iloc[:, 2:].to_numpy(float).reshape(len(samples), em.size, ex.size)
    return xr.DataArray(vals.transpose(0, 2, 1), dims=("sample", "ex", "em"),
                        coords={"sample": samples, "ex": ex, "em": em})


def rel_diff(a: xr.DataArray, b: xr.DataArray, floor: float = 0.01,
             keep: xr.DataArray | None = None) -> dict:
    """Relative differences |a-b|/|b| on cells above ``floor`` x the sample maximum of b
    (and inside ``keep``, an (ex, em) boolean mask, if given)."""
    big = abs(b) > floor * abs(b).max(("ex", "em"))
    if keep is not None:
        big = big & keep
    r = (abs(a - b) / abs(b)).where(big & a.notnull() & b.notnull())
    v = r.values[np.isfinite(r.values)]
    return {"cells": int(v.size), "median": float(np.median(v)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def corrections(inp: Path, sd: Path, cfg: dict) -> tuple[dict, list[str]]:
    raw = xr.load_dataset("outputs/ombrone_raw.nc", engine="scipy")
    res, lines = {}, ["## Corrections", ""]

    sr_area = pd.read_csv(sd / "raman_area.csv", index_col="sample")["raman_area"]
    for hi in (428, 427):
        ek_area = C.raman.raman_area(raw["blank_intensity"], ex=350, em_range=(371, hi),
                                     baseline=False).to_series()
        d = (ek_area / sr_area.reindex(ek_area.index) - 1).abs()
        res[f"raman_area_371_{hi}"] = {"median": float(d.median()), "max": float(d.max())}
    a8, a7 = res["raman_area_371_428"], res["raman_area_371_427"]
    lines += ["**Raman area** (blank, ex 350). staRdom documents em 371-428 nm but evaluates "
              "`seq(371, 428, by = 2)`, which ends at 427 nm. Relative difference vs eemkit "
              f"over 371-428: median {a8['median']:.1e}, max {a8['max']:.1e}; over 371-427: "
              f"median {a7['median']:.1e}, max {a7['max']:.1e}. The residual is staRdom's "
              "trapezoid rule on points interpolated at odd wavelengths, which straddle the "
              "kinks of the measured 2 nm spectrum; eemkit integrates the piecewise-linear "
              "spectrum exactly.", ""]

    cfg_sr = copy.deepcopy(cfg)
    cfg_sr["raman"]["em_range"] = [371, 427]
    ek_ds = correct_until_raman(raw, cfg_sr)
    ek = ek_ds["intensity"]
    sr = read_eems(sd / "after_raman_eemkit_order.csv").sel(sample=ek["sample"].values)
    r1 = rel_diff(ek, sr)
    res["after_raman_same_order"] = r1
    # undo each toolbox's own Raman division: isolates inner filter + blank
    sr_area_x = xr.DataArray(sr_area.reindex(ek["sample"].values).values, dims="sample",
                             coords={"sample": ek["sample"].values})
    r0 = rel_diff(ek * ek_ds["raman_area"], sr * sr_area_x)
    res["ife_and_blank"] = r0
    s = cfg["scatter"]
    widths = {b: tuple(s[b]) for b in C.scatter.BANDS}
    fluo = ~C.scatter.scatter_mask(ek.ex.values, ek.em.values, widths, s["raman_shift_cm1"])
    fluo = fluo & (ek.em > ek.ex + s["zero_below_rayleigh"]) & (ek.ex >= cfg["crop"]["ex_min"])
    sr2 = read_eems(sd / "after_raman_stardom_order.csv").sel(sample=ek["sample"].values)
    r2 = rel_diff(sr2, sr)
    r2f = rel_diff(sr2, sr, keep=fluo)
    res["order_effect_blank_before_ife"] = {"all_cells": r2, "modelled_region": r2f}
    lines += ["**Inner filter + blank + Raman units**, same order and Raman range in both "
              f"(205 samples, {r1['cells']} cells above 1 % of each sample's maximum): "
              f"relative difference median {r1['median']:.1e}, 99th percentile "
              f"{r1['p99']:.1e}, max {r1['max']:.1e}. Before the Raman division (inner filter "
              f"+ blank only): median {r0['median']:.1e}, max {r0['max']:.1e}, so the whole "
              "difference is the Raman-area quadrature above.", "",
              "**Order effect** (staRdom tutorial order, blank subtracted before the "
              "inner-filter correction, vs eemkit order; both computed with staRdom). "
              "The two differ by blank x (IFE factor - 1), large only where the blank is "
              f"large: in the region kept for PARAFAC (outside scatter, em > ex + "
              f"{s['zero_below_rayleigh']}, ex >= {cfg['crop']['ex_min']}) median "
              f"{r2f['median']:.1e}, 99th percentile {r2f['p99']:.1e}, max {r2f['max']:.1e}; "
              f"over all cells, including the scatter bands, median {r2['median']:.1e}, "
              f"max {r2['max']:.1e}. A difference of method, not of implementation.", ""]

    m = pd.read_csv(sd / "scatter_mask.csv", index_col="em")
    sr_mask = xr.DataArray(m.to_numpy(bool).T, dims=("ex", "em"),
                           coords={"ex": m.columns.astype(float), "em": m.index.values})
    rows = []
    for shift in (3600, s["raman_shift_cm1"]):
        ek_mask = C.scatter.scatter_mask(sr_mask.ex.values, sr_mask.em.values, widths, shift)
        diff = ek_mask != sr_mask
        # eemR keeps em == centre - width; eemkit removes it (closed interval)
        rows.append({"shift_cm1": shift, "cells_masked_eemkit": int(ek_mask.sum()),
                     "cells_masked_stardom": int(sr_mask.sum()),
                     "cells_different": int(diff.sum()),
                     "agreement_pct": float(100 * (1 - diff.mean()))})
    res["scatter_mask"] = rows
    lines += ["**Scatter bands** (same widths; staRdom's water Raman shift is fixed at "
              "3600 cm-1):", "",
              "| eemkit shift (cm-1) | cells removed eemkit | staRdom | cells different |"
              " agreement |", "|---|---|---|---|---|"]
    lines += [f"| {r['shift_cm1']} | {r['cells_masked_eemkit']} | {r['cells_masked_stardom']}"
              f" | {r['cells_different']} | {r['agreement_pct']:.2f} % |" for r in rows]
    lines += ["", "Remaining differences at the same shift are band edges: eemR keeps the "
              "point at exactly centre - width, eemkit removes it.", ""]
    return res, lines


def indices(inp: Path, sd: Path, cfg: dict) -> tuple[dict, list[str]]:
    ek_eem = read_eems(inp / "eemkit_after_raman.csv")
    ds = xr.Dataset({"intensity": ek_eem})
    ek = I.compute_indices(ds, ex_method="linear", fi={"em": (450.0, 500.0)},
                           bix={"em_den": 430.0})
    sr = pd.read_csv(sd / "indices.csv", index_col="sample").reindex(ek.index)
    pairs = {"FI (em 450/500)": ("FI", "fi"), "HIX": ("HIX", "hix"), "BIX": ("BIX", "bix"),
             "peak B": ("peak_B", "b"), "peak T": ("peak_T", "t"), "peak A": ("peak_A", "a"),
             "peak M": ("peak_M", "m"), "peak C": ("peak_C", "c")}
    res, lines = {}, ["## Indices", "",
                      "Computed by both toolboxes on the same eemkit EEMs (after Raman units). "
                      "FI with eemR's emission pair 450/500 nm.", "",
                      "| index | median rel. diff. | max rel. diff. | Pearson r |",
                      "|---|---|---|---|"]
    for name, (a, b) in pairs.items():
        d = (ek[a] / sr[b] - 1).abs()
        r = float(np.corrcoef(ek[a], sr[b])[0, 1])
        res[name] = {"median": float(d.median()), "max": float(d.max()), "pearson_r": r}
        lines.append(f"| {name} | {d.median():.1e} | {d.max():.1e} | {r:.6f} |")
    lines += ["", "HIX: eemR sums the spectrum interpolated every 1 nm (rectangle rule), eemkit "
              "integrates the piecewise-linear spectrum (trapezoid); this explains a small "
              "residual difference.", ""]
    return res, lines


def parafac(sd: Path, final: Path, out: Path) -> tuple[dict, list[str]]:
    ek_ex = pd.read_csv(final / "ex_loadings.csv", index_col="ex")
    ek_em = pd.read_csv(final / "em_loadings.csv", index_col="em")
    sr_ex = pd.read_csv(sd / "pf_ex_loadings.csv", index_col="ex")
    sr_em = pd.read_csv(sd / "pf_em_loadings.csv", index_col="em")
    info = pd.read_csv(sd / "pf_info.csv").iloc[0]
    assert np.allclose(ek_ex.index, sr_ex.index) and np.allclose(ek_em.index, sr_em.index)
    order, tex, tem = match_components(ek_ex.values, ek_em.values, sr_ex.values, sr_em.values)
    sr_ex, sr_em = sr_ex.iloc[:, order], sr_em.iloc[:, order]

    ek_f = pd.read_csv(final / "fmax.csv", index_col="sample")
    ek_f = ek_f[[c for c in ek_f.columns if c.endswith("_Fmax")]]
    sr_f = pd.read_csv(sd / "pf_fmax.csv", index_col="sample").iloc[:, order]
    sr_f = sr_f.reindex(ek_f.index)
    model = xr.load_dataset(final / f"model_{len(order)}.nc", engine="scipy")

    rows, res = [], {"components": []}
    for k in range(len(order)):
        a, b = ek_f.iloc[:, k].to_numpy(), sr_f.iloc[:, k].to_numpy()
        ratio = b / a
        c = {"component": f"C{k + 1}", "tcc_ex": float(tex[k]), "tcc_em": float(tem[k]),
             "fmax_pearson_r": float(np.corrcoef(a, b)[0, 1]),
             "fmax_ratio_median": float(np.median(ratio)),
             "fmax_ratio_range": [float(ratio.min()), float(ratio.max())]}
        res["components"].append(c)
        rows.append(f"| C{k + 1} | {tex[k]:.4f} | {tem[k]:.4f} | {c['fmax_pearson_r']:.5f} | "
                    f"{c['fmax_ratio_median']:.4f} ({ratio.min():.3f}-{ratio.max():.3f}) |")
    def reconstruct(fm: pd.DataFrame, ex: pd.DataFrame, em: pd.DataFrame) -> np.ndarray:
        return np.einsum("sf,xf,yf->sxy", fm.to_numpy(), (ex / ex.max()).to_numpy(),
                         (em / em.max()).to_numpy())

    r_ek, r_sr = reconstruct(ek_f, ek_ex, ek_em), reconstruct(sr_f, sr_ex, sr_em)
    big = r_ek > 0.01 * r_ek.max(axis=(1, 2), keepdims=True)
    rd = np.abs(r_sr - r_ek)[big] / r_ek[big]
    res["reconstruction_rel_diff"] = {"median": float(np.median(rd)),
                                      "p99": float(np.percentile(rd, 99))}
    unit = ek_ex / np.linalg.norm(ek_ex, axis=0)
    cong = unit.T.to_numpy() @ unit.to_numpy()
    i, j = np.unravel_index(np.argmax(np.triu(cong, 1)), cong.shape)
    res["max_within_model_ex_congruence"] = {"pair": [f"C{i + 1}", f"C{j + 1}"],
                                             "tcc": float(cong[i, j])}
    res["min_tcc"] = float(min(tex.min(), tem.min()))
    res["passed"] = res["min_tcc"] > TCC_THRESHOLD
    res["stardom"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()}
    res["eemkit"] = {"explained_variance_pct": model.attrs["explained_variance_pct"],
                     "core_consistency_pct": model.attrs["core_consistency_pct"]}
    lines = ["## PARAFAC", "",
             f"Same input: eemkit's corrected EEMs of the {len(ek_f)} samples in the final model "
             "(scatter as missing values). Both fitted with 3 non-negative components, 20 "
             "random starts, convergence 1e-8. Differences of method: eemkit scales each "
             "sample to unit Frobenius norm and uses HALS with missing-value imputation; "
             "staRdom scales each sample by its standard deviation and uses multiway::parafac "
             "(ALS with non-negativity).", "",
             f"| | eemkit | staRdom {info['stardom']} |", "|---|---|---|",
             f"| explained variance / R² | {model.attrs['explained_variance_pct']:.3f} % | "
             f"{100 * info['rsq']:.3f} % |",
             f"| core consistency | {model.attrs['core_consistency_pct']:.1f} % | "
             + (f"{info['corcondia']:.1f} % |" if pd.notna(info["corcondia"])
                else "n/a (eempf_corcondia returns NA with missing values) |"),
             f"| starts converged | {model.attrs['n_init_at_best_minimum']}/20 at best minimum "
             f"| {100 * info['converged_fraction']:.0f} % converged |", "",
             "| component | TCC ex | TCC em | Fmax Pearson r | Fmax ratio staRdom/eemkit, "
             "median (range) |", "|---|---|---|---|---|", *rows, "",
             f"**Minimum TCC {res['min_tcc']:.4f}: "
             f"{'PASSED' if res['passed'] else 'FAILED'}** (criterion > {TCC_THRESHOLD}).", "",
             "Fmax agrees in ranking (Pearson r) for every component; constant per-component "
             "scale offsets are a trade-off between overlapping components, not a different "
             f"fit: the excitation loadings of C{i + 1} and C{j + 1} have congruence "
             f"{cong[i, j]:.3f} within each model, and the EEMs reconstructed by the two models "
             f"differ by a median of {np.median(rd):.1e} (99th percentile "
             f"{np.percentile(rd, 99):.1e}). Fmax values from different toolboxes should "
             "therefore not be mixed in one study.", ""]

    k = len(order)
    fig, axes = plt.subplots(2, k, figsize=(3.4 * k, 5.2), squeeze=False)
    for f in range(k):
        ax = axes[0, f]
        ax.plot(ek_ex.index, ek_ex.iloc[:, f] / ek_ex.iloc[:, f].max(), "C0--", label="eemkit ex")
        ax.plot(ek_em.index, ek_em.iloc[:, f] / ek_em.iloc[:, f].max(), "C0-", label="eemkit em")
        ax.plot(sr_ex.index, sr_ex.iloc[:, f] / sr_ex.iloc[:, f].max(), "C1:", lw=2,
                label="staRdom ex")
        ax.plot(sr_em.index, sr_em.iloc[:, f] / sr_em.iloc[:, f].max(), "C1-.", lw=1.5,
                label="staRdom em")
        ax.set_title(f"C{f + 1}: TCC ex {tex[f]:.3f}, em {tem[f]:.3f}", fontsize=9)
        ax.set_xlabel("wavelength (nm)")
        ax = axes[1, f]
        ax.scatter(ek_f.iloc[:, f], sr_f.iloc[:, f], s=8)
        lim = [0, max(ek_f.iloc[:, f].max(), sr_f.iloc[:, f].max()) * 1.05]
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlabel("eemkit Fmax (R.U.)")
        ax.set_ylabel("staRdom Fmax (R.U.)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "parafac_eemkit_vs_stardom.png", dpi=150)
    plt.close(fig)
    return res, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--final", default=Path("outputs/parafac_final"), type=Path)
    ap.add_argument("--out", default=Path("outputs/benchmark"), type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    inp, sd = args.out / "input", args.out / "stardom"

    summary, lines = {}, ["# eemkit vs staRdom benchmark (Ombrone data)", "",
                          "Reproduce with:", "", "```",
                          "python scripts/benchmark_export.py",
                          "Rscript scripts/benchmark_stardom.R corrections outputs/benchmark/input"
                          " outputs/benchmark/stardom",
                          "Rscript scripts/benchmark_stardom.R parafac outputs/benchmark/input "
                          "outputs/benchmark/stardom 3 20 20000 4",
                          "python scripts/benchmark_compare.py", "```", ""]
    for name, fn in (("corrections", lambda: corrections(inp, sd, cfg)),
                     ("indices", lambda: indices(inp, sd, cfg)),
                     ("parafac", lambda: parafac(sd, args.final, args.out))):
        res, txt = fn()
        summary[name] = res
        lines += txt
    (args.out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    (args.out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
