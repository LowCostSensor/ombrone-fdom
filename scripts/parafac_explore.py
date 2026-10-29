"""Exploratory PARAFAC models for the Ombrone dataset (ranks, diagnostics, split-half).

Usage:
    python scripts/parafac_explore.py --dataset outputs/ombrone_corrected.nc \
        --config config/processing.toml --out outputs/parafac
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from correct import load_config

from eemkit import EEMKitWarning
from eemkit import parafac as P


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def plot_loadings(models: dict[int, xr.Dataset], path: Path) -> None:
    ranks = sorted(models)
    fig, axes = plt.subplots(len(ranks), 1, figsize=(7, 2.1 * len(ranks)), squeeze=False)
    for ax, r in zip(axes[:, 0], ranks, strict=True):
        m = models[r]
        for f in range(r):
            ax.plot(m.ex, m.ex_loadings[:, f], color=f"C{f}", ls="--", lw=1)
            ax.plot(m.em, m.em_loadings[:, f], color=f"C{f}", lw=1.3, label=f"C{f + 1}")
        ax.set_ylabel(f"{r} comp.")
        ax.legend(fontsize=7, ncol=r, loc="upper right")
    axes[-1, 0].set_xlabel("wavelength (nm): dashed = excitation, solid = emission")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outliers(models: dict[int, xr.Dataset], ranks, path: Path, n_label: int = 6) -> None:
    fig, axes = plt.subplots(1, len(ranks), figsize=(4.2 * len(ranks), 3.8), squeeze=False)
    for ax, r in zip(axes[0], ranks, strict=True):
        m = models[r]
        lev = m.leverage_sample.values
        sse = m.sse_sample_normalized.values
        rivers = m.river.values
        for k, river in enumerate(sorted(set(rivers))):
            sel = rivers == river
            ax.scatter(lev[sel], sse[sel], s=10, color=f"C{k}", label=river)
        for i in np.argsort(-(lev / lev.max() + sse / sse.max()))[:n_label]:
            ax.annotate(str(m.sample.values[i]).replace("_2024-", " "), (lev[i], sse[i]),
                        fontsize=6)
        ax.set_xlabel("sample leverage")
        ax.set_ylabel("residual SS (normalised EEM)")
        ax.set_title(f"{r} components")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=Path("outputs/ombrone_corrected.nc"), type=Path)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/parafac"), type=Path)
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="sample_ids to leave out, overriding parafac.exclude (what-if runs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)["parafac"]
    if args.exclude is not None:
        cfg["exclude"] = args.exclude
    sh_cfg = cfg["split_half"]
    fit_kw = dict(normalization=cfg["normalization"], tol=cfg["tol"], max_iter=cfg["max_iter"],
                  random_state=cfg["random_state"])

    ds = xr.load_dataset(args.dataset, engine="scipy")
    if cfg["exclude"]:
        ds = ds.drop_sel(sample=cfg["exclude"])
    strata = None
    if sh_cfg["stratify"]:
        ds = ds.assign_coords(stratum=("sample", ["|".join(str(ds[c].values[i])
                                                           for c in sh_cfg["stratify"])
                                                  for i in range(ds.sizes["sample"])]))
        strata = "stratum"

    fh = open(args.out / "explore.log", "w", encoding="utf-8")
    log(f"{ds.sizes['sample']} samples, ex {ds.ex.min().item():.0f}-{ds.ex.max().item():.0f},"
        f" em {ds.em.min().item():.0f}-{ds.em.max().item():.0f}, "
        f"{float(np.isnan(ds.intensity).mean()) * 100:.1f} % missing", fh)
    models, splits, rows = {}, {}, []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EEMKitWarning)
        for r in cfg["ranks"]:
            t = time.time()
            m = P.fit_parafac(ds, r, n_init=cfg["n_init"], **fit_kw)
            m.to_netcdf(args.out / f"model_{r}.nc", engine="scipy")
            models[r] = m
            log(f"rank {r}: explained {m.attrs['explained_variance_pct']:.3f} %, CORCONDIA "
                f"{m.attrs['core_consistency_pct']:.1f} %, starts at best "
                f"{m.attrs['n_init_at_best_minimum']}/{m.attrs['n_init']} "
                f"({time.time() - t:.0f} s)", fh)
            t = time.time()
            sh = P.split_half(ds, r, stratify=strata,
                              split_random_state=sh_cfg["random_state"],
                              threshold=sh_cfg["threshold"], n_init=sh_cfg["n_init"], **fit_kw)
            sh.to_netcdf(args.out / f"split_half_{r}.nc", engine="scipy")
            splits[r] = sh
            log(f"rank {r}: split-half min TCC {sh.attrs['min_tcc']:.3f} -> "
                f"{'VALIDATED' if sh.attrs['validated'] else 'not validated'} "
                f"({time.time() - t:.0f} s)", fh)
    for w in caught:
        log(f"WARNING {w.message}", fh)

    table = P.summary_table(models)
    table["split_half_min_tcc"] = [splits[r].attrs["min_tcc"] for r in table.index]
    table["split_half_validated"] = [bool(splits[r].attrs["validated"]) for r in table.index]
    table.to_csv(args.out / "summary.csv")
    tcc = pd.concat({r: pd.DataFrame(np.minimum(s.tcc_ex.values, s.tcc_em.values),
                                     index=s.test.values,
                                     columns=[f"C{c}" for c in s.component.values])
                     for r, s in splits.items()})
    tcc.to_csv(args.out / "split_half_tcc.csv")
    log("\n" + table.round(3).to_string(), fh)

    plot_loadings(models, args.out / "loadings_by_rank.png")
    plot_outliers(models, [r for r in (4, 5, 6) if r in models], args.out / "outliers.png")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(table.index, table.core_consistency_pct, "o-", label="core consistency (%)")
    ax.set_xlabel("components")
    ax.set_ylabel("core consistency (%)")
    ax2 = ax.twinx()
    ax2.plot(table.index, table.split_half_min_tcc, "s--", color="C1", label="split-half min TCC")
    ax2.axhline(sh_cfg["threshold"], color="C1", lw=0.8, ls=":")
    ax2.set_ylabel("min Tucker congruence")
    fig.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out / "rank_diagnostics.png", dpi=150)
    plt.close(fig)
    log("done", fh)
    fh.close()


if __name__ == "__main__":
    main()
