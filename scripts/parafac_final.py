"""Final PARAFAC model for the Ombrone dataset, as chosen in config [parafac.final].

Refits the chosen rank with tight convergence, repeats the split-half
validation, and exports scores (Fmax), loadings, an OpenFluor file and
figures.

Usage:
    python scripts/parafac_final.py --dataset outputs/ombrone_corrected.nc \
        --config config/processing.toml --out outputs/parafac_final
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
from correct import load_config

from eemkit import parafac as P

META = ("pair_id", "river", "campaign", "site", "replicate", "variant")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=Path("outputs/ombrone_corrected.nc"), type=Path)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/parafac_final"), type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)["parafac"]
    final = cfg.get("final", {})
    if not final.get("rank"):
        raise SystemExit("config: parafac.final.rank is not set; choose the number of "
                         "components after reviewing outputs/parafac")
    args.out.mkdir(parents=True, exist_ok=True)
    sh_cfg = cfg["split_half"]
    fit_kw = dict(normalization=cfg["normalization"], tol=final["tol"],
                  max_iter=final["max_iter"], random_state=cfg["random_state"])

    ds = xr.load_dataset(args.dataset, engine="scipy")
    if cfg["exclude"]:
        ds = ds.drop_sel(sample=cfg["exclude"])
    ds = ds.assign_coords(stratum=("sample", ["|".join(str(ds[c].values[i])
                                                       for c in sh_cfg["stratify"])
                                              for i in range(ds.sizes["sample"])]))
    r = final["rank"]
    m = P.fit_parafac(ds, r, n_init=final["n_init"], **fit_kw)
    m.to_netcdf(args.out / f"model_{r}.nc", engine="scipy")
    sh = P.split_half(ds, r, stratify="stratum", split_random_state=sh_cfg["random_state"],
                      threshold=sh_cfg["threshold"], n_init=sh_cfg["n_init"], **fit_kw)
    sh.to_netcdf(args.out / f"split_half_{r}.nc", engine="scipy")

    fmax = m["fmax"].to_pandas()
    fmax.columns = [f"C{c}_Fmax" for c in fmax.columns]
    for c in reversed(META):
        fmax.insert(0, c, m[c].to_series().reindex(fmax.index).values)
    fmax.to_csv(args.out / "fmax.csv", float_format="%.6g")
    pd.DataFrame(m["ex_loadings"].values, index=m.ex.values,
                 columns=[f"C{c}" for c in m.component.values]).rename_axis("ex").to_csv(
        args.out / "ex_loadings.csv")
    pd.DataFrame(m["em_loadings"].values, index=m.em.values,
                 columns=[f"C{c}" for c in m.component.values]).rename_axis("em").to_csv(
        args.out / "em_loadings.csv")
    P.to_openfluor(m, args.out / f"openfluor_{r}comp.txt", **final.get("openfluor", {}))

    fig, axes = plt.subplots(1, r, figsize=(3.2 * r, 3), squeeze=False)
    for f, ax in enumerate(axes[0]):
        comp = np.outer(m.ex_loadings[:, f], m.em_loadings[:, f])
        ax.contourf(m.em, m.ex, comp, levels=20)
        ax.set_title(f"C{f + 1}")
        ax.set_xlabel("emission (nm)")
    axes[0, 0].set_ylabel("excitation (nm)")
    fig.tight_layout()
    fig.savefig(args.out / "components.png", dpi=150)
    plt.close(fig)

    a = m.attrs
    lines = [f"# Final model: {r} components", "",
             f"- explained variance: {a['explained_variance_pct']:.3f} %",
             f"- core consistency: {a['core_consistency_pct']:.1f} %",
             f"- starts at best minimum: {a['n_init_at_best_minimum']}/{a['n_init']}",
             f"- converged: {bool(a['converged'])} ({a['n_iter']} iterations, tol {a['tol']})",
             f"- split-half (S4C6T3) min TCC: {sh.attrs['min_tcc']:.4f} -> "
             f"{'VALIDATED' if sh.attrs['validated'] else 'NOT validated'}",
             f"- samples: {m.sizes['sample']}; excluded: {cfg['exclude'] or 'none'}",
             f"- spectral correction: {a['spectral_correction']}"]
    (args.out / "final_model.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
