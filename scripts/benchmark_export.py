"""Export the Ombrone data at each processing stage for the staRdom benchmark.

Writes plain CSV tables that scripts/benchmark_stardom.R reads into staRdom
eemlists, so both toolboxes start from exactly the same numbers:

* ``raw_eem.csv``, ``blank_eem.csv``: EEMs as read (sample, em, one column per ex);
* ``absorbance.csv``: absorbance after the absorbance-blank subtraction
  (wavelength, one column per sample), the input of the inner-filter step;
* ``eemkit_after_raman.csv``: eemkit EEMs after IFE, blank and Raman units;
* ``eemkit_corrected.csv``: eemkit EEMs after scatter removal and crop, for the
  samples in the final PARAFAC model (removed scatter as NA);
* ``eemkit_raman_area.csv``: the Raman areas eemkit divided by.

Usage:
    python scripts/benchmark_export.py --out outputs/benchmark/input
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from correct import correct_until_raman, load_config

from eemkit import corrections as C


def eem_table(da: xr.DataArray) -> pd.DataFrame:
    """(sample, ex, em) -> rows (sample, em), one column per excitation."""
    da = da.transpose("sample", "em", "ex")
    n_s, n_em, n_ex = da.shape
    df = pd.DataFrame(da.values.reshape(n_s * n_em, n_ex),
                      columns=[f"{x:g}" for x in da["ex"].values])
    df.insert(0, "em", np.tile(da["em"].values, n_s))
    df.insert(0, "sample", np.repeat(da["sample"].values, n_em))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=Path("outputs/ombrone_raw.nc"), type=Path)
    ap.add_argument("--corrected", default=Path("outputs/ombrone_corrected.nc"), type=Path)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/benchmark/input"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    raw = xr.load_dataset(args.raw, engine="scipy")
    eem_table(raw["intensity"]).to_csv(args.out / "raw_eem.csv", index=False)
    eem_table(raw["blank_intensity"]).to_csv(args.out / "blank_eem.csv", index=False)
    absb = C.subtract_absorbance_blank(raw, missing=cfg["missing"]["policy"])["absorbance"]
    absb.to_pandas().T.rename_axis("wavelength").to_csv(args.out / "absorbance.csv")

    after = correct_until_raman(raw, cfg)
    eem_table(after["intensity"]).to_csv(args.out / "eemkit_after_raman.csv", index=False)
    after["raman_area"].to_series().to_csv(args.out / "eemkit_raman_area.csv")

    cor = xr.load_dataset(args.corrected, engine="scipy")
    exclude = cfg["parafac"]["exclude"]
    if exclude:
        cor = cor.drop_sel(sample=exclude)
    eem_table(cor["intensity"]).to_csv(args.out / "eemkit_corrected.csv", index=False,
                                       na_rep="NA")
    print(f"raw/blank: {raw.sizes['sample']} samples; corrected: {cor.sizes['sample']} "
          f"samples -> {args.out}")


if __name__ == "__main__":
    main()
