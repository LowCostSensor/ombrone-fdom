"""Apply the Phase 2 corrections to the Ombrone dataset, as set in config/processing.toml.

Usage:
    python scripts/correct.py --dataset outputs/ombrone_raw.nc \
        --config config/processing.toml --out outputs/ombrone_corrected.nc
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import xarray as xr

import eemkit
from eemkit import corrections as C


def path_length(cfg: dict, key: str) -> float:
    val = cfg["inner_filter"].get(key, "")
    if val in ("", None):
        raise SystemExit(f"config: inner_filter.{key} is empty; the cuvette path length must "
                         "be provided before the inner-filter correction can run")
    return float(val)


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def correct_until_raman(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Absorbance blank, inner filter, EEM blank and Raman normalisation."""
    l_abs = path_length(cfg, "absorbance_path_length_cm")
    l_fl = path_length(cfg, "fluorescence_path_length_cm")
    missing = cfg["missing"]["policy"]
    ds = C.subtract_absorbance_blank(ds, missing=missing)
    ds = C.correct_inner_filter(ds, absorbance_path_length_cm=l_abs,
                                fluorescence_path_length_cm=l_fl, missing=missing)
    ds = C.subtract_blank(ds, missing=missing)
    r = cfg["raman"]
    return C.normalize_raman(ds, ex=r["ex"], em_range=tuple(r["em_range"]),
                             baseline=r["baseline"], missing=missing)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=Path("outputs/ombrone_raw.nc"), type=Path)
    ap.add_argument("--config", default=Path("config/processing.toml"), type=Path)
    ap.add_argument("--out", default=Path("outputs/ombrone_corrected.nc"), type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = correct_until_raman(xr.load_dataset(args.dataset, engine="scipy"), cfg)
    s = cfg["scatter"]
    widths = {b: tuple(s[b]) for b in C.scatter.BANDS if b in s}
    ds = C.remove_scatter(ds, widths=widths, raman_shift_cm1=s["raman_shift_cm1"],
                          zero_below_rayleigh=s["zero_below_rayleigh"])
    if s["interpolate"]:
        ds = C.interpolate_missing(ds, method=s["interpolate"])
    ds = C.crop(ds, ex=(cfg["crop"]["ex_min"], None))

    status = C.spectral_correction_status(ds)
    print(f"Spectral correction: {status}")
    if status == "none":
        print("  -> intensities are NOT spectrally corrected; Raman units are internal to "
              "this instrument and not comparable with other studies")
    print(f"Samples: {ds.sizes['sample']}, ex {ds.ex.min().item():.0f}-{ds.ex.max().item():.0f}"
          f", em {ds.em.min().item():.0f}-{ds.em.max().item():.0f}")
    print(f"IFE factor: max {float(ds['ife_factor'].max()):.3f}")
    for h in eemkit.get_history(ds):
        print(f"  {h['step']}: {json.dumps(h['params'])}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out, engine="scipy")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
