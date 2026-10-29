"""Read the Ombrone dataset through eemkit using the curated pairing table.

Usage:
    python scripts/build_dataset.py --data-root "<...>/Progetto PARAFAC" \
        --pairing config/pairing.csv --out outputs/ombrone_raw.nc
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import eemkit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--pairing", default=Path("config/pairing.csv"), type=Path)
    ap.add_argument("--out", default=Path("outputs/ombrone_raw.nc"), type=Path)
    args = ap.parse_args()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", eemkit.EEMKitWarning)
        ds = eemkit.load_dataset(args.pairing, data_root=args.data_root)
    for w in caught:
        print(f"WARNING: {w.message}")

    print(ds)
    print("\nHistory:")
    for h in eemkit.get_history(ds):
        print(f"  {h['step']}: {json.dumps(h['params'])}")
    nan = np.isnan(ds["intensity"]).sum(("ex", "em")).to_series()
    print(f"\nSamples with NaN in intensity: {int((nan > 0).sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out, engine="scipy")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
