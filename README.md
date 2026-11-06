# ombrone-fdom

Raw data and a fully reproducible analysis of dissolved organic matter
fluorescence (fDOM) and absorbance (CDOM) in three rivers of the Ombrone basin
(Farma, Merse, Orcia; Tuscany, Italy), sampled from April to November 2024.
The data were first described in:

> Polvani A., Nardin R., Liu X., Di Grazia F., Boldrini A., Cirrone R. G.,
> Galgani L., Tamasi G., Loiselle S. A. (2026). Disentangling spatial and
> seasonal controls on heavy metal pollution in Mediterranean rivers through
> fluorescence DOM. *Environmental Pollution* 396, 127858.
> [doi:10.1016/j.envpol.2026.127858](https://doi.org/10.1016/j.envpol.2026.127858)

The analysis here uses [eemkit](https://github.com/LowCostSensor/eemkit) and
also serves as its validation against the R package staRdom.

## Contents

| path | content |
|---|---|
| [`data/`](data/README.md) | 473 raw CSV files from the Cary Eclipse and Cary 60, unchanged except for anonymised operator names (CC BY 4.0) |
| `config/pairing.csv` | which EEM, blank and absorbance file belong to each sample, checked by hand |
| `config/processing.toml` | every processing parameter, each with its reason and source |
| `config/ombrone_project.toml` | the same settings as an eemkit project, for the desktop application |
| `scripts/` | the analysis, step by step (below) |
| `docs/` | figure captions and the desktop application guide (Italian and English) |

## Results in brief

* Corrections: absorbance inner filter (10 mm cuvettes), Milli-Q blank
  subtraction, Raman units (ex 350, em 381–426 nm), scatter removed as missing
  values, excitation ≥ 240 nm. No instrument spectral correction.
* PARAFAC on 199 samples (six Orcia samples of May 2024, sites 5 and 6, left
  out as outliers): three components, explained variance 99.02 %, core
  consistency 85.5 %, split-half validated (minimum Tucker congruence 0.9915).

| component | excitation max (nm) | emission max (nm) | interpretation |
|---|---|---|---|
| C1 | 240 (secondary ~320) | 408 | humic-like (peaks A + M) |
| C2 | 250 (secondary ~360) | 464 | terrestrial humic-like (peaks A + C) |
| C3 | 250 | 340 | tryptophan-like (peak T) |

* Benchmark against staRdom 1.1.32: corrected EEMs agree to a median relative
  difference of 2.2 × 10⁻⁴, indices to machine precision (HIX 3 × 10⁻⁴), and
  the PARAFAC loadings with Tucker congruence ≥ 0.999.

## Reproduce

Create the environment (conda-forge; it installs eemkit from GitHub):

```bash
conda env create -f environment.yml
conda activate fdom-eem
```

Then run the steps in order; every script writes to `outputs/` (not under
version control):

```bash
python scripts/inventory.py             # scan of data/raw: files, metadata, QC
python scripts/build_dataset.py         # data/raw + config/pairing.csv -> outputs/ombrone_raw.nc
python scripts/phase2_diagnostics.py    # blanks, Raman band, scatter widths, noise
python scripts/correct.py               # corrections -> outputs/ombrone_corrected.nc
python scripts/indices.py               # FI, HIX, BIX, Coble peaks
python scripts/parafac_explore.py       # 2-7 components, split-half validation
python scripts/parafac_final.py         # final 3-component model (20 starts)
python scripts/figures.py               # figures for the paper
```

The PARAFAC steps take from minutes to about an hour, depending on the
computer. The benchmark against staRdom needs R with staRdom 1.1.32:

```bash
python scripts/benchmark_export.py
Rscript scripts/benchmark_stardom.R corrections outputs/benchmark/input outputs/benchmark/stardom
Rscript scripts/benchmark_stardom.R parafac outputs/benchmark/input outputs/benchmark/stardom 3 20 20000 4
python scripts/benchmark_compare.py
```

The PARAFAC fit can also be compared with the N-way toolbox, the engine of
drEEM (MATLAB). After `benchmark_export.py`, in MATLAB, from this folder,
with drEEM (or the N-way toolbox) on the path:

```matlab
addpath scripts
benchmark_nway('outputs/benchmark/input', 'outputs/benchmark/nway', 3, 20, 20000)
```

then:

```bash
python scripts/benchmark_compare.py --engine nway
```

Or open `config/ombrone_project.toml` in the eemkit desktop application
(`eemkit-gui`) and follow the [guide](docs/guida_app/app_guide_eemkit_en.html).

## License and citation

* Data (`data/`): [CC BY 4.0](data/LICENSE).
* Code and documentation: [GNU GPL v3.0 or later](LICENSE).

Cite Polvani et al. (2026) for the data, and this repository (see
[CITATION.cff](CITATION.cff)) for the analysis.
