# Ombrone 2024: raw fluorescence and absorbance data

Raw instrument exports of river-water samples from the Ombrone basin
(Tuscany, Italy), 2024: fluorescence excitation-emission matrices (fDOM) and
absorbance spectra (CDOM). The data were described and analysed in
Polvani et al. (2026), *Environmental Pollution* 396, 127858,
[doi:10.1016/j.envpol.2026.127858](https://doi.org/10.1016/j.envpol.2026.127858).

## Files

`raw/` holds the 473 CSV files exactly as exported by the instruments, with
the original folder structure and file names (SHA-256 identical to the
originals). Folder names are those of the measurement sessions.

| folder | content |
|---|---|
| `raw/fDOM_ombrone/` | Agilent Cary Eclipse 3-D EEMs, "ASCII with log" CSV |
| `raw/CDOM ombrone/` | Agilent Cary 60 absorbance spectra, "ASCII with log" CSV |

Files named `BIANCO` are the Milli-Q blanks measured in the same session.
Binary instrument files (`.FDSW`, `.BSW`) are not included: the CSV files
contain the same spectra and the instrument settings in their log.

## Which file belongs to which sample

[`../config/pairing.csv`](../config/pairing.csv) lists one sample per row with
its EEM, EEM blank, absorbance and absorbance blank files, the river,
campaign (year-month), site and replicate, and whether it is used
(`include`). It was compiled and checked by hand; 205 of 227 rows are
included:

| river | sites | campaigns | samples |
|---|---|---|---|
| Farma | 6 | 2024-09 to 2024-11 | 52 |
| Merse | 6 | 2024-04 to 2024-07 | 69 |
| Orcia | 7 | 2024-05 to 2024-08 | 84 |

Each site and campaign was measured in triplicate (68 site-campaigns; two
have fewer replicates and one has four). The 22 excluded rows are explained in
the `status` and `notes` columns: the December Farma campaign has no blank
file, one absorbance measurement failed, two samples have no absorbance
spectrum and one has no EEM.

## Measurement settings

Read from the file logs.

| | Cary Eclipse (fluorescence) | Cary 60 (absorbance) |
|---|---|---|
| range | ex 200–450 nm, em 250–600 nm | 200–900 nm |
| step | ex 10 nm, em 2 nm | 1 nm |
| other | slits 5/5 nm, 1200 nm/min, 0.1 s, PMT medium, *Corrected spectra* OFF (2 samples: em step 1 nm, 600 nm/min) | dual beam, 600 nm/min, 3 cycles |

Cuvette path length: 10 mm for both instruments. No instrument spectral
correction was applied, so fluorescence indices are specific to this
instrument.

## License

The data are licensed under the Creative Commons Attribution 4.0
International license ([CC BY 4.0](LICENSE)). When you use them, cite
Polvani et al. (2026) and this dataset.
