# Figure captions (draft)

Figures are produced by `scripts/figures.py` in `outputs/figures/` (PDF and
300 dpi PNG). R.U. = Raman units (Lawaetz & Stedmon 2009). Spectra are not
corrected for the instrument's spectral response, so R.U. and the indices are
internal to this fluorometer (Agilent Cary Eclipse).

**Figure 1.** Processing of one sample, Orcia June 2024, site 6, replicate 2
(the sample whose total Fmax is the median of the final model). (a) Raw EEM;
(b) after the absorbance-based inner-filter correction (Kothawala et al. 2013)
and subtraction of the paired blank; (c) in Raman units; (d) after removal of
first- and second-order Rayleigh and Raman scatter (grey) and of excitations
below 240 nm. In (a)-(c) the colour scale is capped at the maximum of the
region kept in (d); brighter scatter shares the darkest colour.

**Figure 2.** Corrected EEMs of one sample per river, each the sample with the
median total Fmax of its river. Common colour scale.

**Figure 3.** Choice of the number of PARAFAC components (199 samples; Orcia May
2024 sites 5-6 excluded). Explained variance, core consistency (Bro & Kiers
2003) and minimum Tucker congruence of the split-half validation (S4C6T3,
Murphy et al. 2013), with the 0.95 threshold (Lorenzo-Seva & ten Berge 2006).
The circled 3-component model was retained. The 7-component model passes the
split-half test but has a core consistency of -3639 %.

**Figure 4.** The three PARAFAC components. Top: component EEMs (outer product
of the loadings). Bottom: excitation (dashed) and emission (solid) loadings,
scaled to a maximum of 1; grey lines are the six split-half models.

**Figure 5.** Fmax of each component by sampling campaign and river: mean ± SD
over sites and replicates (lines), individual samples (points). The Ombrone
tributaries were sampled in different months (Merse April-July, Orcia May-August,
Farma September-November).

**Figure 6.** Fluorescence indices by campaign and river (mean ± SD, samples as
points): HIX (Ohno 2002), BIX (Huguet et al. 2009), FI (Cory & McKnight 2005).
Computed after Raman normalisation and before scatter removal; all 205 samples,
including Orcia May sites 5-6 (the BIX values up to 7).

**Figure S1.** Leverage and residuals of the 4-component model on all 205
samples: the six Orcia May site 5-6 samples have leverages of 0.12-0.26 and
define the fourth component.

**Figure S2.** Leverage and residuals of the final 3-component model. The
highest leverages belong to the three replicates of Merse July site 3, a
consistent group retained in the model.
