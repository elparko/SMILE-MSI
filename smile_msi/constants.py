"""Engine-wide numeric constants with one documented home.

Centralizes the m/z extraction tolerances that were previously bare literals
scattered across the extraction, spatial, multivariate, single-cell, imaging and
quantification modules (audit plan 18, Issue B). Naming them removes the "magic
number" problem and makes the *intentional* difference between the broad
extraction window and the tighter quantification window explicit in one place.
"""
from __future__ import annotations

#: Default m/z extraction window (ppm) for ion-image / feature-matrix integration.
#: Broad enough to keep a single ion from splitting across pixels; used by every
#: spatial / multivariate / single-cell / imaging extraction entry point.
DEFAULT_TOL_PPM = 50.0

#: Tighter window (ppm) for absolute quantification. Minimizes co-integration of
#: interferents near the analyte / internal standard (ICH-style quant practice);
#: intentionally narrower than :data:`DEFAULT_TOL_PPM`. Keeping the two distinct is
#: a deliberate scientific choice — calibration curves and concentration
#: back-calculation are sensitive to neighbouring-ion contamination, while the
#: broad spatial window favours not splitting one ion across pixels.
QUANT_TOL_PPM = 10.0

#: Clustering width (ppm) that pools each sample's picked peaks into one shared cohort
#: feature axis (:func:`smile_msi.cohort.consensus_targets`). A cohort feature is born
#: from peaks within this window across slides, so any downstream extraction of a
#: consensus feature should use a window **at least this wide**: reading a feature that
#: was clustered at this tolerance with a narrower window can fall between two slides'
#: apexes and integrate baseline (the "blank ion image / between two peaks" failure).
CONSENSUS_TOL_PPM = 20.0
