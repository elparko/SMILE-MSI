# Find peaks & annotate
#
# Detect the working feature set on the mean spectrum, match it against the
# in-silico lipid database, and surface the confident hits as a table.
#
# Paste into the Script Console (Data ▸ Analysis script…, ⇧⌘J) and ▶ Run, or
# load it via Workflows ▸ Open file…. The analysis names below (find_peaks,
# annotate, …) are pre-bound bare functions — no imports needed; `np` is numpy.

peaks = find_peaks(snr=5)
log(f"{len(peaks)} peaks")

hits = annotate(mode="negative", match_ppm=10)
named = hits[hits["lipid"].astype(str) != ""].copy()
named["confidence_score"] = np.asarray(named["confidence_score"], dtype=float)

table(named.sort_values("confidence_score", ascending=False).head(25), "Top lipid IDs")
log(f"{len(named)} of {len(hits)} features annotated")
