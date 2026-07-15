# Compare two groups (A vs B)
#
# Per-ion comparison of two tagged groups (AUC + fold-change + test), then renders
# the most enriched ion as an image.
#
# Requires two groups: draw and name Regions on the Ion image tab, tag them
# Group A / B in the flow's Setup table (or pass named ROIs to compare()). The
# guard below logs a clear message and stops if the slide isn't set up for it.

if len(group_names()) < 2:
    log("Tag two groups (A and B) first — nothing to compare.")
else:
    find_peaks(snr=5)
    res = compare("Group A", "Group B", method="mwu")

    sig = res[res["q_value"] < 0.05].sort_values("AUC", ascending=False)
    table(sig, "Significant ions (q<0.05)")
    log(f"{len(sig)} of {len(res)} ions significant at q<0.05")

    if len(sig):
        image(float(sig.iloc[0]["mz"]), "Most enriched in A")
