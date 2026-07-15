# Co-localization with an ion
#
# Rank every feature by spatial similarity to a target m/z, then map the top hit
# next to the target so you can eyeball the overlap.
#
# `active_mz` is the ion currently selected in the app (None if none is selected);
# the fallback m/z keeps the script runnable on any slide — change it to your ion.

TARGET = active_mz or 888.62

find_peaks(snr=5)
ranked = colocalize(target_mz=TARGET, method="pearson")

table(ranked[:25], f"Most co-localized with m/z {TARGET:.4f}")
image(TARGET, f"Target · m/z {TARGET:.4f}")
if len(ranked) > 1:
    top = ranked[1]                  # ranked[0] is the target itself (score 1.0)
    image(float(top["mz"]), f"Top co-localized · m/z {top['mz']:.4f} (r={top['score']:.2f})")
