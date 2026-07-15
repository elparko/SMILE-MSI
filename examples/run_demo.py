"""Demo: run the full pipeline on the synthetic sample and write a report.

    python examples/run_demo.py

Point `--input` of the CLI (or this path) at your own peak export to use real data.
Group names are derived from the CSV column headers (e.g. ``Control_mean`` -> ``Control``).
"""
import os

from smile_msi import pipeline

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "sample_peaks.csv")

df = pipeline.read_table(SAMPLE)
# Auto-detect the m/z and per-group mean/SD columns, and name the groups from the headers.
g = pipeline.guess_columns(df.columns)
a_label = pipeline.label_from_column(g["a_mean"], "Group A")
b_label = pipeline.label_from_column(g["b_mean"], "Group B")

res = pipeline.annotate_and_score(
    df, mz_col=g["mz"],
    a_mean=g["a_mean"], a_sd=g["a_sd"],
    b_mean=g["b_mean"], b_sd=g["b_sd"],
    mode="negative", ppm=5.0,
)
out = os.path.join(HERE, "demo_report.xlsx")
pipeline.build_report(res, out, a_label=a_label, b_label=b_label, mode="negative")
print(res[["mz", "AUC", "best_lipid", "best_adduct", "best_ppm"]].to_string(index=False))
print(f"\n{a_label} vs {b_label} — wrote {out}")
