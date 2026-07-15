# SMILE MSI — Analysis Workflow Scripting Guide

You are writing a short **Python script** that analyses one loaded mass-spectrometry imaging (MSI) slide and surfaces results. It runs inside SMILE MSI's Script Console (Data ▸ Analysis script…) against the slide already open in the app.

> **This file is auto-generated.** It is the checked-in mirror of the app's built-in **AI guide**, emitted by `smile_msi.scripting.capabilities_doc()`. To regenerate it, open the Script Console (Data ▸ Analysis script…), click **AI guide**, then **Save as Markdown…** — or call `scripting.capabilities_doc(api)` directly. Regenerate it whenever the `ScriptAPI` surface or the `EXAMPLES` change, so the reference stays in sync.

## Execution model
- The script is ordinary Python 3. `numpy` is available as `np`. No imports are needed for the analysis functions — they are pre-bound bare names.
- It runs **against the open slide**, exposed as `ds` (an `MSIDataset`). You do not load data; it is already there.
- There is no value to `return`. You surface results with the output helpers below; anything you `print()` is captured into the run log too.
- Runs may take seconds (peak picking, segmentation, multivariate). That is normal.
- The script is sandbox-free (full Python on the user's machine) but should avoid network access and writing files — use the console's *Export* for artifacts.

## Output helpers (how results become visible)
- `log(*args)` — add a line to the run log.
- `table(df, title='')` — show a pandas DataFrame (or list-of-dicts) as a table.
- `image(x, title='')` — show an ion image. `x` is an m/z (float), a per-pixel vector, or a 2-D array.
- `record(name, value)` — stash a named value to inspect after the run.

## Core concepts
- **Features** = the working set of m/z the analyses operate on. Produce it with `find_peaks()` / `find_spatial_features()` (they set it as a side effect), or pass `features=[...]` explicitly. Most steps default to the working set.
- **Regions** = named ROIs drawn in the app; resolve a mask with `region('name')`.
- **Groups** = ROIs tagged Group A/B/… for comparisons; `group('A')` or pass names to `compare` / `discriminating` / `markers`. With no groups tagged, group-based steps fall back to the latest `segment()` clusters.
- **ppm / norm / reduce** are the extraction defaults; set `api.ppm = 5`, `api.norm = 'rms'` etc. once at the top to change them for the whole script.

## Analysis functions (bare names)
- `mean_spectrum(mask=None)` — ``(mz_axis, intensities)`` mean spectrum over the whole slide or a region/mask.
- `ion_image(mz)` — 2-D ion image (H×W array) for one m/z at the current ppm / reduce / norm.
- `ion_vector(mz)` — Per-pixel intensity vector (length n_pixels) for one m/z.
- `feature_matrix(features=None)` — The pixels×features intensity matrix for the working (or given) feature set.
- `find_peaks(snr: float = 3.0, min_rel_intensity: float = 0.0, max_peaks: int = 500, prominence: float = 1.0, mask=None)` — Detect peaks in the mean spectrum → set + return the working feature set.
- `find_spatial_features(snr: float = 3.0, min_rel_intensity: float = 0.0, min_frequency: float = 0.0, min_morans: float = 0.0, mask=None)` — Spatially-aware peak detection (S/N → reproducibility → Moran's I) → working set.
- `set_features(features)` — Set the working feature set later steps default to (m/z floats or peak dicts).
- `get_features()` — The current working feature set (list of m/z floats).
- `segment(features=None, n_clusters: int = 0, spatial: bool = True, spatial_sigma: float = 1.0)` — Cluster pixels into regions (``n_clusters=0`` → auto by silhouette). Sets
- `compare(a, b, features=None, method: str = 'mwu', samples=None)` — Per-ion comparison of two groups A vs B (AUC + signed log2 fold-change + test) → DataFrame.
- `discriminating(features=None, groups=None, top_n: int = 15)` — Top ions enriched in each group vs the rest (one-vs-rest AUC + FDR) → {group: df}.
- `multigroup(features=None, groups=None, method: str = 'kruskal')` — Per-ion test across 3+ groups (Kruskal-Wallis or ANOVA) → DataFrame.
- `markers(features=None, groups=None, shrink: float = 2.0, top_n: int = 15)` — Nearest shrunken centroids — automatic marker ions per group → {group: df}.
- `roi_localization(region, features=None)` — How confined each ion is to one region vs the rest of the tissue → DataFrame.
- `region_membership(groups=None, features=None, min_prevalence: float = 0.5)` — Partition ions by which groups they're present in (Venn compartments) → dict.
- `colocalize(target_mz=None, features=None, method: str = 'pearson', mask=None)` — Rank all features by spatial similarity to a target m/z (defaults to active ion).
- `coloc_modules(features=None, n_modules: int = 0, threshold: float = 0.5, method: str = 'pearson', mask=None)` — Group features into co-localized modules (``n_modules=0`` → auto).
- `pca(features=None, n_components: int = 5)` — Principal-component score images + loadings.
- `nmf(features=None, n_components: int = 5)` — Non-negative matrix factorization (additive parts) score images + loadings.
- `plsda(features=None, groups=None, n_components: int = 2, orthogonal: bool = False)` — Supervised PLS-DA / OPLS-DA classification; ranks ions by VIP.
- `embedding(features=None, method: str = 'umap')` — 2-D pixel embedding (UMAP / t-SNE) + molecular-similarity tissue map.
- `annotate(features=None, mode: str = 'negative', match_ppm: float = 10.0, image_ppm: float = 10.0)` — Match the feature set to the lipid database (isotopes + adducts + confidence) → DataFrame.
- `run(step_id, features=None, groups=None, mask=None, target_mz=None, **params)` — Run any registered analysis by id (see the guide's step table) with raw params —
- `region(name)` — The boolean pixel mask of a named region/ROI.
- `group(name)` — The boolean pixel mask of a named group (the union of its tagged ROIs).
- `region_names()` — Names of the regions/ROIs available to this script.
- `group_names()` — Names of the groups (A/B/…) tagged on this slide's ROIs.

## All analyses via `run(step_id, ...)`
Every analysis is also reachable by id through `run()` (forward-compatible escape hatch). `run()` resolves `features` / `groups` / `mask` / `target_mz` for you:

**Peaks**
- `find_spatial_features` — Spatially-aware peak detection (S/N → reproducibility → Moran's I denoise). Produces the working feature set later steps use. _(needs: —; params: snr, min_rel_intensity, min_frequency, min_morans, tol_ppm, norm)_
- `find_peaks` — Detect peaks in the mean spectrum (S/N + prominence). Produces the working feature set. _(needs: —; params: snr, min_rel_intensity, prominence, max_peaks)_

**Segmentation**
- `auto_segment` — Cluster pixels into regions. 0 clusters = auto-pick by silhouette. _(needs: feature_set; params: n_clusters, spatial, spatial_sigma, tol_ppm, norm)_

**Statistics**
- `roi_comparison` — Per-ion comparison of two groups (AUC + signed log2 fold-change + test). Each named sub-ROI is a replicate. _(needs: ab, feature_set; params: method, tol_ppm, norm)_
- `discriminating_features` — Top ions enriched in each group vs the rest (one-vs-rest AUC + FDR). _(needs: feature_set, groups; params: top_n, tol_ppm, norm)_
- `multigroup_features` — Per-ion test across 3+ groups (Kruskal-Wallis or ANOVA). _(needs: feature_set, groups; params: method, tol_ppm, norm)_
- `roi_localization` — How confined each ion is to the first group's ROI vs the rest of the tissue (one group is enough). _(needs: feature_set, region; params: tol_ppm, norm)_
- `shrunken_centroids` — Nearest shrunken centroids — automatic marker selection per group. _(needs: feature_set, groups; params: shrink, top_n, tol_ppm, norm)_
- `region_membership` — Partition ions by which groups they are present in (Venn compartments). _(needs: feature_set, groups; params: min_prevalence, tol_ppm, norm)_

**Co-localization**
- `colocalize` — Rank all ions by spatial similarity to a target m/z. _(needs: feature_set, target_mz; params: target_mz, method, tol_ppm, norm)_
- `coloc_modules` — Group ions into co-localized modules by hierarchical clustering. _(needs: feature_set; params: n_modules, threshold, method, tol_ppm, norm)_

**Multivariate**
- `pca` — Principal-component score images + loadings. _(needs: feature_set; params: n_components, tol_ppm, norm)_
- `nmf` — Non-negative matrix factorization (additive parts) score images + loadings. _(needs: feature_set; params: n_components, tol_ppm, norm)_
- `plsda` — Supervised classification; ranks ions by VIP (variable importance). _(needs: feature_set, groups; params: n_components, orthogonal, tol_ppm, norm)_
- `embedding` — 2-D pixel embedding + molecular-similarity tissue map. _(needs: feature_set; params: method, tol_ppm, norm)_

**Annotation**
- `annotate` — Match the feature set to the lipid database (isotopes + adducts + confidence). _(needs: feature_set; params: mode, match_ppm, image_ppm, norm)_

## The session handle: `s.run_analysis(step_id, **params)`
The loaded session is also bound as `s` (an alias of `api`). `s.run_analysis(step_id, **params)` is the documented public form of `run()`: it dispatches through the **same** registry with the registry defaults backfilled, and assembles the step's inputs (`features` / groups / region masks / `target_mz`) from the session's current state exactly as the named wrappers do. Override any of them with `features=`, `mask=`, `a=`/`b=` (the two comparison groups; default = the first two tagged groups), `region=`, `groups=` or `target_mz=`; every other keyword is a tunable engine param. This is the call a migrated Flow preset is written in, so a saved recipe reproduces a Flow losslessly:
```python
s.run_analysis("find_spatial_features", snr=3.0, min_morans=0.05)
s.run_analysis("auto_segment", n_clusters=2)
s.run_analysis("roi_comparison", method="mwu")   # A/B = first two tagged groups
```

## Best practices
1. Start by establishing features: `find_peaks(snr=5)` (or `find_spatial_features`).
2. Prefer the named functions; reach for `run(id, ...)` only for steps without one.
3. Surface *something* — at least one `table`, `image`, `log`, or `record` — so the run isn't silent.
4. Guard group/region steps: check `group_names()` / `region_names()` first and `log()` a clear message if the slide isn't set up for them.
5. Keep it deterministic and reasonably small; this is interactive, not a batch job.
6. Errors are reported with your line numbers — read the traceback and fix in place.

## Example
```python
# Requires two tagged groups (or two named ROIs). Lists ions that differ,
# then renders the most discriminating one as an image.
find_peaks(snr=5)
res = compare("Group A", "Group B", method="mwu")
sig = res[res["q_value"] < 0.05].sort_values("AUC", ascending=False)
table(sig, "Significant ions (q<0.05)")
if len(sig):
    image(float(sig.iloc[0]["mz"]), "Most enriched in A")
```
