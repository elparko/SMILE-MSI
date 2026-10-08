# SMILE MSI — Analysis Workflow Scripting Guide

You are writing a short **Python script** that analyses one loaded mass-spectrometry imaging (MSI) slide and surfaces results. It runs inside SMILE MSI's Script Console (Data ▸ Analysis script…) against the slide already open in the app.

> **This file is auto-generated.** It is the checked-in mirror of the app's built-in **AI guide**, emitted by `smile_msi.scripting.capabilities_doc()`. To regenerate it, open the Script Console (Data ▸ Analysis script…), click **AI guide**, then **Save as Markdown…** — or call `scripting.capabilities_doc(api)` directly. Regenerate it whenever the `ScriptAPI` surface or the `EXAMPLES` change, so the reference stays in sync.

## Execution model
- The script is ordinary Python 3. `numpy` is available as `np`. No imports are needed for the analysis functions — they are pre-bound bare names.
- It runs **against the open slide**, exposed as `ds` (an `MSIDataset`). You do not load data; it is already there.
- There is no value to `return`. You surface results with the output helpers below. `print()` output is captured separately from the run log, as the run's `stdout` (the Script Console shows it under *stdout*; the MCP `run_script` result returns it as `stdout`). Use `log()` for lines that belong in the run log.
- Runs may take seconds (peak picking, segmentation, multivariate). That is normal.
- The script is sandbox-free (full Python on the user's machine) but should avoid network access and writing files — use the console's *Export* for artifacts.

## Output helpers (how results become visible)
- `log(*args)` — add a line to the run log.
- `table(df, title='')` — show a pandas DataFrame (or list-of-dicts) as a table.
- `image(x, title='')` — show an ion image. `x` is an m/z (float), a per-pixel vector, or a 2-D array.
- `record(name, value)` — stash a named value to inspect after the run.
- `ds.ratio_image(num_mz, den_mz, tol_ppm=…, norm='none', eps=1.0)` — an H×W image of one ion over another (e.g. sulfatide/PC); show it with `image(...)`. It takes its own `tol_ppm` (pass `api.ppm` to match the session) and `eps` keeps a ~0 denominator finite.

## Core concepts
- **Features** = the working set of m/z the analyses operate on. Produce it with `find_peaks()` / `find_spatial_features()` (they set it as a side effect), or pass `features=[...]` explicitly. Most steps default to the working set.
- **Regions** = named ROIs drawn in the app; resolve a mask with `region('name')`.
- **Groups** = ROIs tagged Group A/B/… for comparisons; `group('A')` or pass names to `compare` / `discriminating` / `markers`. With no groups tagged, group-based steps fall back to the latest `segment()` clusters.
- **ppm / norm / reduce** are the extraction defaults; set `api.ppm = 5`, `api.norm = 'rms'` etc. once at the top to change them for the whole script.
- **Regions by construction** — build masks from the signal instead of drawing: `threshold_mask(composite([...m/z...]), 60)`, `ring(mask, width_px=6, mode='outer')` (a collar outside it), `invert(mask)` (everything else). `add_region('name', mask)` stages a region the console can push into the app (Apply to app ▸ Add regions to the slide) and makes it usable by name.
- **`threshold_mask` fills holes by default** (`fill_holes=True`): it is built for solid compartments, so a **rim or ring** signal comes back as a solid disc covering the core it encloses — pass `fill_holes=False` for one (the run log warns when filling grew the mask a lot). The cut is a **percentile of every pixel with signal > 0**, and off-tissue pixels with any noise count, so `60` is not the 60th percentile of the tissue; for that, threshold `values * tissue_mask`, or pass `percentile=False` and an absolute intensity.
- **Segmentation clusters every acquired pixel** unless you pass `mask=`: with off-tissue background on the slide, `segment(n_clusters=2)` usually splits tissue from background. Use `segment(mask=tissue)` to cluster the tissue alone.

## Analysis functions (bare names)
- `mean_spectrum(mask=None)` — ``(mz_axis, intensities)`` mean spectrum over the whole slide or a region/mask.
- `ion_image(mz)` — 2-D ion image (H×W array) for one m/z at the current ppm / reduce / norm.
- `ion_vector(mz)` — Per-pixel intensity vector (length n_pixels) for one m/z.
- `feature_matrix(features=None)` — The pixels×features intensity matrix for the working (or given) feature set.
- `find_peaks(snr: float = 3.0, min_rel_intensity: float = 0.0, max_peaks: int = 0, prominence: float = 1.0, mask=None)` — Detect peaks in the mean spectrum → set + return the working feature set.
- `find_spatial_features(snr: float = 3.0, min_rel_intensity: float = 0.0, min_frequency: float = 0.0, min_morans: float = 0.0, mask=None, max_candidates: int = 2000)` — Spatially-aware peak detection (S/N → reproducibility → Moran's I) → working set.
- `set_features(features)` — Set the working feature set later steps default to (m/z floats or peak dicts).
- `get_features()` — The current working feature set (list of m/z floats).
- `segment(features=None, n_clusters: int = 0, spatial: bool = True, spatial_sigma: float = 1.0, mask=None)` — Cluster pixels into regions (``n_clusters=0`` → auto by silhouette). Sets
- `compare(a, b, features=None, method: str = 'mwu', samples=None)` — Per-ion comparison of two groups A vs B (AUC + signed log2_fc + test) → DataFrame.
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
- `annotate(features=None, mode: str = 'negative', match_ppm: float = 5.0, image_ppm: float = 10.0)` — Match the feature set to the lipid database (isotopes + adducts + confidence) → DataFrame.
- `run(step_id, features=None, groups=None, mask=None, target_mz=None, **params)` — Run any registered analysis by id (see the guide's step table) with raw params —
- `region(name)` — The boolean pixel mask of a named region/ROI.
- `group(name)` — The boolean pixel mask of a named group (the union of its tagged ROIs).
- `region_names()` — Names of the regions/ROIs available to this script.
- `group_names()` — Names of the groups (A/B/…) tagged on this slide's ROIs.
- `composite(features, weight: str = 'raw')` — Per-pixel summed intensity of several m/z (a lipid class, the sulfatides, …).
- `threshold_mask(values, cut, percentile: bool = True, fill_holes: bool = True, min_pixels: int = 0)` — Mask of pixels where a signal (per-pixel vector, or one m/z) reaches ``cut``.
- `ring(mask, width_px=None, width_um=None, mode: str = 'outer')` — A rim (inner) / collar (outer) / band of the given width around a mask's boundary.
- `invert(mask)` — Every acquired pixel NOT in ``mask`` (the 'everything else' region).
- `add_region(name, mask, color=None)` — Stage a named region from a mask (usable by name from here on; pushable into the app).

## All analyses via `run(step_id, ...)`
Every analysis is also reachable by id through `run()` (forward-compatible escape hatch). `run()` resolves `features` / `groups` / `mask` / `target_mz` for you:

**Peaks**
- `find_spatial_features` — Spatially-aware peak detection (S/N → reproducibility → Moran's I denoise). Produces the working feature set later steps use. _(needs: —; params: snr, min_rel_intensity, min_frequency, min_morans, max_candidates, tol_ppm, norm)_
- `find_coherent_features` — Spatial feature detection (S/N → reproducibility → Moran's I) plus an artifact-rejection quality gate (spatial-chaos morphology + hotspot concentration) that screens out delocalization / matrix-crystal ions which pass autocorrelation but aren't real. Produces a working feature set. _(needs: —; params: snr, min_rel_intensity, min_frequency, min_morans, min_quality, max_hotspot, max_candidates, tol_ppm, norm)_
- `find_peaks` — Detect peaks in the mean spectrum (S/N + prominence). Produces the working feature set. _(needs: —; params: snr, min_rel_intensity, prominence, max_peaks)_

**Segmentation**
- `auto_segment` — Cluster pixels into regions. 0 clusters = auto-pick by silhouette. _(needs: feature_set; params: n_clusters, spatial, spatial_sigma, tol_ppm, norm)_

**Statistics**
- `roi_comparison` — Per-ion comparison of two groups (AUC + fold-change + test). Each named sub-ROI is a replicate. _(needs: ab, feature_set; params: method, save_lists, max_q, tol_ppm, norm)_
- `region_comparison` — Overlay two regions' whole mean spectra (or their difference) — the visual companion to Region comparison (A vs B). Toggle overlay ⇄ difference in the result. _(needs: ab; params: norm)_
- `class_comparison` — Roll the per-ion identifications up to whole lipid classes (PE, PC, SM, …) and compare two groups: each class's summed signal gets the same exact rank-based AUC + fold-change + test as the per-ion Region comparison. Identification mode/tolerance set how peaks map to classes. _(needs: ab, feature_set; params: method, mode, id_ppm, tol_ppm, norm)_
- `class_composition` — Each group's lipid-class composition: the share (%) of annotated-lipid signal in every class, for group A and group B side by side (each column sums to 100%). Identification mode/tolerance set how peaks map to classes. _(needs: ab, feature_set; params: mode, id_ppm, tol_ppm, norm)_
- `marker_panel` — Sweep several region pairs (each sub-region vs its parent by default), keep every pair's AUC-distinctive ions, and merge them into ONE feature list for downstream analyses. Click 'Auto-pair sub→parent' to pair each sub-ROI with its parent region, or 'Add pair…' to build A-vs-B pairs by hand. Per-pixel AUC effect size for candidate selection — follow with a Region comparison / cohort step for replicate-level p-values. _(needs: ab_multi, feature_set; params: direction, threshold, max_q, subtract_a_from_b, save_each, tol_ppm, norm)_
- `discriminating_features` — Characteristic ions for each group vs the rest (one-vs-rest AUC + FDR). With exactly two tagged groups this is A-vs-B. Keep only the enriched, significant markers, save them as a per-group ★ list, and pass the survivors to the next step (e.g. a Venn). Ungrouped/background pixels are excluded from 'the rest'. _(needs: feature_set, groups; params: enriched_only, min_auc, max_q, top_n, save_lists, tol_ppm, norm)_
- `multigroup_features` — Per-ion test across 3+ groups (Kruskal-Wallis or ANOVA). _(needs: feature_set, groups; params: method, tol_ppm, norm)_
- `roi_localization` — How confined each ion is to the first group's ROI vs the rest of the tissue (one group is enough). _(needs: feature_set, region; params: tol_ppm, norm)_
- `filter_auc` — Trim the previous comparison's ions by ROC AUC and make the survivors the working feature set. AUC 0.5 = no discrimination; >0.5 = higher in group B; <0.5 = higher in group A (more confined to the first ROI). The exported table keeps every ion with a 'kept' flag, so you get both the untrimmed and trimmed lists. Add after a Region comparison (A vs B) or ROI localization step. _(needs: stats; params: direction, threshold)_
- `shrunken_centroids` — Nearest shrunken centroids — automatic marker selection per group. _(needs: feature_set, groups; params: shrink, top_n, tol_ppm, norm)_
- `region_membership` — Partition ions by which groups they are present in (Venn compartments). _(needs: feature_set, groups; params: min_prevalence, tol_ppm, norm)_

**Co-localization**
- `colocalize` — Rank all ions by spatial similarity to a target m/z. _(needs: feature_set, target_mz; params: target_mz, method, tol_ppm, norm)_
- `coloc_modules` — Group ions into co-localized modules by hierarchical clustering. _(needs: feature_set; params: n_modules, threshold, method, tol_ppm, norm)_
- `region_correlation` — Molecular-fingerprint similarity of every named region to every other — a region×region heatmap plus each region's closest match (the old Region-match view). _(needs: feature_set; params: method, tol_ppm, norm)_

**Multivariate**
- `dgmm` — Spatially-aware Gaussian-mixture segmentation of a single ion image into intensity zones (background → bright). _(needs: target_mz; params: k, norm)_
- `pca` — Principal-component score images + loadings. _(needs: feature_set; params: n_components, tol_ppm, norm)_
- `nmf` — Non-negative matrix factorization (additive parts) score images + loadings. _(needs: feature_set; params: n_components, tol_ppm, norm)_
- `plsda` — Supervised classification; ranks ions by VIP (variable importance). _(needs: feature_set, groups; params: n_components, orthogonal, tol_ppm, norm)_
- `classify_cv` — Cross-validated accuracy + confusion matrix for the PLS-DA classifier. Uses leave-one-tissue-out folds when the slide has ≥2 detected tissue pieces, else stratified k-fold (the engine flags an optimistic, leakage-prone estimate). _(needs: feature_set, groups; params: n_components, orthogonal, n_folds, tol_ppm, norm)_
- `classify_map` — Fit the PLS-DA classifier, then paint every pixel with its predicted class — a class-label map over the whole slide. _(needs: feature_set, groups; params: n_components, orthogonal, tol_ppm, norm)_
- `shap_biomarkers` — Per-region biomarker ions ranked by SHAP importance (RandomForest + TreeExplainer); each ion's direction is the Spearman sign of intensity vs its SHAP value. Needs the optional 'shap' package. _(needs: feature_set, groups; params: n_estimators, max_pixels, tol_ppm, norm)_
- `embedding` — 2-D pixel embedding + molecular-similarity tissue map. _(needs: feature_set; params: method, tol_ppm, norm)_

**Annotation**
- `annotate` — Match the feature set to the lipid database (isotopes + adducts + confidence). _(needs: feature_set; params: mode, match_ppm, image_ppm, norm)_

**Cohort**
- `cohort_group_comparison` — Compare two subject groups across the whole cohort: each sample is one replicate, so the per-ion test (fold-change + q) is across subjects — the statistically defensible answer that the single-slide Region comparison only approximates per-pixel. _(needs: —; params: —)_
- `cohort_nested_comparison` — Per-ion mixed model y ~ group * compartment + (1 | subject) across the cohort — the correct multi-group-across-subjects test when each subject contributes several compartments/regions. _(needs: —; params: —)_
- `cohort_embedding` — One UMAP/t-SNE over every sample's pixels on a shared feature axis — see how samples and groups separate in molecular space across the cohort. _(needs: —; params: —)_
- `cohort_segmentation` — One segmentation tree learned jointly and applied to every sample, so the same cluster labels mean the same thing across slides. _(needs: —; params: —)_

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
