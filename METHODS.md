# Methods & references

Every quantitative method in the SMILE MSI engine, where it lives in the code, and a
literature source for it. Citations were verified against publisher / DOI-resolver /
PubMed / arXiv pages. The auto-generated provenance report
(`smile_msi/provenance.py`) cites the subset of these that a given analysis actually
used; see `Provenance.methods_paragraph()` / `references_used()`.

Trivial standard operations are listed without citations at the end.

---

## 1. Spectral preprocessing

| Method | Where | What it does |
|---|---|---|
| **SNIP baseline** | `preprocess.py` `baseline_snip()` / `reduce_baseline(method='snip')` (core `_snip()`) | Statistics-sensitive non-linear iterative peak-clipping on an LLS-transformed spectrum. |
| **Savitzky–Golay smoothing** | `preprocess.py` `smooth()` (`scipy.signal.savgol_filter`) | Local least-squares polynomial smoothing. |
| **Lock-mass recalibration** | `preprocess.py` `recalibrate()` | Per-spectrum m/z shift from reference masses, median-corrected, re-interpolated. |
| **Absolute calibration-offset measurement** | `intake.py` `measure_calibration_offset()` / `default_calibration_anchors()` | Compares the mean-spectrum apexes to *known* reference ion m/z (engine-computed from the in-silico DB via `masses.ion_mz`, not hardcoded) and reports the systematic ppm offset, its mass-slope, and the `1/(1+offset)` correction factor. Unlike `mass_drift()` (per-pixel spread vs. an auto-picked peak) and `auto_recalibrate()` (self-referential spread reduction), this recovers the *absolute* error that drives confident lipid mislabels — the measure step before a one-point lock-mass `recalibrate()`. |

- Ryan, C.G., Clayton, E., Griffin, W.L., Sie, S.H. & Cousens, D.R. (1988). SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications. *Nuclear Instruments and Methods in Physics Research B*, 34(3), 396–402. doi:10.1016/0168-583X(88)90063-8
- Morháč, M. & Matoušek, V. (2008). Peak clipping algorithms for background estimation in spectroscopic data. *Applied Spectroscopy*, 62(1), 91–106. doi:10.1366/000370208783412762
- Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627–1639. doi:10.1021/ac60214a047

## 2. Peak detection & noise

| Method | Where | What it does |
|---|---|---|
| **Profile peak detection** | `msi.py` `pick_peaks()` (`scipy.signal.find_peaks`) | Local maxima on the mean/skyline spectrum above an S/N height threshold (+ optional prominence). The S/N is on the **absolute** intensity (`intensity/noise`) with **no baseline subtraction** by default — unlike MALDIquant `detectPeaks` / Cardinal `peakPick(method='mad')`, which define S/N on the baseline-corrected residual `(intensity − baseline)/noise`. `prominence` is the only baseline-relative gate in this path; the reported `snr` matches the cited convention only after a baseline step. |
| **Centroided peak 'picking'** | `msi.py` `_pick_centroids()` | Skips peak *detection* on already-centroided data and thresholds the recorded centroids directly (keeps their exact m/z). |
| **Representation detection** | `intake.py` `detect_representation()` | Centroid vs profile, reconciling the imzML flag with a data heuristic (run-length / m/z-gap); data wins on disagreement. |
| **MAD noise estimate** | `msi.py` `_mad()` | Robust scale = median(\|x − median(x)\|) × 1.4826 (the normal-consistency constant). |
| **Per-m/z-window noise** | `msi.py` `_local_noise()` | Windowed MAD (32 bins) interpolated back to the axis, so the S/N floor tracks the local baseline. |
| **Intensity-weighted centroid** | `msi.py` `_centroid()` | Refines a *profile* peak's m/z by the intensity-weighted mean of its neighbourhood (not used on centroided data). |

- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. doi:10.1038/s41592-019-0686-2
- Hampel, F.R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393. doi:10.1080/01621459.1974.10482962. (the 1.4826 = 1/Φ⁻¹(3/4) normal-consistency constant for the MAD)
- Rousseeuw, P.J. & Croux, C. (1993). Alternatives to the median absolute deviation. *Journal of the American Statistical Association*, 88(424), 1273–1283. doi:10.1080/01621459.1993.10476408. (secondary; on the Sn/Qn *alternatives* to the MAD, not the 1.4826 constant)
- Gibb, S. & Strimmer, K. (2012). MALDIquant: a versatile R package for the analysis of mass spectrometry data. *Bioinformatics*, 28(17), 2270–2271. doi:10.1093/bioinformatics/bts447

### Choosing a peak-finding approach — origin, licence & best fit

Every approach below is **reimplemented from the cited publications/documentation**, not
copied from the tools' source — so none of their licences attach (copyright protects code,
not algorithms; see `THIRD_PARTY_LICENSES.md`). The table is the at-a-glance map of *which
idea comes from where* and *when it is the right fit*.

| Approach | In this app | Origin (tool · licence of the original) | Best fit |
|---|---|---|---|
| **Local-maxima + S/N (per-window MAD), optional prominence** | `pick_peaks()` profile path | MALDIquant `detectPeaks` (GPL-3) · Cardinal `peakPick` method `mad` (Artistic-2.0) · on SciPy `find_peaks` (BSD-3) | **Profile / continuum** data — raw instrument spectra with a baseline and multi-sample peaks. The default, simplest reliable standard. |
| **Threshold recorded centroids (skip detection) → align/bin → frequency-filter** | `_pick_centroids()` + `spatial.find_spatial_features()` | Cardinal `peakAlign` + `peakFilter` (`freq.min`) convention (Artistic-2.0) | **Already-centroided** data — SCiLS exports, vendor-centroided imzML. Running a profile finder here is off-standard. |
| **CWT (continuous wavelet transform) peak detection** | *not implemented* (candidate) | MassSpecWavelet (LGPL) · `scipy.signal.find_peaks_cwt` (BSD-3) | Noisy / overlapping / partially-resolved **profile** peaks where simple local-maxima over- or under-calls. Heavier; usually unnecessary for high-res centroided MSI. |
| **Spatial feature finding** (mean candidates → pixel-frequency gate → Moran's-I denoise → auto interval width → *opt-in* isotope collapse) | `spatial.find_spatial_features()` | Built from published primitives: spatial autocorrelation (Moran's I), pixel-frequency reproducibility gating (Cardinal `peakFilter` convention, Artistic-2.0), spatial-coherence segmentation (Alexandrov & Kobarg, published), isotope collapse = `isotopes.deisotope()` (13C spacing, Senko et al. 1995) | Imaging-aware selection: keep ions that form coherent images, drop spatially-random noise, and (with `collapse_isotopes=True`) fold M+1/M+2 into the monoisotopic peak — the "region-complete" reduction (one compound → one feature). |

**The three knobs that actually set the feature count** (across every tool, not the detection
maths): the **S/N or intensity threshold**, the **m/z binning/alignment tolerance** (wide
merges peaks → fewer; narrow splits → more), and the **frequency filter** (minimum fraction of
pixels a peak must appear in; Cardinal `peakFilter` default = 1%). To compare like-for-like
across tools, match these three plus whether **isotopes/adducts are collapsed** (see §4) — that
collapse, not the picker, is the single biggest reason raw counts differ between tools.

## 3. Normalization

| Method | Where | What it does |
|---|---|---|
| **TIC / RMS / median normalization** | `msi.py` `norm_factors()`, `feature_matrix()` | Per-pixel scaling by total ion current, root-mean-square, or median intensity. The "median" statistic is the median of *positive* intensities; `norm_factors` rescales the **factors** to mean ≈ 1 (not the intensities); a per-pixel amplification cap (`tic_max_amp`, default 3×) limits how much a low-TIC pixel can be scaled up (the factor is floored at `1/tic_max_amp`). |
| **Normalization-state detection** | `intake.py` `detect_normalization()` | Flags data that is *already* normalized — a per-pixel statistic (TIC/RMS/median) with coefficient of variation ≈ 0 — so the intake inspector suggests not re-normalizing. Suggestions are limited to TIC / none (median/exclusion-list recipes are withheld pending patent review). |

- Deininger, S.-O., Cornett, D.S., Paape, R., Becker, M., Pineau, C., Rauser, S., Walch, A. & Wolski, E. (2011). Normalization in MALDI-TOF imaging datasets of proteins: practical considerations. *Analytical and Bioanalytical Chemistry*, 401(1), 167–181. doi:10.1007/s00216-011-4929-z

## 4. Lipid annotation, isotopes & FDR

| Method | Where | What it does |
|---|---|---|
| **Monoisotopic/atomic masses** | `masses.py` `ELEMENTS` | Static element mass table used to build theoretical ion m/z. |
| **Isotope-pattern model** | `isotopes.py` `isotope_consistency()`, `deisotope()` | ¹³C–¹²C spacing (1.00336 Da) and ~1.1%·n_carbon M+1 expectation (averagine-style). |
| **Target–decoy FDR** | `annotate.py` `estimate_fdr()` | Decoy DB = real formulas paired with chemically *implausible element adducts* (He, Be, Sc, …; `masses.DECOY_ELEMENT_MASSES`), a **random (seeded) sample** of `n_decoy` of them. Each ID is scored by MSM (mass × spectral-isotope × spatial-isotope); the target per peak is the **best-MSM** of its in-tolerance candidates. Each sampled element is an **independent** null (one best decoy per peak), and the q-value is the **median over decoy samples** of `(#decoys_e≥t + 1)/(#targets≥t + 1)`, made monotonic (non-increasing in score) — not a single pooled ranking, which would over-count a peak that coincides with many decoy elements (Palmer et al. 2017). The returned `decoy_rate`/`target_rate` are display fields only, not the FDR. |
| **MS/MS class confirmation** | `msms.py` `classify()`, `confirm_class()` (`DIAGNOSTIC_FRAGMENTS`, `DIAGNOSTIC_LOSSES`) | Rule-based MS/MS confirmation: presence of an expected diagnostic product ion / neutral loss within a Da tolerance (**not** a spectral dot-product). Conservative rules — a *confirmed* class is high-confidence; absence is not refutation. |

*Triage heuristics (not calibrated posteriors).* The `match.py` MS1 ranking score
(`-|ppm| + scale·(tissue_prior + adduct_prior)`, with `scale = ppm_tol · 0.1` so the
class/adduct priors only break ties within a fixed fraction of the mass window) and the
Features-tab `confidence_score` (`isotopes.py` `confidence()` / `confidence_detail()`) are
in-house triage heuristics for ordering candidates, orthogonal to and not interchangeable
with the MSM/FDR confidence above. When a dataset is present, `annotate.build_feature_list`
**re-ranks** each peak's candidates by their MSM score (mass × spectral × spatial), so an
isobaric runner-up the priors mis-ranked is promoted when the image evidence supports it
(METASPACE's ranking criterion; Palmer et al. 2017).

- Wang, M., Huang, W.J., Kondev, F.G., Audi, G. & Naimi, S. (2021). The AME 2020 atomic mass evaluation (II). Tables, graphs and references. *Chinese Physics C*, 45(3), 030003. doi:10.1088/1674-1137/abddaf
- Meija, J. et al. (2016). Atomic weights of the elements 2013 (IUPAC Technical Report). *Pure and Applied Chemistry*, 88(3), 265–291. doi:10.1515/pac-2015-0305
- Senko, M.W., Beu, S.C. & McLafferty, F.W. (1995). Determination of monoisotopic masses and ion populations for large biomolecules from resolved isotopic distributions. *Journal of the American Society for Mass Spectrometry*, 6(4), 229–233. doi:10.1016/1044-0305(95)00017-8
- Elias, J.E. & Gygi, S.P. (2007). Target-decoy search strategy for increased confidence in large-scale protein identifications by mass spectrometry. *Nature Methods*, 4(3), 207–214. doi:10.1038/nmeth1019
- Palmer, A. et al. (2017). FDR-controlled metabolite annotation for imaging mass spectrometry. *Nature Methods*, 14(1), 57–60. doi:10.1038/nmeth.4072.
- Hsu, F.-F. & Turk, J. (2009). Electrospray ionization with low-energy collisionally activated dissociation tandem mass spectrometry of glycerophospholipids: mechanisms of fragmentation and structural characterization. *Journal of Chromatography B*, 877(26), 2673–2695. doi:10.1016/j.jchromb.2009.02.033
- Murphy, R.C. (2015). *Tandem Mass Spectrometry of Lipids*. Royal Society of Chemistry. doi:10.1039/9781782626350
- Han, X. & Gross, R.W. (2005). Shotgun lipidomics: electrospray ionization mass spectrometric analysis and quantitation of cellular lipidomes directly from crude extracts of biological samples. *Mass Spectrometry Reviews*, 24(3), 367–412. doi:10.1002/mas.20023

## 5. Segmentation & dimensionality reduction

| Method | Where | What it does |
|---|---|---|
| **PCA** | `spatial.py` `_embed()`; `multivariate.py` | Linear dimensionality reduction before clustering (`sklearn.decomposition.PCA`). |
| **k-means** | `spatial.py` `segment()` (`sklearn.cluster.KMeans`) | Partitions PCA scores into k molecular regions. |
| **Silhouette** | `spatial.py` `_silhouette()` (`sklearn.metrics.silhouette_score`) | Cluster-count validity index for auto-choosing k. |
| **Two-stage agglomerative** | `spatial.py` `hierarchy()` + `cut()` | MiniBatch k-means over-segmentation → Ward linkage → `fcluster` cut. |
| **Divisive bisecting k-means** (default segmenter) | `spatial.py` `hierarchy(method='bisecting')` (`_bisecting_linkage()`) + `cut()` | The default segmentation: MiniBatch k-means micro-clusters are recursively split top-down (divisive bisecting k-means), returned as a scipy linkage so a tree cut to *k* reproduces bisecting stopped at *k*. |
| **Spatially-aware segmentation** | `multivariate.py` `spatial_segment()` | Gaussian-smooths each feature image before PCA + k-means. |
| **NMF** | `multivariate.py` `nmf_images()` (`sklearn.decomposition.NMF`, NNDSVD init) | Additive (parts-based) component decomposition. |
| **t-SNE / UMAP** | `multivariate.py` `embedding()` / `pooled_embedding()` | 2-D embedding of pixels for the linked scatter view (after `log1p` + `StandardScaler`, see below). |
| **ComBat batch correction** | `batchfx.py` `combat()` (orchestrated by `cohort.correct_batches()`; opt-in in `multivariate.pooled_embedding(batch_correct=True)`) | Empirical-Bayes per-feature location/scale adjustment removing cross-batch technical shifts; an optional protected-covariate design (ComBat's native covariate model matrix — *not* reComBat's regularized regression, which is not implemented) preserves the biological contrast. A `ref_batch` aligns the other batches onto the reference batch's location/scale (sva semantics). |
| **Batch-mixing QC (kBET-like) + silhouette-by-batch** | `batchfx.py` `batch_mixing()` / `correction_summary()` | Quantifies residual batch effect: a neighbourhood χ² rejection rate, silhouette by batch vs by biology, and per-feature batch-variance reduction — the before/after acceptance read. |

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572. doi:10.1080/14786440109462720
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441. doi:10.1037/h0071325
- Lloyd, S.P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137. doi:10.1109/TIT.1982.1056489
- MacQueen, J.B. (1967). Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
- Sculley, D. (2010). Web-scale k-means clustering. *Proc. 19th International Conference on World Wide Web (WWW '10)*, 1177–1178. doi:10.1145/1772690.1772862
- Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244. doi:10.1080/01621459.1963.10500845
- Steinbach, M., Karypis, G. & Kumar, V. (2000). A comparison of document clustering techniques. *KDD Workshop on Text Mining*. (divisive bisecting k-means)
- Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. doi:10.1016/0377-0427(87)90125-7
- Alexandrov, T. & Kobarg, J.H. (2011). Efficient spatial segmentation of large imaging mass spectrometry datasets with spatially aware clustering. *Bioinformatics*, 27(13), i230–i238. doi:10.1093/bioinformatics/btr246
- Lee, D.D. & Seung, H.S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788–791. doi:10.1038/44565
- Boutsidis, C. & Gallopoulos, E. (2008). SVD based initialization: a head start for nonnegative matrix factorization. *Pattern Recognition*, 41(4), 1350–1362. doi:10.1016/j.patcog.2007.09.010
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605. https://www.jmlr.org/papers/v9/vandermaaten08a.html
- McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426. doi:10.48550/arXiv.1802.03426
- Johnson, W.E., Li, C. & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127. doi:10.1093/biostatistics/kxj037
- Adler, M. et al. (2022). reComBat: batch-effect removal in large-scale multi-source gene-expression data integration. *Bioinformatics Advances*, 2(1), vbac016. doi:10.1093/bioadv/vbac016
- Büttner, M., Miao, Z., Wolf, F.A., Teichmann, S.A. & Theis, F.J. (2019). A test metric for assessing single-cell RNA-seq batch correction (kBET). *Nature Methods*, 16(1), 43–49. doi:10.1038/s41592-018-0254-1
- ComBat-in-MSI validations: *Anal. Chem.* (2025) doi:10.1021/acs.analchem.5c04371; doi:10.1021/acs.analchem.5c02020; bioRxiv (2026) doi:10.64898/2026.01.30.702769.

Clean-room note: ComBat's reference implementation is in the GPL **sva** package — reimplemented from the publication, copying no source (copyright protects code, not algorithms; see `THIRD_PARTY_LICENSES.md`). The Adler et al. (2022) reComBat reference (above) is a `cf.` pointer for the regularized-covariate extension we deliberately do **not** implement; only ComBat's native covariate model matrix is used to protect biology.

*Embedding/PCA pre-step:* before PCA (and before t-SNE/UMAP), the feature matrix is `log1p`-transformed
and `StandardScaler`-standardized (`multivariate._matrix`, `embedding()`, `pooled_embedding()`); the
pooled multi-sample embedding standardizes per sample to remove slide-to-slide offsets.

## 6. Spatial statistics & region comparison

| Method | Where | What it does |
|---|---|---|
| **Moran's I** | `spatial.py` `spatial_autocorrelation()` | Spatial autocorrelation of each ion image over the 4-neighbour grid. |
| **Mann–Whitney U / rank ROC AUC** | `spatial.py` `_auc_mwu()` | Per-ion in-vs-out discrimination; AUC = U/(n_a·n_b) with tie-corrected z p-value. |
| **Parametric / binormal AUC** | `pipeline.py` `compute_auc()` | `AUC = Φ((mB−mA)/√(sA²+sB²))` from per-group mean/SD — a distinct estimator from the rank-based MWU AUC, shipped in the Excel/export tables. The two can disagree (the binormal form assumes normal, equal-shape groups). |
| **Welch / Student t-test (region comparison)** | `spatial.py` `roi_comparison(method='welch'\|'student')` + cohort path | Two-sample t-test option for region comparison: `'welch'` = unequal variance (Welch–Satterthwaite df), `'student'` = pooled variance. |
| **Benjamini–Hochberg FDR** | `spatial.py` `_bh_fdr()` | Step-up multiple-testing correction (q-values). |
| **Kruskal–Wallis H** | `spatial.py` `multigroup_features()` (primary), `region_membership()` (secondary) (`scipy.stats.kruskal`) | Across-region difference test for multi-group screening and Venn/membership gating. |
| **ROI localization (in-vs-rest)** | `spatial.py` `roi_localization()` | Per-peak signed log2 fold-change + rank AUC of an ROI vs. the rest of the on-tissue slide. |
| **Euclidean distance transform** | `spatial.py` (`scipy.ndimage.distance_transform_edt`) | Ring/border masks around a region. |

- Moran, P.A.P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1–2), 17–23. doi:10.1093/biomet/37.1-2.17
- Mann, H.B. & Whitney, D.R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60. doi:10.1214/aoms/1177730491
- Hanley, J.A. & McNeil, B.J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36. doi:10.1148/radiology.143.1.7063747
- Bamber, D. (1975). The area above the ordinal dominance graph and the area below the receiver operating characteristic graph. *Journal of Mathematical Psychology*, 12(4), 387–415. doi:10.1016/0022-2496(75)90001-2
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. doi:10.1111/j.2517-6161.1995.tb02031.x
- Kruskal, W.H. & Wallis, W.A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621. doi:10.1080/01621459.1952.10483441
- Student [Gosset, W.S.] (1908). The probable error of a mean. *Biometrika*, 6(1), 1–25. doi:10.1093/biomet/6.1.1
- Welch, B.L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35. doi:10.1093/biomet/34.1-2.28
- Maurer, C.R., Qi, R. & Raghavan, V. (2003). A linear time algorithm for computing exact Euclidean distance transforms of binary images in arbitrary dimensions. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 25(2), 265–270. doi:10.1109/TPAMI.2003.1177156

## 7. Co-localization

| Method | Where | What it does |
|---|---|---|
| **Pearson / cosine similarity** | `spatial.py` `colocalize()`, `_similarity()` | Ranks ions by **spatial** (image-to-image) similarity to a target ion. The cosine here is over the two ion *images*, not a spectral library match. |
| **Manders / Dice footprint overlap** | `spatial.py` `_similarity()` (`m1`/`m2`/`dice`) | Footprint-overlap coefficients. Each image is auto-thresholded at the **median of its positive pixels** (Costes-style auto-threshold), distinct from Manders (1993)'s zero/presence threshold. |
| **Co-localization modules** | `spatial.py` `coloc_modules()` (`sklearn.cluster.AgglomerativeClustering`, average linkage / UPGMA) | Groups ions into modules on a correlation-distance matrix. |

- Ovchinnikova, K., Stuart, L., Rakhlin, A., Nikolenko, S. & Alexandrov, T. (2020). ColocML: machine learning quantifies co-localization between mass spectrometry images. *Bioinformatics*, 36(10), 3215–3224. doi:10.1093/bioinformatics/btaa085
- Costes, S.V., Daelemans, D., Cho, E.H., Dobbin, Z., Pavlakis, G. & Lockett, S. (2004). Automatic and quantitative measurement of protein-protein colocalization in live cells. *Biophysical Journal*, 86(6), 3993–4003. doi:10.1529/biophysj.103.038422
- Sokal, R.R. & Michener, C.D. (1958). A statistical method for evaluating systematic relationships. *University of Kansas Science Bulletin*, 38, 1409–1438. (average-linkage / UPGMA — used by `coloc_modules`)

## 8. Software stack

- Harris, C.R. et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. doi:10.1038/s41586-020-2649-2
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. doi:10.1038/s41592-019-0686-2
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf
- Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. doi:10.1109/MCSE.2007.55

---

## Standard operations (no citation)

Min–max / percentile contrast clipping (`imaging.py`, `multivariate.py`), alpha blending of
overlays, ppm error = (obs − theo)/theo × 10⁶ (`masses.py`), adduct m/z = (M + shift)/|z|,
log1p intensity transform, L2 vector normalization, geometric (constant-ppm) m/z binning,
sparse CSC ion cube (`scipy.sparse`), bisection lookup (`bisect`), connected-component
labelling (`scipy.ndimage.label`), SHA-256 file fingerprinting (`provenance.py`), and
CSV/TSV target-list parsing are standard implementation details rather than novel methods.

## Notes on a few references

- **t-SNE** and **scikit-learn** are *JMLR* papers with no DOI — cite the JMLR URLs above.
- **UMAP**: the method is the arXiv paper; the package also has a JOSS paper (doi:10.21105/joss.00861).
- **AME2020**: this is *Part II* (the tables/values paper, Wang et al.), which is the correct one to cite for mass values; Part I (Huang et al.) covers the evaluation procedure.
- **Ward 1963** is pp. 236–244; **Sokal & Michener 1958** is vol. 38 (both occasionally mis-cited).

---

## Methods added in the Cardinal / METASPACE alignment (2026-06-15)

These extend the engine toward the two reference tools.

- **Isotope envelopes** (`masses.isotope_distribution`, `ion_isotope_pattern`): exact
  convolution of per-element isotope distributions (CIAAW abundances), replacing the
  averagine carbon proxy. Element abundances — Meija, J. et al. (2016), *Pure Appl. Chem.*
  88(3):265–291, doi:10.1515/pac-2015-0305.
- **Spectral & spatial isotope scores; MSM scoring; implausible-adduct target–decoy FDR**
  (`isotopes.isotope_scores`, `annotate.estimate_fdr`): Palmer, A. et al. (2017),
  "FDR-controlled metabolite annotation for imaging mass spectrometry", *Nature Methods*
  14(1):57–60, doi:10.1038/nmeth.4072. Decoy elements are a seeded random sample; each
  sampled element forms an independent per-peak null, and the reported FDR is the **median**
  of the per-sample estimates `(d_e+1)/(t+1)` (Palmer's median-over-decoy-samples estimator)
  rather than one pooled ranking. The `+1` is an add-one (rule-of-succession-style) smoothing
  that avoids a spurious 0% FDR (cf. Laplace's rule of succession, 1814).
- **Sample-summarized region testing** (`spatial.roi_comparison(samples=...)`): per-sample
  summarization to avoid pseudoreplication, as in Cardinal's `meansTest` — Bemis, K.D. et al.
  (2023), *Nature Methods* 20:1883, doi:10.1038/s41592-023-02060-1. Both region-comparison
  UI entry points (the *Region comparison* tab and the *ROI stats (A vs B)* popup) expose a
  **Replicate unit** control that drives `samples=`: *Auto* runs the across-ROI test when each
  side has ≥2 ticked regions, else falls back to a clearly-labelled descriptive per-pixel read.
  The ROI-stats popup additionally passes `effect='pixel'`, so the rank-based **AUC, means and
  signed log2 fold-change stay per-pixel** (the discrimination effect size — Bamber 1975; Hanley & McNeil
  1982 — consistent with the per-pixel ROC panel) while only the `p`/`q` summarize across
  replicates; the result records this as `df.attrs['effect_unit'] == 'pixel'`.
- **Nearest shrunken centroids** (`spatial.shrunken_centroids`): Tibshirani, R. et al. (2002),
  *PNAS* 99(10):6567–6572, doi:10.1073/pnas.082099299; spatial variant — Bemis, K.D. et al.
  (2016), *Mol. Cell. Proteomics* 15(5):1761, doi:10.1074/mcp.O115.053918.
- **Spatial weights (gaussian SA / bilateral SASA)** (`multivariate.spatial_segment`):
  Bemis et al. (2016, as above); bilateral filtering — Tomasi & Manduchi (1998),
  doi:10.1109/ICCV.1998.710815.
- **Manders & Dice colocalization** (`spatial._similarity`): Manders, E.E.M. et al. (1993),
  *J. Microsc.* 169(3):375–382, doi:10.1111/j.1365-2818.1993.tb03313.x; Dice, L.R. (1945),
  *Ecology* 26(3):297–302, doi:10.2307/1932409.
- **Baselines locmin / hull / median** (`preprocess.py`): standard MS baseline operators as
  exposed by Cardinal's `reduceBaseline`.
- **Per-m/z-window noise for peak picking** (`msi._local_noise`): windowed MAD; the
  1.4826 normal-consistency constant — Hampel (1974), doi:10.1080/01621459.1974.10482962
  (Rousseeuw & Croux 1993 cover the Sn/Qn alternatives to the MAD, not this constant).
- **spatialDGMM (per-ion spatial mixture) & segmentationTest** (`multivariate.spatial_dgmm`,
  `segmentation_test`): Bemis, K.D. et al. (2019), *Bioinformatics* 35(14):i208,
  doi:10.1093/bioinformatics/btz219; spatially-variant finite mixtures — Sanjay-Gopal &
  Hebert (1998), *IEEE Trans. Image Process.* 7(7):1014, doi:10.1109/83.704309.
- **PLS-DA / OPLS-DA + VIP + leave-one-sample-out CV** (`multivariate.plsda`,
  `cross_validate`): PLS — Wold, S. et al. (2001), doi:10.1016/S0169-7439(01)00155-1;
  OPLS — Trygg & Wold (2002), *J. Chemom.* 16(3):119, doi:10.1002/cem.695; VIP — Chong &
  Jun (2005), doi:10.1016/j.chemolab.2004.12.011; grouped CV to avoid pixel leakage as in
  Cardinal — Bemis et al. (2023), doi:10.1038/s41592-023-02060-1.
- **SHAP biomarker discovery** (`explain.shap_importance` / `shap_panel`): a RandomForest
  (Breiman 2001, doi:10.1023/A:1010933404324) is fit one-vs-rest over the region/FTU
  labels and explained with **TreeExplainer** SHAP values, giving every pixel a per-ion
  per-region Shapley contribution. Each ion's **importance** = `mean(|SHAP|)` over pixels
  (magnitude, → bubble size); its **direction** = the Spearman rank correlation between
  intensity and its own SHAP value (sign, → bubble colour). Mirrors the human-kidney MSI
  atlas donor × molecule bubble plots. Refs: SHAP — Lundberg & Lee (2017), arXiv:1705.07874;
  TreeExplainer — Lundberg et al. (2020), *Nat. Mach. Intell.* 2:56, doi:10.1038/s42256-019-0138-9;
  Spearman (1904). `shap` is an optional dependency (`uv sync --extra shap`).

## Reproducibility: analysis profiles & determinism (2026-06-19)

- **Analysis profiles** (`profiles.py`): a named, versioned bundle of the processing settings
  that should stay constant across samples/slides (tolerances, normalization + TIC amp cap,
  peak-picking thresholds, the pre-processing chain, segmentation algorithm/metric/defaults,
  spatial-finder gates, random seed). The active profile pre-fills new-sample defaults; per-sample
  deviations from it are recorded into the session. Each profile carries the app + dependency
  versions that defined it, and a result records the active profile's name + version + content-hash.
  Design follows the "captured, defaulted, deterministic, reportable" principle and the
  config-profile pattern of workflow systems (nf-core profiles; Galaxy/QIIME2 provenance).
- **Determinism**: every stochastic step already takes a `random_state` (`spatial.py`,
  `multivariate.py`); the GUI now threads the active profile's seed through segmentation,
  joint segmentation, decomposition and embeddings, so the same input + profile reproduces
  identical results on a given machine (manifold layouts are reproducible within a machine, not
  bit-identical across BLAS builds).

---

## Methods added in the 2026 literature-gap roadmap (plans 02–10)

These extend the engine along the axes the field moved onto since ~2023 (see
`LITERATURE_GAPS_2026.md` / `plans/`). Each new method is also wired into the auto-generated
provenance (`provenance.py` `REFERENCES` / `methods_paragraph` / `_used_ref_tags`).

| Method | Where | Source |
|---|---|---|
| **Standards-compliant imzML + MIAMSIE validation + METASPACE round-trip** | `standards.py` | Schramm et al. 2012 (imzML), doi:10.1016/j.jprot.2012.07.026; McDonnell et al. 2015 (reporting), doi:10.1007/s00216-014-8322-6; Palmer et al. 2017 (METASPACE FDR), doi:10.1038/nmeth.4072 |
| **Absolute quantification (on-tissue calibration)** | `quantify.py` | Quantitative MSI: Hamm et al. 2012, doi:10.1016/j.jprot.2012.07.035; absolute case doi:10.1021/acs.analchem.5b04409 |
| **Spectral-library MS/MS (cosine · modified-cosine · spectral-entropy)** | `specmatch.py` | Li et al. 2021 (entropy), doi:10.1038/s41592-021-01331-z; Stein & Scott 1994 (cosine), doi:10.1016/1044-0305(94)87009-8 |
| **Multimodal landmark registration (affine/similarity/rigid + optional elastic)** | `registration.py` | Umeyama 1991, doi:10.1109/34.88573; scikit-image (van der Walt 2014); SimpleITK (Lowekamp 2013) |
| **Single-cell spatial metabolomics (segmentation + area-weighted pixel→cell)** | `singlecell.py` | SpaceM, Rappez et al. 2021, doi:10.1038/s41592-021-01198-0; watershed, Beucher & Meyer 1993 |
| **3-D serial-section reconstruction (phase-correlation registration + volume)** | `volume3d.py` | Reddy & Chatterjee 1996, doi:10.1109/83.506761; volume rendering, Levoy 1988, doi:10.1109/38.511 |
| **Spatial multi-omics co-mapping (resample + metabolite↔gene correlation)** | `comap.py` | Spearman 1904; spatially-resolved multi-omics, npj Imaging 2024, doi:10.1038/s44303-024-00025-3 |
| **Learned ion-image embeddings (self-supervised contrastive colocalization)** | `ionembed.py` + `spatial.colocalize(method="learned")` | DeepION, Wang et al. 2024, doi:10.1021/acs.analchem.3c05002; SimCLR, Chen et al. 2020 |

Optional/heavy dependencies (`metaspace2020`, `scikit-image`/`SimpleITK`, `cellpose`/`stardist`,
`pyqtgraph[opengl]`, `anndata`, and `torch` for the learned-embedding arm) are **lazy-imported with
a clear install error — never a silent fallback** (`THIRD_PARTY_LICENSES.md`; pyproject extras
`metaspace` / `register` / `cells` / `viz3d` / `spatialomics`). Batch correction (plan 01) is
documented in §5.
