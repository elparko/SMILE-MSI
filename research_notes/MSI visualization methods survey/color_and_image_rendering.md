# Color encoding and 2D image-level rendering methods for MSI (whole-cube / many-ion images)

Scope: published methods for showing many ions or the whole hyperspectral cube as one high-impact 2D image, judged on visual impact, scientific value, and how hard each would be in Python (numpy/scipy/matplotlib). Each method is flagged **[NEW]** (absent from SMILE-MSI), **[UPGRADE]** (SMILE-MSI has a weaker version), or **[HAVE]**.

What SMILE-MSI already does, checked in the code on 2026-10-07:
- `smile_msi/multivariate.py::_coords_rgb` colours pixels from a **2D** embedding with an HSV wheel: hue is the angle and saturation is the radius, with V fixed at 1. It does not use a perceptual space and has no lightness axis.
- `gui/exportdialog.py::_compose_overlay_rgb` already adds together N tinted channels and clips the sum, so it can composite more than 3 ions.
- `umapstudio.py` has glasbey, tab10 and okabe_ito palettes (colorcet is optional).
- `export.py` draws region outlines with `ax.contour(mask, levels=[0.5])`.
- `gui/segment.py` cuts a hierarchical **tree** with a "Detail" slider, but it colours segments with a **12-colour palette that cycles** (`PALETTE[cl % len(PALETTE)]`). The code's own comment says the key "repeats heavily" past 12. This is the clearest gap relative to the Lipizones look the user wants.

Coordinator note: the target aesthetic is the "Lipizones" mouse-brain lipid atlas. Many hierarchical clusters are coloured so that related clusters share related hues. Another researcher is covering that paper. Below I only verify its citation and survey the colour-encoding methods that produce that look.

---

## Q0 (priority). Tree-aware palettes: colouring many hierarchical clusters so related clusters share related hues (the Lipizones look)

### Takeaway
The look comes from **hierarchical, or "tree", colour maps**. The standard published algorithm is **Tree Colors** (Tennekes & de Jonge, IEEE TVCG 2014). It recursively splits the hue circle among the subtrees of a dendrogram in HCL space, and it changes chroma and luminance with depth. As a result, siblings get neighbouring hues and distant branches get distant hues. Two alternatives exist:
- **Embedding-derived cluster colours**: map each cluster's centroid position in a 3D UMAP or PCA into CIELAB or OKLab, as in U-CIE and spatially mapped t-SNE.
- **Spatially-aware palette optimisation**: Palo and Spaco, which deliberately do the opposite and make spatially *adjacent* clusters contrast.

All of these fit in under about 150 lines of numpy. SMILE-MSI's hierarchical segmentation already builds the tree they need, so this is the highest-value **[NEW]** item.

### Cited Findings
- **Lipizones paper (citation check only).** "The lipidomic architecture of the mouse brain." Fusar Bassini, D'Angelo, La Manno et al., bioRxiv, posted 13–14 Oct 2025, DOI 10.1101/2025.10.13.682018.
  - 539 lipidome-defined clusters ("lipizones") were built from 172 lipids, and the data were mapped to the Allen CCF. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.13.682018.full.pdf); [preLights](https://prelights.biologists.com/?p=42883)
  - The data are 109 coronal sections from 11 mice, about 7 million readings, assembled into 3D. — [Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/)
  - A news item says the paper has since appeared in *Nature*, and one search summary gave DOI 10.1038/s41586-026-11050-0. **I did not verify the Nature DOI**; treat it as unconfirmed. — [News-Medical, 23 Sep 2026](https://www.news-medical.net/news/20260923/New-3D-lipid-atlas-maps-the-chemistry-of-mouse-brains.aspx)
  - I could not retrieve how its palette was generated (see Gaps).
- **Same lab, earlier paper.** uMAIA ("Unified mass imaging maps the lipidome of vertebrate development", Schede et al., *Nature Methods* 2025, DOI 10.1038/s41592-025-02771-7) is an earlier La Manno / D'Angelo MALDI-MSI framework. It built a 4D zebrafish lipid atlas covering more than 100 lipids. — [PMC12446072](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12446072/); [EPFL news](https://actu.epfl.ch/news/mapping-the-lipid-blueprint-of-life-in-4d-3)
- **Tree Colors.** Tennekes & de Jonge, "Tree Colors: Color Schemes for Tree-Structured Data", *IEEE TVCG* 20(12):2072–2081, 2014, DOI 10.1109/TVCG.2014.2346277.
  - It assigns colours to tree-structured categories in the HCL model, with tunable parameters.
  - A user survey found the colours also "unveil tree structure in non-hierarchical visualizations", which matches the spatial-map use case.
  - Open-source R implementation: `treemap::treepalette(method="HCL")`.
  - An erratum appeared in TVCG 21(1):136, 2015.
  - Sources: [paper PDF](https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/2072_20tvcg12-tennekes-2346277.pdf); [R docs](https://rdocumentation.org/packages/treemap/versions/2.4-4/topics/treecolors); [2013 VIS poster](https://ieeevis.org/year/2013/poster/hierarchical-qualitative-color-palettes)
- **Refinements of Tree Colors (2024).** Mertz & Kohlhammer, "Towards a Quality Approach to Hierarchical Color Maps", IEEE VIS 2024 short paper, arXiv 2407.08287. It covers these Tree Colors variants:
  - Adjusted chroma/luminance ranges that reach maximum chroma at the leaves while staying in gamut.
  - A *proportional* hue split, which is better within subtrees and worse between them.
  - *Local interpolation* of chroma and luminance, which gives leaves equal visual weight.
  - Cheat-sheet rules: permute hues to avoid a perceived ordering, interpolate L and C linearly, and use the full hue range.
  - Sources: [arXiv](https://arxiv.org/pdf/2407.08287); [IEEE VIS 2024](https://nc.ieeevis.org/year/2024/program/paper_v-short-1184.html)
- **Dynamic Tree Colors.** A follow-up gives a hierarchical map that trades discriminability against colour stability under interaction. It was evaluated in an 18-participant study. — [ATHENE listing](https://www.athene-center.de/forschung/publikationen/dynamic-tree-colors-adaptive-discriminable-hierarc-5471)
- **Embedding-centroid colouring.** A 2024 arXiv preprint runs k-means and maps the low-dimensional cluster representations into **OKHSL**, "ensuring visual similarity between clusters reflects their underlying similarity in the data". — [arXiv 2410.07125](https://arxiv.org/pdf/2410.07125)
- **Large hierarchies in practice.** The Allen whole-mouse-brain taxonomy has 4 nested levels: 34 classes, 338 subclasses, 1,201 supertypes and 5,322 clusters. This is the kind of hierarchy such palettes must serve. I did **not** find documentation of how Allen assigns its colours. — [Yao et al. Nature 2023, PMC10719114](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10719114/)
- **Spatially-aware palette tools.** These solve the *complementary* problem: making spatially neighbouring clusters distinguishable.
  - **Palo** (Hou & Ji, *Bioinformatics* 38(14):3654–3656, 2022, DOI 10.1093/bioinformatics/btac368). It finds spatially neighbouring cluster pairs and gives them more distinct colours. R package. — [PMC9272793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9272793/)
  - **Spaco** (Jing et al., *Patterns* 5(3):100915, 2024, DOI 10.1016/j.patter.2023.100915). It builds a cluster-interlacement graph and matches it to a colour-difference graph, so that interlaced clusters get contrasting colours. It accounts for colour-vision deficiency (CVD) and has Python and R versions. A STAR Protocols version came out in June 2024. — [PMC10935509](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10935509/); [OmicVerse tutorial](https://omicverse.readthedocs.io/en/latest/Tutorials-plotting/t_palette.html)
- **MSI cluster colours from 3D UMAP.** A 2025 MSI study (UMAP + k-means) maps 3D UMAP coordinates directly to RGB and paints each pixel with its *cluster-centroid* colour. The authors chose 50 colours by eye: too few under-differentiated, too many became indistinguishable. — [PMC12130678](https://pmc.ncbi.nlm.nih.gov/articles/PMC12130678/)

### Inferences
- **Recommended Python recipe.** This is my synthesis, not from one paper.
  1. Take the segmentation linkage matrix (scipy `linkage` / `to_tree`), which SMILE-MSI already has.
  2. Recursively give each node a hue interval [h0, h1]. Each child gets a sub-interval proportional to its leaf count, with a small gap between siblings (Tree Colors' "fraction" parameter, about 0.75).
  3. Set hue to the interval midpoint. Set chroma rising and lightness falling or alternating with depth.
  4. Convert LCh to sRGB through CIELAB or OKLab and clip to gamut (about 30 lines of numpy; skimage `lab2rgb` also works).
  5. Permute sibling order so adjacent leaves do not form a visible rainbow ramp (Mertz & Kohlhammer).
  - This gives the "families of hues" look: e.g. all white-matter lipizones in blues, all cortical layers in greens and teals.
- **Hybrid recipe.**
  1. Assign coarse branches by Tree Colors.
  2. Within a branch, order leaves by their 1D position along the branch's principal axis in UMAP space. This places colour gradients along real chemical gradients.
  3. Optionally run a Palo or Spaco-style swap step so that *spatially adjacent* sibling leaves get contrasting lightness.
- **Matching colour across views.** For a "zoomable" atlas look, colour the dendrogram branches, the UMAP scatter and the tissue map with the same tree palette. SMILE-MSI already shares one colour function between scatter and tissue in `_coords_rgb`, so the plumbing exists.
- **Value of a tree palette.** The 12-colour cycle in `gui/segment.py` destroys hierarchy information at more than 12 segments. A tree palette makes the Detail slider *meaningful visually*: splitting a segment produces two shades of the parent's hue rather than unrelated new colours.

### Gaps
- I could not access the Lipizones full text, figure legends or code, because bioRxiv, nature.com and preLights are blocked by the egress proxy. So I cannot confirm whether its palette is Tree-Colors-like, UMAP-derived, or hand-curated. The other researcher dissecting the paper should check the methods and plotting code.
- The Nature 2026 publication and its DOI are unverified.
- I found no MSI paper that formally evaluates tree-aware palettes.

---

## Q1. Hyperspectral colour coding: embedding pixels (PCA/t-SNE/UMAP) into colour spaces. Which is most perceptually sound?

### Takeaway
The lineage runs as follows:
- Fonville et al. 2013: the first "single overview image" colouring for MSI, applied to PCA, SOM and t-SNE.
- Abdelmoula et al. 2016 (PNAS): a 3D t-SNE embedding mapped to **L\*a\*b\***.
- Smets et al. 2019: UMAP, competitive with t-SNE and about 4× faster.
- U-CIE 2022: UMAP → CIELAB with gamut fitting, as a generic method.

The most perceptually sound choice is a **3D embedding mapped into a perceptually uniform space (CIELAB, or better OKLab), fitted to the sRGB gamut**. Equal embedding distance then roughly equals equal perceived colour difference. Direct XYZ→RGB and HSV wheels are not perceptually uniform. Newer MSI-specific work argues UMAP colourings can hide small features:
- Sarycheva et al., Anal Chem 2020/21: "structure-preserving and perceptually consistent".
- Gildenblat & Pahnke 2025: "truthful visualizations".

### Cited Findings
- **Fonville et al., "Hyperspectral visualization of mass spectrometry imaging data"**, *Anal. Chem.* 85(3):1415–1423, 2013, DOI 10.1021/ac302330a.
  - A colour-coding scheme from hyperspectral imaging that makes one overview image. Pixels with similar molecular profiles get similar colours.
  - Applied to PCA, self-organizing maps and t-SNE (rat brain MALDI).
  - Sources: [ACS](https://pubs.acs.org/doi/10.1021/ac302330a); [Birmingham repository](https://research.birmingham.ac.uk/en/publications/hyperspectral-visualization-of-mass-spectrometry-imaging-data/)
  - A Mass Spec Reviews figure reproducing it describes the pixels as "color-coded with RGB values determined by the t-SNE manifold learning method". **Whether Fonville used CIELAB is unconfirmed.** — [Verbeeck, Caprioli & Van de Plas, Mass Spectrom Rev 2020, DOI 10.1002/mas.21602, Fig. 18](https://pmc.ncbi.nlm.nih.gov/articles/PMC7187435/figure/mas21602-fig-0018)
- **Abdelmoula et al., PNAS 113(43):12244–12249, 2016**, "Data-driven identification of prognostic tumor subpopulations using spatially mapped t-SNE of MSI data".
  - "Each pixel is colored according to its location in the 3D t-SNE space using L\*a\*b\* color coordinates, revealing a patchwork of subpopulations."
  - Linked to survival in gastric cancer and to metastasis in breast cancer.
  - Source: [PMC5087072](https://pmc.ncbi.nlm.nih.gov/articles/PMC5087072); [TU Delft](https://research.tudelft.nl/en/publications/data-driven-identification-of-prognostic-tumor-subpopulations-usi/)
  - This is the canonical perceptual-space MSI colouring and the best-looking classic: a stained-glass patchwork of tumour subpopulations.
- **Smets et al., Anal. Chem. 2019**, DOI 10.1021/acs.analchem.8b05827. UMAP for MSI, compared against PCA, t-SNE and Barnes-Hut t-SNE.
  - "Competitive with t-SNE in terms of visualization", with roughly a fourfold runtime decrease, and suited to more than 100,000 pixels.
  - Evaluates correlation, cosine and Chebyshev metrics plus a custom "histomatch" metric.
  - Sources: [ACS](https://pubs.acs.org/doi/10.1021/acs.analchem.8b05827); [KU Leuven report 19-55](https://ftp.esat.kuleuven.be/pub/stadius/ida/reports/19-55.pdf)
  - Follow-up: m/z prioritisation for UMAP profiles, *Anal. Chem.* 92:5240–5248, 2020. It reduces to 3 dimensions for hyperspectral visualisation. — [ACS 10.1021/acs.analchem.9b05764](https://pubs.acs.org/doi/10.1021/acs.analchem.9b05764)
- **Hu et al. review, *Advanced Science* 2022** ("Emerging Computational Methods in MSI", DOI 10.1002/advs.202203339).
  - Compressed data are shown with either RGB or CIELAB schemes, which reveal anatomy "in a histology-like way".
  - It also describes a CIELab variant: a 2D t-SNE fills the a\*/b\* chromaticity channels, and lightness L\* comes from a separate component image.
  - Source: [Wiley](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202203339)
  - This variant is attractive because L\* can carry total ion intensity or a PCA-1 "anatomy" image while hue carries chemistry.
- **U-CIE (Koutrouli et al., *Protein Science* 2022, DOI 10.1002/pro.4388).**
  - Reduces data to 3D with UMAP and embeds them in CIELAB, because Euclidean distance there approximates perceived difference.
  - Not every L\*a\*b\* triple is displayable, so the point cloud must be fitted (rotated, scaled, translated) into the sRGB gamut.
  - Sources: [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4388); [PMC9387205](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9387205/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.12.02.470966.full.pdf)
- **MIA (arXiv 2606.00874, 2026, recent)** is a visual analytics system for multimodal spectral imaging.
  - Colours pixels on a **fixed-chroma CIELCh colour disk**.
  - Fits UMAP on a random pixel subset and projects the remaining pixels into it.
  - Source: [arXiv](https://arxiv.org/pdf/2606.00874)
- **Sarycheva et al., "Structure-Preserving and Perceptually Consistent Approach for Visualization of MSI Datasets"**, *Anal. Chem.* (indexed 2020; Skoltech news Feb 2021). The DOI was not retrieved.
  - Benchmarks PCA, ICA, NMF, t-SNE and UMAP.
  - Proposes structure-preserving visualisation plus nonlinear embedding of normalised spectra, exploiting the nonlinear brightness response and the different sensitivity to luminance versus chroma.
  - Two formal properties: "consistency", meaning similar spectra get similar output, and local-contrast preservation.
  - Sources: [AGRIS](https://agris.fao.org/search/en/records/65df2dd863b8185d9cab5d0e); [Skoltech](https://skoltech.ru/en/archived-news/a-new-perceptually-consistent-method-for-msi-visualization); [JINR slides](https://indico.jinr.ru/event/5170/contributions/31700/attachments/22827/40409/Sidorchuk.pdf)
- **Gildenblat & Pahnke, "Truthful visualizations for MSI enable high spatial resolution interactive m/z mapping and exploration"**, bioRxiv 10.1101/2025.03.18.643852 (2025, v4; **recent**).
  - Argues the core problem is preserving biologically meaningful structure, not just reducing dimensions.
  - Reports that UMAP-based visualisation missed most small amyloid plaques and highlighted only the centres of large ones.
  - Has a companion Python package, `msi_visual`, which adapts a global-structure-preserving DR method. It is the same author, but I did not confirm it is the identical project.
  - Sources: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.18.643852v4.full); [GitHub](https://github.com/jacobgil/msi_visual)
- **QUIMBI (Wüllems et al., *Sci. Rep.* 2021, DOI 10.1038/s41598-021-84049-4)**: "interactive dynamic spectral similarity pseudocoloring" for fast MSI exploration, used with the ProViM pre-processing. I did not retrieve the exact colouring algorithm. — [PMC7907387](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7907387/)
- **Commercial adoption.** Shimadzu IMAGEREVEAL MS offers UMAP → 3D (RGB) projection and segmentation. A Shimadzu patent describes assigning R, G and B to three reduced axes and keeping a fitted UMAP so new samples keep the same colours. — [Shimadzu](https://www.shimadzu.com/an/products/life-science-lab-instruments/imaging/imagereveal-ms/option.html); [USPTO 11545348](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11545348)
- **HSNE.** Hierarchical SNE has been applied to 3D MSI for interactive full-resolution exploration (TU Delft). — [TU Delft](https://pure.tudelft.nl/portal/en/publications/interactive-visual-exploration-of-3d-mass-spectrometry-imaging-data-using-hierarchical-stochastic-neighbor-embedding-reveals-spatiomolecular-structures-at-full-data-resolution(b82e9984-6110-4044-9054-0b25bf4a5dd7).html)
- **msiPL (Abdelmoula et al., *Nat. Commun.* 12:5544, 2021, DOI 10.1038/s41467-021-25744-8).** A variational autoencoder whose "encoded features" (latent images) reveal anatomy and tumour heterogeneity. The latent space is a natural source for colour coding. — [PMC8452737](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8452737/)

### Inferences
- **Ranking by perceptual soundness:**
  1. 3D embedding → OKLab or CIELAB, gamut-fitted. This is U-CIE style, or Abdelmoula 2016 style.
  2. 2D embedding → a\*/b\* (or an OKLCh hue/chroma disk) with L\* from an independent intensity or anatomy image. This is Hu-review style.
  3. Fixed-chroma CIELCh disk (MIA).
  4. HSV wheel, which is what SMILE-MSI's `_coords_rgb` does now.
  5. Raw xyz → RGB.
- **SMILE-MSI's HSV wheel** gives yellows and cyans much higher perceived lightness than blues, which creates false "hotspots". Its centre-greyness is a reasonable idea. **[UPGRADE]**: swap HSV for OKLCh, and fix L or tie L to TIC/PCA-1. This costs about 20 lines.
- **Add a 3D-UMAP → OKLab mode with gamut fitting [NEW].**
  1. Fit PCA on the 3D embedding and align the first axis to L.
  2. Scale so the 1st–99th percentile range fits inside the sRGB gamut at each L.
  3. Clip.
  - This is moderately easy (about 60 lines, numpy only). It produces the classic "stained-glass" look of Abdelmoula 2016.
- **Gamut fitting** can be done crudely by shrinking chroma until in gamut, per pixel, after the OKLab→sRGB transform.

### Gaps
- Fonville 2013's exact colour space is unverified, because the full text was not accessible.
- The Sarycheva DOI and algorithm details were not retrieved.
- I found no head-to-head perceptual user study of colour-coding schemes on MSI specifically.

---

## Q2. Multi-channel overlays beyond 3-colour RGB (4–8+ channels, CMY, IMC-style, Glasbey, OKLab blending)

### Takeaway
Published MSI figures rarely composite more than about 4–5 ions in one image. The typical styles are magenta/red/green/yellow pseudocolour, or two-modality overlays. The strongest method for *optimising* N-channel pseudocolour palettes is **psudo** (Warchol et al., EuroVis/CGF 2024). It comes from multiplexed tissue imaging (CyCIF/IMC) and is directly transferable. For categorical many-colour needs, use **Glasbey** palettes (CIELAB max-min distance) computed in OKLab. SMILE-MSI already does additive N-channel compositing. What is **new** would be:
- palette optimisation for overlaps, as in psudo,
- max-blending or "screen" blending options,
- CVD-safe channel defaults.

### Cited Findings
- **Multimodal MSI of mammalian liver**, *Developmental Cell* 2024 (S1534-5807(24)00045-5). Selected lipids were overlaid and pseudo-coloured: magenta PC 32:1, red SM 42:2, green PC 32:0, yellow TG 52:2. It also showed dual GCIB-SIMS overlays at 3 µm. — [Cell](https://www.cell.com/developmental-cell/fulltext/S1534-5807(24)00045-5); [PMC11656446](https://pmc.ncbi.nlm.nih.gov/articles/PMC11656446/)
- **Subcellular transmission-geometry ambient MSI (2025)** shows 4-ion and 5-nucleotide colour overlays. — [PubMed 41093837](https://pubmed.ncbi.nlm.nih.gov/41093837/)
- **psudo** (Warchol, Troidl, Muhlich, …, Sorger, Pfister; bioRxiv 10.1101/2024.04.11.589087; *Computer Graphics Forum* 43(3):e15103, EuroVis 2024, **recent**).
  - Optimises palettes for multichannel spatial data by maximising perceptual differences between channels while limiting confusing blends where channels overlap.
  - Provides a lens that shows channel overlap and a confusion metric.
  - In a crowdsourced study (n=150), users were more accurate with the optimised palettes.
  - Sources: [Harvard VCG](https://vcg.seas.harvard.edu/publications/20240610-psudo); [EG diglib](https://diglib.eg.org/handle/10.1111/cgf15103); [PMC11042212](https://pmc.ncbi.nlm.nih.gov/articles/PMC11042212.2)
- **Glasbey et al., "Colour displays for categorical images"**, *Color Res. Appl.* 32(4):304–309, 2007. Colours are chosen sequentially, or by simulated annealing, to maximise the minimum CIELAB Euclidean distance. — [Strathprints](https://strathprints.strath.ac.uk/30312/)
  - colorcet ships precomputed Glasbey sets, because generating them is slow. — [colorcet](https://colorcet.holoviz.org/user_guide/Categorical.html)
  - The Python `glasbey` package (L. McInnes, MIT, v0.3.0 May 2025) can create and extend palettes. — [docs](https://glasbey.readthedocs.io/en/latest/); [PyPI](https://pypi.org/project/glasbey/)
- **OKLab** (Ottosson, Dec 2020) is a CIELAB successor with better hue and lightness prediction and better blending. It fixes CIELAB's blue hue shifts. It is in the CSS Color 4/5 drafts. — [Wikipedia](https://en.wikipedia.org/wiki/Oklab_color_space); [Smashing interview 2024](https://smashingmagazine.com/2024/10/interview-bjorn-ottosson-creator-oklab-color-space/)
- **MIBI-style composites.** The MIBI patent describes "composite images comprised of pseudo-colored categorical features and quantitative three-color overlays". — [USPTO 10041949](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10041949)

### Inferences
- **Additive blending.** Additive (sum-and-clip) N-channel compositing on a black background gives the fluorescence look, but it saturates to white where more than 2 channels overlap.
- **Options worth adding [NEW, easy]:**
  - "max" blend: per-pixel argmax channel colour, with intensity as alpha.
  - "screen" blend: `1-∏(1-cᵢ·xᵢ)`.
  - Averaging in OKLab with intensity weights, which avoids muddy browns.
- **Porting psudo [NEW, moderate].** Its core objective (maximise pairwise ΔE between channel colours, penalise blends that collide with other channel colours) can be approximated with a small random search over hues in OKLCh. This could be weighted by measured pixel co-occurrence between ions, which SMILE-MSI can compute from colocalisation.
- **CMY on white** (subtractive) is a cheap new export style: multiply `1 - xᵢ·(1-cᵢ)`. It suits print, but I found no MSI paper that uses it, so treat it as untested.

### Gaps
- I found no MSI paper that composites more than 5–6 ions in one additive image or evaluates such composites.
- I found no MSI-specific use of OKLab blending.

---

## Q3. Image fusion with microscopy/histology, pan-sharpening, super-resolution, virtual staining

### Takeaway
- **Van de Plas et al. 2015 (*Nat. Methods*)** is the landmark. Regression from H&E to ion images predicts ion distributions at ≥10× finer resolution, with a 100 µm → 10 µm example.
- **Pan-sharpening** (SIMS+SEM, MALDI+IR) is classical and easy to implement.
- **Deep-learning work since 2024** moves into diffusion-model "virtual staining" of IMS (UCLA/Vanderbilt) and CNN super-resolution, which is high impact but heavy.
- **Molecular contour maps on whole-slide histology** (Sharman et al. JASMS 2023) are a simple, beautiful, honest alternative to alpha blending.

### Cited Findings
- **Van de Plas, Yang, Spraggins & Caprioli, "Image fusion of mass spectrometry and microscopy: a multimodality paradigm for molecular tissue mapping"**, *Nat. Methods* 12(4):366–372, 2015, DOI 10.1038/nmeth.3296.
  - Multivariate regression models one modality's variables from the other's.
  - Predicts ion distributions at ≥10× the measured resolution. Example: m/z 778.5 predicted at 10 µm from 100 µm IMS plus 10 µm H&E.
  - Also predicts ion distributions in unmeasured areas, and is extensible to MRI, CT and PET.
  - Sources: [PMC4382398](https://pmc.ncbi.nlm.nih.gov/articles/PMC4382398); [Vanderbilt MSRC](https://lab.vanderbilt.edu/msrc-research-and-development/?p=343)
- **Pan-sharpening SIMS with SEM** (Tarolli, Jackson & Winograd, 2014): up to about an order of magnitude resolution gain, with a cross-correlation reliability metric. — [PMC4224624](https://pmc.ncbi.nlm.nih.gov/articles/PMC4224624); [Penn State](https://pure.psu.edu/en/publications/improving-secondary-ion-mass-spectrometry-image-quality-with-imag/)
- **Pan-sharpening MALDI with IR bands.** "Multimodal Chemical Analysis of the Brain by High Mass Resolution MS and IR Spectroscopic Imaging", *Anal. Chem.* 2018 (ac8b02913; authors not retrieved). MSI was acquired first, then IR on the same section. It pan-sharpens with many IR bands while keeping ion-image fidelity. — [ACS figshare SI](https://acs.figshare.com/articles/Multimodal_Chemical_Analysis_of_the_Brain_by_High_Mass_Resolution_Mass_Spectrometry_and_Infrared_Spectroscopic_Imaging/7107773/1)
- **Sharman et al., "MALDI IMS-Derived Molecular Contour Maps: Augmenting Histology Whole-Slide Images"**, *JASMS* 34(5):905–912, 2023, DOI 10.1021/jasms.2c00370.
  - Argues heatmap alpha blending "obscur[es] subtle quantitative differences and distribution gradients".
  - Draws ion-intensity contour lines on PAS-stained whole-slide images (S. aureus kidney abscess).
  - Also contours NMF components, summarising hundreds of ions as a few contour sets.
  - Sources: [PMC10787559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10787559); [TU Delft PDF](https://research.tudelft.nl/files/151825697/jasms.2c00370.pdf)
- **Virtual staining of label-free IMS** (UCLA Ozcan group with Vanderbilt/Spraggins; arXiv 2411.13120, **recent**).
  - A diffusion model generates PAS-like histology from IMS at about 10× larger pixel size.
  - In blind tests on human kidney, the output closely matched real stains.
  - An identifier "sciadv.adv0741" suggests *Science Advances* 2025, **unverified**.
  - Sources: [arXiv](https://arxiv.org/pdf/2411.13120); [UCLA EE](https://www.ee.ucla.edu/deep-learning-advances-imaging-mass-spectrometry-with-virtual-histological-detail/); [hyper.ai](https://hyper.ai/en/papers/sciadv.adv0741)
- **LCRN residual network** (*Advanced Science* 2025, DOI 10.1002/advs.202512662, **recent**). Fuses MALDI with microscopy for up to 20× super-resolution of plant tissue, using an "edge perceptual loss". — [PMC12866858](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12866858/)
- **Analyst 2026 benchmark.** Benchmarks SwinIR, MambaIR and ResShift (diffusion) against GAN-based MOSR for MSI super-resolution, using ResShift fine-tuned on mouse-brain images. — [RSC d6an00012f](https://pubs.rsc.org/en/content/articlelanding/2026/an/d6an00012f/unauth)
- **DeepSURE.** An H&E-constrained super-resolution preprint (ChemRxiv 2022). — [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/63a2c21aa53ea640a052bcf8)

### Inferences
- **Easiest high-impact additions:**
  - **Contour-on-histology export [UPGRADE/NEW, easy].** SMILE-MSI already uses `ax.contour` for masks. Extend it to iso-intensity lines (e.g. 50/75/90th percentile) for 1–3 ions or NMF components, drawn on the registered optical image.
  - **Pan-sharpening [NEW, easy-moderate].** For example, Brovey or IHS: `ion_up * (pan / lowpass(pan))`, with the optical luminance as "pan". Or use the Van de Plas-style ridge/PLS regression from optical features (RGB + texture filters) to each ion, which is scikit-learn-level work.
- **Labelling.** Both pan-sharpening and regression fusion produce *predicted* images. The UI should label them as such, with a reliability metric (cross-validated R²).
- **Deep-learning options** (diffusion virtual staining, SR networks) are high impact but out of scope for a numpy/scipy desktop app without GPU models.

### Gaps
- I did not verify the journal publication of the UCLA virtual-staining work.
- The authors of the 2018 IR pan-sharpening paper were not retrieved.

---

## Q4. Relief / hillshade / 2.5D, contour/isoline maps, bivariate colour maps, ratio images

### Takeaway
- **Contour/isoline ion maps** have direct MSI precedent (Sharman 2023).
- **Ratio imaging** has a recent untargeted MSI method (Cheng et al., eLife 2024/2025).
- **Bivariate two-ion colour maps and hillshaded "relief" ion images** have **no MSI-specific publication that I could find**. Their design principles come from cartography and scientific visualisation.
- All four are easy to implement in matplotlib and would be **[NEW]** for SMILE-MSI.

### Cited Findings
- **Contour maps on histology**: Sharman et al. JASMS 2023 (see Q3). — [PMC10787559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10787559)
- **Cheng et al., "Untargeted pixel-by-pixel metabolite ratio imaging as a novel tool for biomedical discovery in MSI"**, *eLife* 13:RP96892 (reviewed preprint 2024; final 2025).
  - An R workflow that computes ratio images for *all* metabolite pairs to find biology hidden in single-ion heatmaps.
  - Code: GitHub `qic2005/Untargeted-mass-spectrometry-ratio-imaging`.
  - Sources: [eLife](https://elifesciences.org/articles/96892); [PMC11919253](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11919253/)
- **Bivariate colour design.**
  - Ware et al., "Designing Pairs of Colormaps for Visualizing Bivariate Scalar Fields" (EuroVis 2020 short) found the **saturation-separation pair** best overall (one colormap high-saturation, the other low). — [EG diglib](https://diglib.eg.org/handle/10.2312/evs20201047)
  - Bivariate choropleths blend two univariate progressions into a 3×3 (or n×n) matrix legend. — [Magrit](https://magrit.cnrs.fr/en/functionalities/bivariate.html)
- **TrIQ** (threshold intensity quantization): contrast optimisation for MSI ion-image display, *PeerJ Comput. Sci.* 2021. — [PeerJ cs-585](https://peerj.com/articles/cs-585); [PMC8205298](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205298/)

### Inferences
- **Bivariate two-ion map [NEW, easy].**
  1. Normalise ions A and B to [0, 1] after hotspot clipping.
  2. Use a 2D LUT: corners black (neither), cyan (A), magenta or orange (B), white (both). Alternatively, build it in OKLab, with A on one hue and B on its opposite.
  3. Show a square legend.
  - Rationale: overlap regions read immediately, unlike a 2-channel RGB overlay where the meaning of "yellow" depends on the channel colours.
- **Ratio images [NEW, easy].** Display log₂(A/B) with a diverging map (e.g. cmocean `balance`, Crameri `vik`) centred at 0. Mask pixels where both ions are below the noise floor, or set alpha ∝ (A+B) so low-signal ratios fade out ("value-suppressing" style).
- **Hillshade / 2.5D relief [NEW, easy].** Use `matplotlib.colors.LightSource.shade(img, cmap, blend_mode='soft')` on a lightly smoothed ion image. This gives a glossy embossed look that strongly accents gradients. It is attractive for covers but **distorts perceived intensity**, so use it only for illustration. I found no peer-reviewed MSI precedent.
- **Isoline overlays on a colour-coded UMAP image.** Thin white contour lines of one ion over the hyperspectral colour image combine Q1 and Q4 into a single dense figure.

### Gaps
- I found no MSI paper using bivariate 2D colormaps for ion pairs.
- I found no MSI paper using hillshading or relief rendering of ion images.
- I found no user study on contour versus heatmap overlays beyond Sharman's qualitative argument.

---

## Q5. Segmentation/cluster maps rendered beautifully (soft membership, boundaries, SSC)

### Takeaway
- **Spatial shrunken centroids (SSC)** in Cardinal is the established *probabilistic* MSI segmentation. Its per-pixel class probabilities enable **soft-membership rendering**: colour = probability-weighted mix of cluster colours, and lightness or alpha = max probability. Uncertain boundaries then fade or blend instead of showing hard pixel edges.
- Combined with tree-aware colours (Q0), thin boundary lines, and a dark background, this is the core of the atlas aesthetic.

### Cited Findings
- **Bemis et al., "Probabilistic Segmentation of MS Images Helps Select Important Ions and Characterize Confidence in the Resulting Segments"**, *Mol. Cell. Proteomics* 15(5):1761–1772, 2016, DOI 10.1074/mcp.O115.053918.
  - Spatial shrunken centroids: model-based segmentation and classification with feature selection and a choice of segment number.
  - Used on renal cell carcinoma, pig fetus and rodent brain.
  - Sources: [PMC4858953](https://pmc.ncbi.nlm.nih.gov/articles/PMC4858953); [PDF](https://mallicklab.stanford.edu/pdfs/2016-Bemis-probabilistic_segmentation_mass_spectrometry_ms_images.pdf)
- **Cardinal** (Bemis et al., *Bioinformatics* 2015): an R/Bioconductor package for MSI statistics, Artistic-2.0. — [PDF](https://mallicklab.stanford.edu/pdfs/2015-Bemis-cardinal_r_package_statistical_analysis_mass.pdf)
- **Recent use.** A 2025 *Analyst* paper used Cardinal SSC on isotope-labelled duckweed and found 5 segments. — [RSC d5an00649j](https://pubs.rsc.org/en/content/articlehtml/2025/an/d5an00649j)
- **Centroid colouring.** Pixel colour = cluster-centroid colour from 3D UMAP (2025 study, 50 colours). — [PMC12130678](https://pmc.ncbi.nlm.nih.gov/articles/PMC12130678/)
- **Spatially-aware palette assignment** (Palo, Spaco), see Q0. — [PMC9272793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9272793/); [PMC10935509](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10935509/)

### Inferences
- **Soft-membership render [NEW, easy].** It needs probabilities, from SSC or from k-means with softmax(−d²/τ), or GMM.
  1. `rgb = Σₖ pₖ·colourₖ`, mixed in OKLab, then converted back.
  2. Multiply lightness by max(pₖ) or by entropy-based confidence.
  - Result: crisp cores and soft, darker transition zones. This is "honest" because it shows uncertainty.
- **Boundary lines [UPGRADE].** Compute a label-change mask (`labels != shift(labels)`). Draw it as 1-px semi-transparent white or dark lines over the cluster map. Or upsample the label map with nearest-neighbour ×4–8 and contour each label with smooth (Gaussian-blurred indicator, level 0.5) curves, which gives vector-like outlines.
- **Voronoi or superpixel styles [NEW, moderate].** SLIC superpixels on the embedding-colour image (skimage) followed by mean-colour fill give a stained-glass mosaic. This is decorative; I found no MSI paper using it.

### Gaps
- I found no MSI paper specifically about aesthetic rendering of segmentation maps (soft blends or superpixel mosaics).

---

## Q6. High-visual-impact recent (2020–2026) MSI figures and the techniques behind them

### Takeaway
The most striking recent MSI atlas figures come from **lipid atlas projects (La Manno/D'Angelo lab: uMAIA 2025, Lipizones 2025/26)**. They combine:
- very many clusters with hierarchically related colours,
- 3D registration to a common coordinate framework,
- dark or clean backgrounds.

Classic high-impact single images come from L\*a\*b\* t-SNE colouring (Abdelmoula 2016) and fusion (Van de Plas 2015). I could not systematically retrieve journal cover art.

### Cited Findings
- **Lipizones** (bioRxiv 2025; reportedly Nature 2026). 539 lipizones and a 3D atlas registered to the Allen CCF. EPFL press coverage was titled "New Brain Atlas Redraws Anatomical Boundaries". — [Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/); [News-Medical](https://www.news-medical.net/news/20260923/New-3D-lipid-atlas-maps-the-chemistry-of-mouse-brains.aspx)
- **uMAIA**, *Nat. Methods* 2025: a 4D zebrafish lipid atlas. — [PMC12446072](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12446072/)
- **Spatial metabolome/lipidome/glycome atlas of brain**, *Nat. Commun.* 2025 (s41467-025-59487-7). I did not retrieve the visualisation details. — [Nature](https://www.nature.com/articles/s41467-025-59487-7)
- **Liver multimodal MSI**, *Dev. Cell* 2024: 4-colour lipid overlays plus 3 µm SIMS overlays. — [Cell](https://www.cell.com/developmental-cell/fulltext/S1534-5807(24)00045-5)

### Inferences
- What makes these figures look good is (a) **many categories with structured colour** (tree or embedding palettes), rather than a few ions, and (b) **consistency of colour across panels** (dendrogram ↔ UMAP ↔ tissue ↔ 3D).
- SMILE-MSI's main missing piece is (a); see Q0.

### Gaps
- Cover-art searches (ACS Anal. Chem., JASMS) returned nothing usable, so no cover-art catalogue could be built.
- The Lipizones figure techniques are deferred to the other researcher.
- I could not access nature.com, bioRxiv full text, or PMC article bodies (egress-blocked), so most figure details are from abstracts and secondary pages.

---

## Q7. Perceptual best practices: colormaps, hotspot clipping, gamma, dark backgrounds

### Takeaway
- Use **perceptually uniform, CVD-safe sequential maps** (viridis/cividis, Crameri `batlow`/`lajolla`, cmocean) for single ions, and **never jet/rainbow**. This is explicitly argued for MSI by Knizer et al. 2022 (J. Mass Spectrom.) and generally by Crameri et al. 2020 (Nat. Commun.).
- Use hotspot/percentile clipping (SMILE-MSI has it) or quantisation such as TrIQ for contrast.
- Use diverging maps only for signed data (ratios, loadings).
- Use categorical palettes built in CIELAB/OKLab (Glasbey) or tree-aware schemes for clusters.

### Cited Findings
- **Knizer et al., "On the Importance of Color in Mass Spectrometry Imaging"**, *J. Mass Spectrom.* 57(12), 2022 (online Jan 2023), DOI 10.1002/jms.4898 (Muddiman and PNNL authors). Jet-like non-uniform gradients remain common in MSI and increase misinterpretation. They are hard for CVD viewers. The paper gives best practices and colormap resources. — [PNNL](https://www.pnnl.gov/publications/importance-color-mass-spectrometry-imaging)
- **Nuñez, Anderton & Renslow, "Optimizing colormaps with consideration for color vision deficiency…"**, *PLOS ONE* 2018: the origin of **cividis** (PNNL). — [arXiv 1712.01662](https://arxiv.org/pdf/1712.01662)
- **Crameri, Shephard & Heron, "The misuse of colour in science communication"**, *Nat. Commun.* 11:5444, 2020, DOI 10.1038/s41467-020-19160-7. Rainbow and red–green maps distort data. Scientific colour maps are perceptually uniform and CVD-accessible. They are freely available (fabiocrameri.ch/colourmaps, Zenodo). — [PMC7595127](https://pmc.ncbi.nlm.nih.gov/articles/PMC7595127); [UiO](https://mn.uio.no/ceed/english/research/publications/articles/2020/the-misuse-of-colour-in-science-communication.html)
- **CMasher** evaluates jet as "absolutely not CVD-friendly" and supplies extra perceptually uniform maps. — [arXiv 2003.01069](https://arxiv.org/pdf/2003.01069)
- **CVD prevalence**: about 8% of men and 0.5% of women. — [UiO blog](https://www.mn.uio.no/ceed/english/about/blog/2020/using-better-colours-in-science.html)
- **TrIQ** quantisation for MSI contrast. — [PeerJ](https://peerj.com/articles/cs-585)

### Inferences
- **Dark backgrounds** suit additive (fluorescence-like) composites and maps whose low end is black, such as magma/inferno or Crameri `lajolla` reversed. Light backgrounds suit subtractive/CMY and categorical cluster maps. Offering both in the figure editor is cheap.
- **Gamma.** A gamma or asinh stretch (`asinh(x/s)`) after percentile clipping often looks better than linear for lipid images with long-tailed intensities. Show the transform in the colorbar label. This is a general astronomy-imaging convention; no MSI source was found.
- **Crameri maps via `cmcrameri` [NEW, trivial].** Adding Crameri maps (`batlow`, `roma`, `vik`, `lajolla`, `oslo`) is a pip dependency, and cmocean is similar.

### Gaps
- I found no MSI-specific empirical study on gamma or asinh stretching, or on dark versus light backgrounds.

---

## Summary table: novelty and effort for SMILE-MSI

| Method | Key source | Visual impact | Effort (numpy/mpl) | Status |
|---|---|---|---|---|
| Tree Colors palette on hierarchical segmentation (+ shared across dendrogram/UMAP/tissue) | Tennekes & de Jonge 2014; Mertz & Kohlhammer 2024 | Very high (Lipizones look) | Low (~100 lines) | **NEW** (replaces 12-colour cycle) |
| Cluster colour from centroid in 3D UMAP → OKLab | arXiv 2410.07125; PMC12130678 | High | Low | **NEW** |
| Spatially-aware palette swap (adjacent clusters contrast) | Palo 2022; Spaco 2024 | Medium-high | Moderate | **NEW** |
| 3D UMAP/t-SNE → CIELAB/OKLab pixel colouring with gamut fit | Abdelmoula 2016; U-CIE 2022 | Very high | Low-moderate | **UPGRADE** (HSV wheel now) |
| 2D embedding → a\*b\* hue, L\* from TIC/PCA-1 | Hu et al. 2022 review | High | Low | **NEW** |
| Soft-membership cluster rendering (probability blend, confidence-darkened) | Bemis 2016 (SSC probabilities) | High | Low | **NEW** |
| Optimised N-channel overlay palettes; max/screen/OKLab blends | psudo 2024 | High | Moderate | **UPGRADE** (additive exists) |
| Iso-intensity contour maps on histology (ions or NMF components) | Sharman JASMS 2023 | High, honest | Low | **UPGRADE** (mask contours exist) |
| Pan-sharpening / regression fusion with optical image | Van de Plas 2015; Tarolli 2014 | High | Moderate | **NEW** |
| Bivariate 2-ion colormap with 2D legend | Ware et al. EuroVis 2020 (general) | Medium-high | Low | **NEW** (no MSI precedent found) |
| log-ratio images, diverging map, alpha by signal | Cheng et al. eLife 2024/25 | Medium | Low | **NEW** |
| Hillshade / relief rendering | matplotlib LightSource (no MSI precedent found) | High (decorative) | Trivial | **NEW** (illustrative only) |
| Crameri / cmocean colormaps | Crameri 2020; Knizer 2022 | Medium | Trivial | **NEW** |
| Diffusion virtual staining / DL super-resolution | UCLA arXiv 2411.13120; Adv Sci 2025 | Very high | Very high (GPU/models) | Out of scope |
