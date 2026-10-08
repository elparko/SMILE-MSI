# Colour the Tree to Get the Lipizones Look

**Bottom line.** The Lipizones look does not depend on a renderer. It comes from five design decisions, and SMILE-MSI could reproduce most of them at low cost:

- Colours come from the cluster hierarchy, not from a cycling palette.
- Each pixel is drawn as a separate dot on a black background.
- The same colours appear in every linked view.
- Coarse-to-fine animations show the tissue dividing level by level.
- Smoothed marching-cubes meshes are exported as self-contained Plotly HTML.

All of it is matplotlib and Plotly code in the authors' BSD-3 package EUCLID ([GitHub EUCLID](https://github.com/lamanno-epfl/EUCLID); [PyPI euclid-msi](https://pypi.org/project/euclid-msi/0.0.6/)).

SMILE-MSI already has most of the base this needs. `gui/segment.py` builds a hierarchical clustering (HCA) tree and cuts it live with a Detail slider. `multivariate.py::_coords_rgb` gives the UMAP plot and the tissue map one shared colour function. `volume3d.py` builds registered serial-section volumes. The biggest single gap is the colour of segments. Segment colours come from a 12-colour palette that repeats (`palettes.SEGMENT`, indexed `PALETTE[cl % len(PALETTE)]`), so similar segments get unrelated colours. Above 12 segments, the code's own comment says the key "repeats".

The best return per hour of work is:

1. **Tree-aware palette.** Replace the cycling palette with a tree-aware one (EUCLID's colour chain, or Tennekes and de Jonge's Tree Colors).
2. **Dot-mosaic export.** Add a dot-mosaic, black-background export.
3. **Perceptual embedding colours.** Move the UMAP colour wheel from HSV to OKLab or CIELAB.
4. **Animations.** Add "splitter" and slice-sweep movies.

Each is a few days of work at most. Second-tier additions take more effort and add new capability:

- 3D segment-label meshes.
- A "Chemistry" tab: Kendrick mass defect (KMD) scatter, chain-length × unsaturation grids, and lipid-ontology enrichment.
- A force graph of ion co-localisation with thumbnail nodes.
- A web or SpatialData export.

Full Allen-atlas registration, deep-learning virtual staining and native VR are high-cost, specialised features. They belong at the bottom of the roadmap.

## Lipizones gets its look from five reproducible choices

The reference paper is Fusar Bassini, Schede, Capolupo, …, D'Angelo and La Manno, "The lipidomic architecture of the mouse brain". It was a bioRxiv preprint on 14 October 2025 (DOI 10.1101/2025.10.13.682018) ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1); [preLights](https://prelights.biologists.com/highlights/the-lipidomic-architecture-of-the-mouse-brain/)). It has since appeared in *Nature*, with press coverage dated 23 September 2026 ([News-Medical](https://www.news-medical.net/news/20260923/New-3D-lipid-atlas-maps-the-chemistry-of-mouse-brains.aspx)).

> **Citation flag.** The *Nature* DOI 10.1038/s41586-026-11050-0 and the volume and pages (658:205–217) come from search-result metadata. No researcher could open nature.com. Two of the four note sets mark the DOI as unverified. Confirm it before you cite it ([Nature listing](https://www.nature.com/articles/s41586-026-11050-0)).

**The study.** The authors imaged **172 lipids by MALDI-MSI across 109 coronal sections from 11 mice**, about **7–7.5 million pixels**, and registered the data into the Allen CCFv3 ([Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/)). They grouped the pixels into **539 "lipizones"** with a four-level hierarchy:

| Level | Name | Count |
|---|---|---|
| 3 | classes | 8 |
| 5 | subclasses | 31 |
| 8 | supertypes | 222 |
| terminal | lipizones | 539 |

The first split separates grey-matter-rich from white-matter-rich tissue ([preLights](https://prelights.biologists.com/highlights/the-lipidomic-architecture-of-the-mouse-brain/)). The paper reports two findings that a visualisation tool should be able to show. White matter is "a rich patchwork" of biochemical zones. Some lipizones link neuronal cell bodies to the regions their axons target ([Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/)). Peak counts differ between the preprint and the *Nature* version (one figure says "173 non-redundant lipids"). Figure numbering also differs between versions.

The useful fact for SMILE-MSI is that the plotting code is public. It is in EUCLID (BSD-3, `euclid-msi==0.0.6`), which depends on matplotlib, seaborn, Plotly, openTSNE and scikit-image ([PyPI JSON](https://pypi.org/pypi/euclid-msi/json)). One researcher read `euclid_msi/plotting.py` and `clustering.py` directly, so the recipes below come from the source code, not from guesses about the figures ([GitHub EUCLID](https://github.com/lamanno-epfl/EUCLID)).

**Choice 1: the clustering is a divisive binary tree, not Ward linkage.** `learn_euclid_clustering` splits recursively. At each node it does the following:

1. Fits a local NMF.
2. Joins the local embedding with the parent-level embedding (down-weighted ÷1.5) and the global embedding (÷2).
3. Runs KMeans with K = 60.
4. Merges those 60 clusters into two groups with backSPIN.

A split is kept only if it passes all three gates:

- at least 2 differential lipids, with fold change ≥ 0.2 and p < 0.05;
- spatial continuity across sections;
- XGBoost separability of at least 0.6.

The splitting stops below 150 voxels or beyond depth 15. The EPFL MLIBRA project describes the same idea as "iterative bipartitioning plus a generative probabilistic model" ([SDSC MLIBRA](https://www.datascience.ch/projects/mlibra)).

**Choice 2: colours come from the hierarchy and from chemical similarity.** `assign_cluster_colors` works on each top-level division separately:

1. It computes cluster centroids on z-scored lipid profiles.
2. It orders the clusters along a **greedy nearest-neighbour chain**.
3. It maps the cumulative chain distance (0–1) onto a colormap specific to that division. The colormaps cycle through RdYlBu, terrain, PiYG, cividis, plasma, PuRd, inferno and PuOr.
4. It boosts the saturation of every other cluster by up to 70%, so neighbouring siblings stay distinct.

As a result, hundreds of colours still read as anatomy: related zones form families of related hues ([GitHub EUCLID](https://github.com/lamanno-epfl/EUCLID)).

**Choice 3: the rendering is pointillist on pure black.** `plot_mosaic` draws a section as a scatter of round, coloured dots with visible gaps on `facecolor='black'`, with the axes turned off. The README banner shows it with a coronal "glass-brain" outline inset, a locator box, and white dashed boundary annotations ([lbae README assets](https://github.com/lamanno-epfl/lbae)). The dots hide the blocky MSI pixel grid and give a stained-glass, nebula-like texture.

**Choice 4: the figures move.** EUCLID has three movie functions:

- `make_splitter_movie` animates levels 1→6. Each pixel's colour is interpolated from its level-k ancestor colour to its level-k+1 colour (10 frames per level, 5 fps, MP4), so the brain visibly "divides" from 2 colours to 2ⁿ.
- `make_lipizone_rain_movie` reveals one colour group at a time.
- `create_lipids_movie` flies through 3D-interpolated lipid volumes in greyscale on black ([GitHub EUCLID](https://github.com/lamanno-epfl/EUCLID)).

**Choice 5: the views are linked and shared through the web.** The 3D pipeline is a series of plain steps ([GitHub EUCLID](https://github.com/lamanno-epfl/EUCLID)):

1. Voxelise each lipizone on a 528×320×456 grid (the CCF at 25 µm).
2. Apply a 3D binary closing with a 12-voxel cube.
3. Keep the 4 largest connected components.
4. Mirror left and right.
5. Gaussian-smooth with σ = 2.5.
6. Run skimage marching cubes at level 0.5.
7. Render as Plotly `Mesh3d` at opacity 0.8 and save as a self-contained HTML file.

Lipid volumes use `go.Volume` with about 15 isosurfaces in Inferno, inside a translucent grey brain "root" shell at opacity 0.1. The other views come from the same data:

- A Plotly treemap on the `plotly_dark` template.
- t-SNEs of pixels and of lipizone centroids. In the t-SNE of pixels, points are coloured both by lipizone and by official Allen colours.
- "Staircase" heatmaps: columns are ordered by optimal leaf ordering, rows by argmax, in greyscale clipped to the 2–98th percentiles.
- Lipid-triplet RGB galleries. Each channel gets a seeded random colour with components in [0.3, 1.0] rather than pure R/G/B. Each lipid name is printed in its channel's colour.

The Lipid Brain Atlas Explorer (lbae-v2.epfl.ch) is a Python Dash app. It offers three-lipid RGB, an Allen annotation overlay, volcano plots, treemap navigation, per-lipizone "ID cards" and 3D reconstructions ([lbae README](https://github.com/lamanno-epfl/lbae)).

**What is not known.** Nobody could read the *Nature* figure legends. The published figures may differ from EUCLID 0.0.6, whose authors call it "very much work in progress". The authors have not written about their design choices anywhere we could find. The "why it's striking" reading in this section is therefore inferred from the code and the README images.

**The idea underneath.** What makes the look, as distinct from the brain registration, is categories coloured by their structure and the same colours in every panel. Older work shows this works elsewhere. A color-coded mouse-kidney dendrogram mapped back onto tissue predates Lipizones. Its source is search-derived and unconfirmed (it is probably PMC2742436, Fig. 4) ([PMC2742436 Fig. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC2742436/figure/F4)). The same lab's earlier zebrafish atlas is uMAIA (*Nature Methods* 2025) ([PMC12446072](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12446072/)).

## Perceptual colour encoding is the cheapest upgrade with the biggest payoff

### Tree palettes formalise what EUCLID does informally

The published, general algorithm for this look is **Tree Colors** (Tennekes and de Jonge, *IEEE TVCG* 2014). It recursively divides the hue circle among the subtrees of a hierarchy in HCL space, and varies chroma and luminance with depth. A user survey found that these colours "unveil tree structure" even in views that are not hierarchical ([paper PDF](https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/2072_20tvcg12-tennekes-2346277.pdf)).

Mertz and Kohlhammer (IEEE VIS 2024) refined it with three rules ([arXiv 2407.08287](https://arxiv.org/pdf/2407.08287)):

- Reach maximum chroma at the leaves while staying in gamut.
- Interpolate lightness and chroma linearly.
- Permute sibling hues, so a smooth rainbow ramp does not suggest an order that isn't there.

**Recommended recipe for SMILE-MSI.** `gui/segment.py` already holds a scipy linkage tree. The steps are:

1. Give each node a hue interval, split among its children in proportion to their leaf counts, with small gaps between siblings.
2. Take hue from the interval midpoint, and raise chroma with depth.
3. Convert through OKLab to sRGB, clipping to gamut.
4. Optionally order the leaves within a branch along a principal axis in UMAP space, as EUCLID does, so colour gradients follow real chemical gradients.

This is about 100 lines of numpy. It also makes the Detail slider readable: splitting a segment produces two shades of its parent's hue instead of two unrelated colours.

**The opposite problem.** Palo ([PMC9272793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9272793/)) and Spaco (*Patterns* 2024; CVD-aware, available in Python) ([PMC10935509](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10935509/)) deliberately give *spatially adjacent* clusters contrasting colours. A Spaco-style swap could run after the tree assignment as an optional final step.

### The UMAP colour wheel should move from HSV to a perceptual space

SMILE-MSI's `_coords_rgb` maps a 2D embedding to an HSV wheel: hue is the angle, saturation is the radius, and value is fixed at 1. At full value, yellows and cyans look far brighter than blues, so the map shows false "hotspots".

The lineage of embedding-to-colour methods runs as follows:

| Work | What it did | Source |
|---|---|---|
| Fonville et al., 2013 | First MSI "hyperspectral overview" colouring | [ACS](https://pubs.acs.org/doi/10.1021/ac302330a) |
| Abdelmoula et al., *PNAS* 2016 | 3D t-SNE mapped to **L\*a\*b\***; the stained-glass map of tumour subpopulations was linked to patient survival | [PMC5087072](https://pmc.ncbi.nlm.nih.gov/articles/PMC5087072) |
| Smets et al., 2019 | Found UMAP visually competitive with t-SNE at about 4× the speed | [ACS](https://pubs.acs.org/doi/10.1021/acs.analchem.8b05827) |
| U-CIE, 2022 | Formalised 3D UMAP → CIELAB, with the point cloud fitted to the sRGB gamut | [PMC9387205](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9387205/) |

> **Citation flag.** Whether Fonville used CIELAB is unconfirmed. One later review describes the original as plain RGB.

Ranked by how perceptually sound they are, the options are:

1. A 3D embedding mapped into OKLab or CIELAB, fitted to the gamut.
2. A 2D embedding mapped to a\*/b\*, with lightness taken from total ion current (TIC) or PCA-1, so hue carries chemistry and lightness carries anatomy (described in the *Advanced Science* 2022 review ([Wiley](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202203339))).
3. A fixed-chroma CIELCh disk, as used by the 2026 MIA system ([arXiv 2606.00874](https://arxiv.org/pdf/2606.00874)).
4. HSV, which is what SMILE-MSI uses now.

OKLab (Ottosson 2020) fixes CIELAB's blue hue shift, and its colour mixing is cleaner ([Wikipedia](https://en.wikipedia.org/wiki/Oklab_color_space)). Swapping HSV for OKLCh costs about 20 lines. A gamut-fitted 3D mode costs about 60.

**Two caveats.** Gildenblat and Pahnke (bioRxiv 2025) report that UMAP colouring missed most small amyloid plaques. They propose "truthful" alternatives, including TOP3: the three largest intensities per pixel, converted from LAB to RGB ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.18.643852v4.full); [msi_visual](https://github.com/jacobgil/msi_visual)). TOP3 is known only from a search snippet, and it is unconfirmed whether `msi_visual` is the same project as the paper. Sarycheva et al. separately argue for "structure-preserving and perceptually consistent" mappings ([Skoltech](https://skoltech.ru/en/archived-news/a-new-perceptually-consistent-method-for-msi-visualization)). Their DOI was not retrieved.

### Multi-ion composites: borrow from astronomy and multiplexed imaging

There are two compositing paths, and they differ:

- The interactive overlay in `imaging.py::rgb_overlay` is limited to three ions.
- The export path `gui/exportdialog.py::_compose_overlay_rgb` already sums N tinted channels and then clips.

Sum-and-clip saturates to white where three or more channels overlap. Better options:

- **Blend modes.** Offer "max", "screen" (`1−∏(1−cᵢxᵢ)`), or OKLab-averaged blends.
- **Optimised palettes.** psudo (*Computer Graphics Forum*, EuroVis 2024) optimises N-channel palettes to maximise perceptual separation and penalise confusing overlaps. In a 150-person study, users were more accurate with its palettes ([Harvard VCG](https://vcg.seas.harvard.edu/publications/20240610-psudo)).
- **Astronomy techniques.** None appear in any MSI paper found, so they would set SMILE-MSI apart:
  - the Lupton arcsinh stretch, which keeps faint structure without blowing out bright regions ([PASP 2004](https://www.doi.org/10.1086/382245));
  - Rector's "representative colour" practice: order hues chromatically by a physical variable (here, m/z or lipid class), building colour per group of channels before combining, rather than dumping channels into R, G and B ([arXiv 1703.00490](https://arxiv.org/pdf/1703.00490)).

EUCLID's lipid-triplet gallery reaches the same look with seeded non-primary hues and lipid names printed in their channel colours.

### Honest overlays: contours, bivariate maps, ratios and soft membership

**Iso-intensity contours.** Sharman et al. (*JASMS* 2023) argue that alpha-blended heatmaps "obscure subtle quantitative differences". They draw ion or NMF iso-intensity contours on whole-slide histology instead ([PMC10787559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10787559)). `export.py` already calls `ax.contour` for masks, so iso-intensity contours at the 50th, 75th and 90th percentiles are a small extension.

**Ratio images.** Cheng et al. (*eLife* 2024/25) compute ratio images for every metabolite pair and show biology that single-ion heatmaps hide ([eLife](https://elifesciences.org/articles/96892)). The display is log₂(A/B) on a diverging map, with opacity proportional to signal.

**Two-ion bivariate maps.** These use a 2D colour lookup table with a square legend. Ware et al. found that pairing a high-saturation map with a low-saturation one works best ([EG diglib](https://diglib.eg.org/handle/10.2312/evs20201047)). No MSI precedent was found.

**Soft membership.** Spatial shrunken centroids (Bemis et al., *MCP* 2016) give per-pixel class probabilities ([PMC4858953](https://pmc.ncbi.nlm.nih.gov/articles/PMC4858953)). Those probabilities allow a render where cores are crisp and uncertain borders are blended and darkened. SMILE-MSI can approximate this with a softmax over k-means distances.

**Hillshaded relief.** `matplotlib.colors.LightSource` gives an attractive embossed look, but it distorts perceived intensity and has no MSI precedent. Treat it as decoration only.

### Colormap hygiene

Knizer et al. (*J. Mass Spectrom.* 2022) document how often jet-like maps cause misreading in MSI ([PNNL](https://www.pnnl.gov/publications/importance-color-mass-spectrometry-imaging)). Crameri et al. (*Nat. Commun.* 2020) supply perceptually uniform maps that are safe for colour-vision deficiency (CVD) ([PMC7595127](https://pmc.ncbi.nlm.nih.gov/articles/PMC7595127)).

SMILE-MSI's colormap lists (`palettes.SEQUENTIAL_CMAPS` and the ion and export combo boxes) still offer `turbo`, a rainbow map, and offer no Crameri or cmocean maps. Adding `cmcrameri` (batlow, lajolla, vik, roma) costs one dependency.

## 3D atlas renders are within reach of SMILE-MSI's existing volume engine

**What `volume3d.py` does now.** It stacks, rigidly registers and resamples **one ion at a time** into a (Z, H, W) volume. It offers orthoslices, MIP, and an optional pyqtgraph `GLVolumeItem` render. Optional B-spline refinement goes through `registration.py`. A search of the code found no meshes, no label volumes, no multi-ion volumes and no export.

**The reusable 3D-MSI figure.** Most 3D-MSI figures published over two decades follow one template: coloured isosurfaces or segment labels inside a translucent organ shell. Examples:

- Trede et al. (*Anal. Chem.* 2012), a 3D-segmented kidney ([Lübeck record](https://research.uni-luebeck.de/de/publications/exploring-three-dimensional-matrix-assisted-laser-desorptionioniz/)).
- Paine et al., a medulloblastoma segmentation in a reconstructed brain built from 223 sections (3.3 TB) ([Sci. Rep. 2019](https://www.nature.com/articles/s41598-018-38257-0)).
- Andersson et al. (*Nat. Methods* 2008), the classic coloured protein volumes ([DOI](https://doi.org/10.1038/nmeth1160)).

> **Citation flag.** The Andersson 2008 citation is recalled from memory and was not re-verified. The same is true of Sinha 2008, Palmer 2015 and Thiele 2014. Verify each before citing.

**A label-volume pipeline from existing dependencies.** EUCLID's mesh pipeline (described in the first section) uses only scikit-image and scipy, which SMILE-MSI already has. The workflow:

1. Cluster all voxels in the stacked volume, or apply a 2D segmentation tree to every section.
2. Close, smooth and run marching cubes on each label.
3. Draw the meshes with pyqtgraph `GLMeshItem` inside a shell derived from TIC, or write them to Plotly HTML.

This reproduces most of the Lipizones 3D look for *any* organ, with no atlas. Two related additions:

- A per-segment point cloud through `GLScatterPlotItem`, the Fig. 1g look.
- Multi-ion RGBA volumes, which the same GL path accepts.

**Section spacing.** Sections are often 100–150 µm apart while pixels are 10–50 µm, so volumes look jagged. MetaVision3D added an interpolation module mainly to make rendering continuous ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.27.568931.full.pdf)). Its journal publication (possibly *Advanced Science*, 2026) is unverified. In SMILE-MSI, anisotropic `scipy.ndimage.zoom` plus smoothing is enough. The Oetjen et al. benchmark 3D datasets (*GigaScience* 2015, public imzML) are the right test data ([PMC4418095](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4418095/)).

**The full Allen-atlas look.** The steps are:

1. Load the atlas with `brainglobe-atlasapi`.
2. Register sections with STalign (LDDMM, 3D-to-2D against CCFv3, pip-installable), which the Lipizones paper used ([GitHub STalign](https://github.com/JEFworks-Lab/STalign)), or with the earlier Abdelmoula 2014 pipeline ([TU Delft](https://pure.tudelft.nl/portal/en/publications/automatic-registration-of-mass-spectrometry-imaging-data-sets-to-the-allen-brain-atlas(8fc18c3f-b2be-4fb5-9431-f50851215417).html)).
3. Render with brainrender, which exports figures and videos ([PubMed 33739286](https://pubmed.ncbi.nlm.nih.gov/33739286/)).

Robust registration is the costly step, at roughly 1–2 weeks. It also only applies to mouse and rat brain, so it should be an optional "brain atlas mode", not a core feature.

**Immersive and physical outputs.** These are barely published:

- One 2026 *Advanced Science* paper exports 3D MALDI volumes of 3D cell-culture models to a mixed-reality tool. Its headset and software are unconfirmed ([Wiley](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202516098)).
- No peer-reviewed 3D-printed MSI map was found.

The pragmatic route is glTF, PLY or STL mesh export, which works with Blender, PowerPoint, WebXR and printers. Native OpenXR in a Qt app is not worth building.

**Surface draping.** Autofocusing AP-MALDI of non-flat samples (Kompauer et al., *Nat. Methods* 2017) drapes ion images over measured topography ([Thermo note](https://assets.thermofisher.com/TFS-Assets/CMD/Technical-Notes/tn-000659-smaldi-msi-tn000659-en.pdf)). The *Nature* URL in the notes (nmeth.4513) does not match the cited DOI 10.1038/nmeth.4433. Check it before citing. A cheaper relative is pyvista `warp_by_scalar` of a single section, which is visually striking but decorative.

## Chemical-space and network views turn ion lists into relationships

SMILE-MSI already has a co-localisation matrix, an ion dendrogram, NMF, sum-composition annotation and a lipid class tree. Every view in this section reuses those, so most are low-effort additions with high impact.

### Ion-centric "map of maps"

**Force graph of ions.** Wüllems et al. (*BMC Bioinformatics* 2019) build a graph of m/z images linked by spatial similarity and detect modularity communities as candidate functional networks ([PMC6549267](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549267/)). Proposed view: a force-directed graph whose nodes are **ion-image thumbnails**, coloured by Leiden community or lipid class.

**Co-localisation metric.** ColocML (*Bioinformatics* 2020; often miscited as *BMC Bioinformatics*) asked 42 experts to rank 2,210 image pairs. **Cosine similarity after median thresholding** nearly matched the best deep model (Spearman 0.794 vs 0.797) ([PMC7214035](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7214035/)). It is a validated drop-in metric for SMILE-MSI's co-localisation code.

**UMAP of ions.** DeepION (*Anal. Chem.* 2024) learns embeddings of whole ion images ([PMC10918617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10918617/)). Its architecture details came from an unverified third-party page. A cheap baseline (UMAP on 1 − coloc, with thumbnails as points) gives a "map of maps" gallery without PyTorch.

### Lipid chemical space

**Referenced KMD.** Referenced Kendrick mass defect (RKMD; Richardson et al., *Anal. Chem.* 2022) filters and rebuilds images by lipid class, chain length and unsaturation ([NSF PAR](https://par.nsf.gov/biblio/10352372)). Its DOI was not retrieved. Proposed views:

- A KMD scatter of every annotated peak, coloured by the region where it is most enriched (from the existing AUC results).
- Lasso selection on the scatter that builds a summed image.
- A class-wise **carbons × double-bond heatmap** per region. It is the standard "lipidome fingerprint" layout of bulk lipidomics, and SMILE-MSI's annotator already provides carbon:double-bond (C:DB) counts.

**Ontology enrichment.** LION/web links more than 50,000 lipids to classification, physical-property, function and organelle terms, with target-list and ranking modes ([PMC6541037](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6541037/)). Lipid Mini-On offers similar enrichment ([PNNL](https://www.pnnl.gov/publications/lipid-mini-mining-and-ontology-tool-enrichment-analysis-lipidomic-data)). Running the ranking mode on per-region AUC values and drawing an enrichment sunburst turns "these 40 ions" into a statement like "PUFA-PE and membrane fluidity are enriched in white matter". This is the highest scientific value in the section, at medium effort.

### Atlas-style relational views

These follow Lipizones directly:

- **Territory × lipid-programme clustermap.** Rows are segments in tree leaf order. Columns are lipids grouped by NMF programme, then class, then C:DB, with side tracks for colour and class. This combines SMILE-MSI's *ion* dendrogram with the new *pixel* tree (NMF-programme precedent: [PMC11466421](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466421/)).
- **Territory similarity graph.** Segment centroids are drawn over a faint section image and linked when their lipid profiles are similar. This shows directly the paper's finding that distant regions can share one lipid identity ([Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/)).

**Distance-to-boundary heatmap.** This is kymograph-style: distance from an ROI boundary on one axis, lipids on the other. It has precedent in a medulloblastoma study that found a cancer-like lipid gradient extending **1.2 mm beyond the tumour border** ([PubMed 33651938](https://pubmed.ncbi.nlm.nih.gov/33651938/)). The full citation was not retrieved. The view needs only `distance_transform_edt` on existing masks.

**Lower priority.** Pathway maps with an ion image at every node lack a standard tool ([Sun et al. PNAS 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6320512)). Radar plots and chord diagrams have no MSI literature behind them. Mirror spectra and per-region "spectral barcodes" are useful additions but have no MSI precedent.

## Shareable, interactive outputs are SMILE-MSI's widest open gap

**What the code lacks today.** A search of the code found no GIF or MP4 export, no HTML export, no OME-Zarr or SpatialData export, and no napari or Vitessce bridge. `cubestore.py` writes a Zarr v2 ZipStore, but it is an internal cache.

**Where the field has converged.** Sharing in spatial omics runs on three pieces:

- **Storage:** OME-Zarr or SpatialData, served as static files.
- **Rendering:** client-side, via Viv or Vitessce (WebGL) or OpenSeadragon (Minerva, TissUUmaps).
- **Showcases:** Vitessce already shows 3D imaging MS in linked views ([Nature Methods 2024](https://www.nature.com/articles/s41592-024-02436-x)).

MSI is a latecomer. A 2025/26 review calls OME-Zarr use for MSI "scarce" ([PMC12879364](https://pmc.ncbi.nlm.nih.gov/articles/PMC12879364/)). Thyra (Maastricht, bioRxiv January 2026, MIT licence) converts imzML and vendor formats into SpatialData Zarr ([GitHub Thyra](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra)). It needs Python 3.12 or later, and its claim of Vitessce integration appears only in the preprint.

**Recommended export.** Write only the reduced products:

- the ion-image stack;
- segmentation labels;
- ROIs as shapes;
- an AnnData table with UMAP coordinates and cluster IDs.

Then emit a Vitessce config. A self-contained Plotly HTML file is lower friction still for small datasets. It is also how EUCLID ships its meshes.

**Interaction ideas.**

- *Live spectrum on brushing.* SMILE-MSI already has the hardest part: a colour wheel shared between UMAP and tissue, plus lasso selection in `gui/scatter.py`. Missing is a live panel showing the mean and difference spectrum of a brushed selection, where clicking a peak loads its image.
- *Movable lens.* A movable circular lens (pyqtgraph `CircleROI`) would swap to optical, a second lipid or the UMAP colours inside the circle. No MSI precedent was found, so it would be distinctive.
- *Story export.* Minerva-style narrative waypoints (saved view + caption + channel set) can be exported as an in-app slideshow or as MP4 ([PMC7989801](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7989801/)).

**Cross-field ideas, ranked by usefulness:**

1. *Astronomy colour craft* (covered in the colour section): arcsinh stretch, representative colour, and a class-based "Hubble palette" preset. Label it as representative colour per Rector et al.
2. *IMC/Xenium-style rendering.* Paint flat-coloured segment polygons over high-resolution optical images, building on the existing `singlecell.py`.
3. *Pseudo-H&E.* A Beer–Lambert rendering of two lipid channels. It is deterministic and must be labelled as not a stain prediction. This technique is from prior knowledge and is unsourced.
4. *Sonification.* No spatial MSI sonification exists. The nearest are an FT-ICR feasibility study ([Wiley RCM](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/rcm.10047?af=R)) and astronomy datacube work ([arXiv 2306.10126](https://arxiv.org/pdf/2306.10126)). It is a cheap outreach novelty with little analytical value.

**Virtual staining.** Diffusion-model virtual staining is the high-impact frontier. A model trained on MALDI imaging of unstained human kidney matched PAS histology in blind tests, despite about 10× larger pixels (*Science Advances* 2025, eadv0741) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/pmid/40749063); [arXiv 2411.13120](https://arxiv.org/abs/2411.13120v1)). It needs GPU training and is out of scope. Classical pan-sharpening and regression fusion with optical images are in scope at medium effort. Van de Plas et al. (*Nat. Methods* 2015) predicted ion images at ≥10× finer resolution ([PMC4382398](https://pmc.ncbi.nlm.nih.gov/articles/PMC4382398)). Label any such output as *predicted*, with a cross-validated R².

## Prioritised SMILE-MSI roadmap

**Ranking method.** Impact means visual and scientific payoff relative to the Lipizones aesthetic. Effort is developer time for this codebase: Low = days, Med = 1–3 weeks, High = new stack or ML. These estimates are the researchers' judgement, not sourced figures. Status was checked against the code on 2026-10-07:

- **HAVE:** works today.
- **UPGRADE:** a weaker version exists.
- **NEW:** absent.

| # | Feature | Status | Impact | Effort | Where in SMILE-MSI | Basis |
|---|---|---|---|---|---|---|
| 1 | Tree-aware segment palette (Tree Colors / EUCLID colour chain; division colormaps, alternating saturation), replacing the 12-colour cycle | UPGRADE | Very high | Low | `gui/segment.py` (`PALETTE[cl % len]`), `palettes.SEGMENT`, `gui/exportdialog.py` L1794, `gui/jointseg.py` | Tennekes 2014; Mertz 2024; EUCLID |
| 2 | Same colours across HCA tree, ion/pixel dendrogram, UMAP scatter, tissue map and legend | UPGRADE | Very high | Low | `gui/segment.py` tree panel, `gui/dendrogram.py`, `umapstudio.py` | Lipizones Fig. 1 (t-SNE + tree + map) |
| 3 | Dot-mosaic ("pointillist") render on black + glass-outline locator inset | NEW | Very high | Low | `figedit.py` / `gui/figeditor.py`, `export.py` | EUCLID `plot_mosaic`; lbae banner |
| 4 | Perceptual embedding colour: OKLCh wheel instead of HSV; 3D UMAP → OKLab gamut-fit mode; L from TIC/PCA-1 | UPGRADE | High | Low | `multivariate.py::_coords_rgb`, `umapstudio.py` | Abdelmoula 2016; U-CIE 2022; Hu 2022 |
| 5 | Splitter / rain / m/z-sweep / slice fly-through movies (MP4/GIF via imageio-ffmpeg) | NEW | High | Low | `gui/segment.py`, `gui/stack3d.py`, `volume3d.py` | EUCLID movie functions |
| 6 | Lipid-triplet section gallery: non-primary seeded hues, channel-coloured names, zero spacing; arcsinh stretch; max/screen/OKLab blends | UPGRADE | High | Low | `gui/exportdialog.py::_compose_overlay_rgb`, `imaging.rgb_overlay` (3-ch), `gui/montage.py` | EUCLID `plot_lipids_rgb_grid`; Lupton 2004; psudo 2024 |
| 7 | Treemap/sunburst/icicle of the segmentation hierarchy (dark template) | NEW | High | Low | `gui/segment.py` (linkage exists) | EUCLID treemap; LBAE |
| 8 | Territory × lipid-programme "staircase" clustermap with class/C:DB tracks | NEW | Very high | Low–Med | `multivariate.py` (NMF), `ionembed.py`, `hierstats.py` | EUCLID `plot_sorted_heatmap`; Pathirage 2024 |
| 9 | 3D segment labels: point cloud + marching-cubes meshes in a TIC shell; Plotly HTML export | NEW | Very high | Med | `volume3d.py` (single-ion only today), `gui/stack3d.py` | EUCLID 3D pipeline; Trede 2012 |
| 10 | Multi-ion RGBA volumes + per-ion isosurfaces; Z-interpolation | NEW | High | Low–Med | `volume3d.py::gl_volume_item`, `build_channel_volume` | Vos 2021; MetaVision3D |
| 11 | Soft-membership segment render + 1-px boundary lines | NEW | High | Low | `gui/segment.py`, `export.py` | Bemis 2016 (SSC) |
| 12 | Iso-intensity contours of ions/NMF components on optical/histology | UPGRADE | High | Low | `export.py` (`ax.contour` on masks), `gui/optical.py` | Sharman JASMS 2023 |
| 13 | Coloc force graph with thumbnail nodes + ion-image UMAP gallery; median-threshold cosine metric | NEW | Very high | Low–Med | `ionembed.py`, `gui/dendrogram.py`, coloc ranking | Wüllems 2019; ColocML 2020; DeepION 2024 |
| 14 | Chemistry tab: KMD/RKMD scatter → lasso image; carbons × DB grids per region | NEW | High | Low | `annotate.py`, `lipiddb.py`, `gui/lipidtree.py` | Richardson 2022 |
| 15 | Territory similarity graph over the section | NEW | High | Low | `gui/segment.py` | Lipizones "postal code" finding |
| 16 | Divisive bipartite segmentation (recursive k=2, AUC/FDR gate, min size) | NEW | High | Med | `gui/segment.py`, region AUC/FDR code | EUCLID; MLIBRA |
| 17 | Brush → live mean/difference spectrum; movable spectral lens | UPGRADE / NEW | High | Low | `gui/scatter.py` (lasso exists), ion view | Minerva; Vitessce |
| 18 | Bivariate 2-ion map with square legend; log-ratio images | NEW | Med–High | Low | `gui/exportdialog.py`, `figedit.py` | Ware 2020; Cheng eLife 2024/25 |
| 19 | Distance-to-boundary lipid heatmap (kymograph) | NEW | Med–High | Low–Med | `spatial.py`, ROI masks | PMID 33651938 |
| 20 | LION-style ontology enrichment sunburst per region/community | NEW | High | Med | region AUC results, `lipiddb.py` | LION/web 2019 |
| 21 | Crameri/cmocean colormaps; demote `turbo` | NEW | Med | Trivial | `palettes.SEQUENTIAL_CMAPS`, `gui/ion.py` | Crameri 2020; Knizer 2022 |
| 22 | Self-contained HTML export; SpatialData/OME-Zarr + Vitessce config | NEW | High (sharing) | Med | `export.py`; `cubestore.py` is cache-only | Vitessce 2024; Thyra 2026 |
| 23 | Mesh export (glTF/PLY/STL) for Blender, VR, 3D printing | NEW | Med | Low (after #9) | `volume3d.py` | Iakab 2026 |
| 24 | Pan-sharpening / regression fusion with optical (labelled "predicted") | NEW | High | Med | `registration.py`, `gui/optical.py` | Van de Plas 2015 |
| 25 | Spaco-style adjacency swap; psudo palette optimisation | NEW | Med | Med | `palettes.py` | Spaco 2024; psudo 2024 |
| 26 | Optional Allen CCF "brain atlas mode" (brainglobe + STalign) | NEW | Very high (brain only) | High | `registration.py`, `volume3d.py` | STalign 2023; brainrender 2021 |
| 27 | Sonification of pixel/ROI spectra | NEW | Low (outreach) | Low | ion view | Arpino 2025 |
| 28 | Diffusion virtual staining / DL super-resolution; native VR | NEW | Very high / Med | Very high | — | Sci. Adv. 2025 — out of scope |
| — | HCA tree + live Detail slider; shared UMAP↔tissue colour + lasso; Glasbey/okabe_ito palettes; N-channel additive export; single-ion volume/MIP/orthoslices; percentile hotspot clipping; ion dendrogram + coloc matrix; NMF/PCA; ST co-mapping | HAVE | — | — | `gui/segment.py`, `multivariate.py`, `umapstudio.py`, `gui/exportdialog.py`, `volume3d.py`, `ionembed.py`, `comap.py` | — |

**Suggested sequencing.** Rows 1–8 together form a two-to-three-week "atlas aesthetic" release that needs no new heavy dependencies. Rows 9–17 make SMILE-MSI an atlas-building tool rather than a single-section viewer.

**Citations to verify before reuse.** These come from the notes' own flags:

- Lipizones *Nature* DOI, volume and pages.
- Andersson 2008, Sinha 2008, Palmer 2015 and Thiele 2014 (recalled only).
- The Petras, Floros, Garg and Kapono cartography papers (unverified).
- The Kompauer DOI/URL mismatch.
- The MetaVision3D journal.
- The Richardson 2022 and Sarycheva DOIs (not retrieved).
- DeepION architecture details.
- The MAGPIE DOI.
- The kidney dendrogram figure (PMC2742436).
- The medulloblastoma citation (PMID only).
- Fonville's colour space.
- TOP3 details (snippet only).
- The Hubble-palette source (a photography site).

## Conclusion

The main finding is that the "beauty" of top MSI atlases comes mostly from how the data are organised, not from rendering technology. Colour that encodes the hierarchy and chemical similarity, kept the same across every linked view, does more than any renderer. That changes SMILE-MSI's priorities. The app's segmentation tree already holds the structure that makes Lipizones figures readable. The 12-colour cycle discards it at the last step. Fixing that one step, then adding dot rendering and level-by-level animation, likely recovers most of the reference aesthetic at minimal cost.

The second finding concerns trust. Several of the most attractive techniques can mislead:

- UMAP colourings that miss small structures.
- Hillshading that distorts perceived intensity.
- Fusion or virtual staining that produces *predicted* images.

The most valuable new features (soft-membership blending, iso-intensity contours, perceptual colour spaces, labelled predictions) look good *and* make uncertainty visible. Building SMILE-MSI's visual upgrade around that pairing would set it apart from both the commercial tools and the one-off atlas code.
