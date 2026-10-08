# 3D, Volumetric, Surface-Mapped and Immersive Visualization of MSI Data

Scope note. This covers serial-section 3D MSI, isosurfaces and multi-ion volumes, surface/mesh mapping, atlas registration (target look: the "lipizones" 3D mouse-brain lipid atlas), immersive/VR/AR/3D printing, animations, and single-cell/subcellular 3D. Each item is judged for what SMILE-MSI could add. SMILE-MSI already has: stack, rigid registration, a (Z,H,W) volume for one ion, orthoslices, MIP, and an alpha-composited volume render (pyqtgraph GLVolumeItem via the optional `viz3d` extra in pyproject.toml; `register` extra = scikit-image + SimpleITK).

Verification legend. "[verified]" means confirmed this session by search results (title/venue/DOI/abstract). "[recalled]" means the citation is from prior knowledge and could not be re-checked because nature.com, PMC, bioRxiv, Wiley, Europe PMC, Crossref and OpenAlex were blocked by the egress proxy this session. Check every [recalled] DOI before it goes into a final report.

---

## Q1. Serial-section 3D MSI: volume rendering, isosurfaces, multi-ion 3D overlays

### Takeaway
Most published serial-section 3D MSI figures (2005–2026) do one of three things. They render a semi-transparent tissue "shell" (an outline mesh from optical images or TIC) with one to three ion isosurfaces or point clouds inside it. They show a 3D segmentation/cluster map, with each voxel coloured by its cluster label. Or they show stacked, exploded 2D sections. A colour-coded multi-label 3D segmentation inside a translucent organ shell is the look that gets reused, and SMILE-MSI cannot draw it today (one ion at a time, no meshes, no label volumes).

### Cited Findings
**Foundational and benchmark work**
- Crecelius et al., "Three-dimensional visualization of protein expression in mouse brain structures using imaging mass spectrometry", JASMS 2005. A mouse brain was cut into 264 sections of 20 µm. Paper-and-toner fiducials aligned the optical and MS images. Optical images were registered to each other using an atlas as template. The corpus callosum surface was reconstructed from segmented contours, and the MALDI ion images were then "inserted into the reconstructed structure". Visual: a surface mesh of one brain structure textured or coloured by protein intensity. [verified] — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1044030505002035); [Vanderbilt project page](https://eecs.vuse.vanderbilt.edu/people/bobbyb/research/maldi.html)
- Seeley & Caprioli, "3D Imaging by Mass Spectrometry: A New Frontier", Anal. Chem. 84(5):2105 (2012). The review covers generating 3D volumes "of an entire organ or animal through registration and stacking of serial tissue sections". [verified] — [ACS](https://pubs.acs.org/ancham/article/84/5/2105/1666619/3D-Imaging-by-Mass-Spectrometry-A-New)
- Andersson, Groseclose, Deutch & Caprioli, "Imaging mass spectrometry of proteins and peptides: 3D volume reconstruction", Nat. Methods 5:101 (2008). This is the classic 3D rat/mouse brain MALDI protein volume, with coloured ion volumes in a translucent brain and co-registration to MRI. [recalled] — [DOI 10.1038/nmeth1160](https://doi.org/10.1038/nmeth1160)
- Sinha et al., "Integrating spatially resolved three-dimensional MALDI IMS with in vivo magnetic resonance imaging", Nat. Methods 5:57 (2008). This was multimodal: the MSI volume was fused with the MRI volume. [recalled] — [DOI 10.1038/nmeth1147](https://doi.org/10.1038/nmeth1147)
- Eberlin, Ifa, Wu & Cooks (Purdue), 3D DESI lipid imaging of mouse brain substructures, Angew. Chem. 2010. [verified via news] — [phys.org](https://phys.org/news/2010-01-3d-view-brain.html). [recalled] DOI 10.1002/anie.200906283.
- Trede et al., "Exploring three-dimensional MALDI imaging mass spectrometry data: three-dimensional spatial segmentation of mouse kidney", Anal. Chem. 2012. This is the canonical 3D segmentation render: a kidney volume with voxels coloured by spatially aware cluster. [title verified] — [Univ. Lübeck record](https://research.uni-luebeck.de/de/publications/exploring-three-dimensional-matrix-assisted-laser-desorptionioniz/)
- Oetjen et al., "Benchmark datasets for 3D MALDI- and DESI-imaging mass spectrometry", GigaScience 4:20 (2015), DOI 10.1186/s13742-015-0059-4. The 3D MALDI sets are mouse pancreas, mouse kidney, human oral squamous cell carcinoma and interacting microbial colonies. The 3D DESI set is a human colorectal adenocarcinoma. All are in imzML with reader scripts and deposited in MetaboLights. These are ideal test data for any new SMILE-MSI 3D feature. [verified] — [PMC4418095](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4418095/)
- Oetjen et al., "MRI-compatible pipeline for three-dimensional MALDI imaging mass spectrometry using PAXgene fixation" (J. Proteomics, 2013). [title verified] — [Univ. Lübeck](https://research.uni-luebeck.de/en/publications/mri-compatible-pipeline-for-three-dimensional-maldi-imaging-mass-/fingerprints/)
- Vos, Ellis, Balluff & Heeren, "Experimental and Data Analysis Considerations for Three-Dimensional Mass Spectrometry Imaging in Biomedical Research", Mol. Imaging Biol. 23(2):149–159 (2021), DOI 10.1007/s11307-020-01541-5. This is the most recent general 3D-MSI review I found. Searches did not turn up a dedicated 2023–2025 3D-MSI rendering review. [verified] — [PMC7910367](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7910367/)
- Palmer & Alexandrov, "Serial 3D imaging mass spectrometry at its tipping point", Anal. Chem. 2015. [recalled; DOI not verified]
- Thiele et al., "2D and 3D MALDI-imaging: conceptual strategies for visualization and data mining", BBA Proteins Proteomics 1844:117 (2014). [recalled]

**Dimensionality-reduction colouring in 3D (multi-ion in one view)**
- Abdelmoula et al., "Interactive Visual Exploration of 3D Mass Spectrometry Imaging Data Using Hierarchical Stochastic Neighbor Embedding Reveals Spatiomolecular Structures at Full Data Resolution", J. Proteome Res. 17(3):1054–1064 (2018), DOI 10.1021/acs.jproteome.7b00725. HSNE embeds millions of 3D-MSI spectra at full resolution, and embedding position maps to colour, so a whole 3D dataset shows as one RGB volume. It was benchmarked on public 3D sets and is reported to separate batch effects and mass misalignment. Built with the TU Delft HSNE/Cytosplore stack. [verified] — [PubMed 29430923](https://pubmed.ncbi.nlm.nih.gov/29430923/); [TU Delft](https://publications.graphics.tudelft.nl/papers/187)
- Paine et al. (Fernández/Heeren groups), "Three-Dimensional Mass Spectrometry Imaging Identifies Lipid Markers of Medulloblastoma Metastasis", Sci. Rep. 2019. It used 223 sections (3.3 TB) from six SmoA1-GFP mouse brains, cut at 10 µm with one section kept every 150 µm. A segmentation map is shown in a reconstructed 3D volume. [verified] — [Nature Sci Rep](https://www.nature.com/articles/s41598-018-38257-0)

**Recent automated 3D pipelines (2023–2026, recent)**
- MetaVision3D (Ma, … Sun; Univ. Florida): bioRxiv 2023.11.27.568931. It turns serial 2D MALDI sections into a 3D spatial metabolome through four modules: MetaAlign3D (enhanced-correlation-coefficient alignment), MetaNorm3D, MetaImpute3D and MetaInterp3D (inter-section interpolation for smoother 3D rendering). It produced a mouse-brain 3D metabolome atlas (WT, 5xFAD, GAA) with a web server at metavision3d.rc.ufl.edu, and the data are on Zenodo (14212427). Licence CC BY-NC-SA 4.0. A DOAJ listing suggests it was published in Advanced Science (Feb 2026); verify this. [verified preprint; journal uncertain] — [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.27.568931.full.pdf); [Zenodo](https://zenodo.org/records/14212427); [DOAJ](https://doaj.org/article/501c5df474e042e2b56285268d0dfdb5)
- M²aia (Cordes et al., GigaScience 2021). An MITK-based open-source desktop app with fast interaction, segmentation, deformable 3D reconstruction, multimodal registration and fused datasets in a shared coordinate system. It is a mature open-source 3D MSI viewer and the closest analogue to SMILE-MSI's 3D engine (C++/MITK/VTK, with Python bindings). [verified] — [PMC8290197](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8290197/)
- 3D MSI of 3D cell cultures: more than 40 consecutive sections from about 300 µm spheroids were reconstructed in 3D. The authors note that no integrated platform yet exists for spheroid 3D MSI. [verified] — [bioRxiv 2022.12.05.519157](https://www.biorxiv.org/content/10.1101/2022.12.05.519157.full.pdf)
- A 2025 multiomic 3D approach integrated immunohistochemistry with MALDI-MSI to reconstruct 3D biomolecular distributions in Alzheimer's disease brain (Anal. Chim. Acta 2025). [title verified only] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0003267025011158)

### Inferences
- **NEW for SMILE-MSI: multi-ion RGB/additive volumes.** Put 2–4 ions into separate colour channels of one volume, or blend separate volumes additively. pyqtgraph GLVolumeItem already takes RGBA voxels, so the existing GL path can do this. Difficulty: low (about 1–2 days).
- **NEW: per-ion isosurfaces (marching cubes).** Use `skimage.measure.marching_cubes`, which is already a dependency, to build meshes, then draw them with pyqtgraph `GLMeshItem` (no new deps) or pyvista `contour()`. Several translucent coloured isosurfaces inside a TIC-derived tissue shell is the classic 3D-MSI figure. Smooth the volume first (Gaussian) and decimate the meshes. Difficulty: low–medium.
- **NEW: 3D cluster/segmentation label volumes.** Run k-means/UMAP/HSNE on all voxels, then show the labels as coloured isosurfaces per label or as a categorical volume render. This reproduces the Trede 2012 / Paine 2019 / lipizone look, and SMILE-MSI's single-ion design does not support it. Difficulty: medium, because of memory and spectra × voxels.
- **NEW: inter-section interpolation.** Section spacing (often 100–150 µm against 10–50 µm pixels) makes volumes look jagged. MetaVision3D-style interpolation or imputation, or simply anisotropic `scipy.ndimage.zoom` along Z plus smoothing, gives much better isosurfaces. Difficulty: low for linear or shape-based interpolation, high for learned imputation.
- **NEW: non-rigid registration.** SimpleITK BSpline/Demons is already available through the `register` extra, so this is cheap to add. M²aia and MetaVision3D both go beyond rigid registration.
- The Oetjen 2015 GigaScience datasets (imzML, public) are the right regression and demo data for any of these features.

### Gaps
- I found no dedicated 2023–2026 review of 3D-MSI rendering techniques (transfer-function design and similar). The Vos 2021 review is the latest general one I found.
- I could not open full texts (publisher, PMC and bioRxiv domains were blocked), so the exact rendering software for most papers (Amira, ParaView, Imaris, SCiLS Lab 3D, MATLAB) is not confirmed. Andersson 2008, Sinha 2008, Palmer 2015 and Thiele 2014 are [recalled] only.

---

## Q2. Mapping MSI and metabolomics onto 3D surfaces and meshes (organisms, plants, rooms, curved samples)

### Takeaway
Two families exist. (a) "Molecular cartography" (Dorrestein/Alexandrov): discrete swab or punch samples are measured by LC-MS, and the feature intensities are painted onto a 3D mesh (body, plant, room or lung) by interpolating between sampling spots in the 'ili WebGL app. (b) True topographic MSI: autofocusing AP-MALDI (Kompauer 2017) or LAESI measures a height map along with the spectra, so the ion image is draped over the measured 3D surface. Both are very striking and both are new for SMILE-MSI. Family (b) is the more relevant to MALDI.

### Cited Findings
- Bouslimani, Porto, Rath … Dorrestein, "Molecular cartography of the human skin surface in 3D", PNAS 112(17):E2120–9 (2015), DOI 10.1073/pnas.1424409112. About 400 skin sites on two volunteers were analysed by MS plus 16S sequencing and mapped onto 3D human body models. Visual: a rotating 3D human figure with heat-mapped molecule abundance. [verified] — [PubMed 25825778](https://pubmed.ncbi.nlm.nih.gov/25825778/); [UCSD news](https://today.ucsd.edu/story/3d_human_skin_maps_aid_study_of_relationships_between_molecules_microbes_an)
- Protsyuk et al., "3D molecular cartography using LC-MS facilitated by Optimus and 'ili software", Nat. Protoc. 13(1):134–154 (2018; online 2017). It is a full protocol from sampling through LC-MS, QC, MS/MS identification and processing to 3D visualization. Examples are a rosemary plant and an ATM keypad. It introduced "cartographical snapshots", shareable files that store a map plus its view settings. 'ili ("skin/surface" in Hawaiian) is open source (Protsyuk, Ryazanov; PI Alexandrov, EMBL, with UCSD). [verified] — [NIJ record](https://nij.ojp.gov/index%2ephp/library/publications/3d-molecular-cartography-using-lc-ms-facilitated-optimus-and-ili-software); [GitHub ElDeveloper/ili](https://github.com/ElDeveloper/ili); [Chemistry World](https://www.chemistryworld.com/news/molecular-map-making-simplified/3008464.article). DOI 10.1038/nprot.2017.122 [recalled].
- 'ili was described in an EU CORDIS report as an open-source Chrome app for 3D surface imaging used by 98 users, with plans to integrate its methods into SCiLS Lab 3D. [verified] — [CORDIS 305259](https://cordis.europa.eu/project/id/305259/reporting/pl)
- Other cartography showcases [all recalled, unverified this session]: Petras et al., "Mass spectrometry-based visualization of molecules associated with human habitats", Anal. Chem. 2016 (an apartment mapped in 3D); Floros et al., "Mass spectrometry based molecular 3D-cartography of plant metabolites", Front. Plant Sci. 2017; Garg et al., "Three-dimensional microbiome and metabolome cartography of a diseased human lung", Cell Host Microbe 2017 (a CF lung explant mesh); Kapono et al., Sci. Rep. 2018 (3D chemical snapshot of a human habitat).
- Kompauer, Heiles & Spengler, "Autofocusing MALDI mass spectrometry imaging of tissue sections and 3D chemical topography of nonflat surfaces", Nat. Methods 14:1156–1158 (2017), DOI 10.1038/nmeth.4433. In AP-MALDI with laser-triangulation autofocus, the stage height is adjusted per pixel, giving topographic aspect ratios up to 50 and ≤10 µm lateral resolution. Output: ion images draped over a measured 3D height map of non-flat samples such as plants and insects. [verified via patent/vendor citations] — [Nature](https://www.nature.com/articles/nmeth.4513) (link from search; DOI per citations); [Thermo technical note on the "3D-Surface Imaging Mode"](https://assets.thermofisher.com/TFS-Assets/CMD/Technical-Notes/tn-000659-smaldi-msi-tn000659-en.pdf)
- Nemes, Barton & Vertes, "Three-dimensional imaging of metabolites in tissues under ambient conditions by LAESI MS", Anal. Chem. 2009 (z-resolved ablation of plant tissue). It predates 2013 and is foundational. [recalled]

### Inferences
- **NEW: surface draping (2.5D).** If a height map exists (autofocus or topography data, or a depth image from a phone or photogrammetry), drape the ion image as a texture or scalars on a `pyvista.StructuredGrid` / `warp_by_scalar` surface. Difficulty: low with pyvista, medium in the pyqtgraph `GLSurfacePlotItem`. Even without real topography, "ion intensity as height" (a 2.5D relief of one section) is an easy, eye-catching extra.
- **NEW: 'ili-compatible export.** 'ili takes a 3D model plus a CSV of per-spot (x, y, z) intensities and runs in the browser. Exporting coordinates/intensities from SMILE-MSI in that format, or exporting glTF/PLY meshes with per-vertex colours (pyvista `export_gltf`, trimesh), gives web viewing for free. Difficulty: low.
- Molecular-cartography meshes (body or room) are mostly irrelevant to a MALDI-section app. Drawing MSI onto a reconstructed organ surface mesh (built by marching cubes from the stacked tissue masks) is the transferable idea.

### Gaps
- I could not verify the Petras 2016, Floros 2017, Garg 2017 and Kapono 2018 citations.
- I found no Python package that reproduces 'ili's interpolation; 'ili itself is JavaScript/WebGL.

---

## Q3. Whole-body, atlas-registered and 3D lipid brain atlases (target aesthetic: "lipizones")

### Takeaway
The target look is a whole mouse brain, registered to the Allen CCFv3, with territories rendered as coloured 3D regions or clouds inside a translucent brain shell. It is built from three parts: (1) register each MSI section to the CCF (affine or LDDMM, e.g., STalign, or the older Abdelmoula 2014 pipeline); (2) cluster pixels into lipid-defined territories (lipizones); (3) render them with the BrainGlobe/brainrender stack (Python, vedo/VTK) or an equivalent pyvista scene, using the CCF root mesh as the translucent shell. All of these tools are open-source Python, so this look is reachable for SMILE-MSI. It is the most "new" and highest-impact item in this survey.

### Cited Findings
**The target work (dissected by another researcher, so only rendering-relevant facts here)**
- Fusar Bassini … D'Angelo, La Manno (EPFL), "The lipidomic architecture of the mouse brain", bioRxiv 2025.10.13.682018 (Oct 2025), now listed on nature.com as a Nature article (URL s41586-026-11050-0, so a 2026 Nature paper; volume/pages unverified). MALDI-MSI of 109 coronal sections from 11 mice at a 25 µm raster, registered to the Allen CCF. In the published version, an extra STalign refinement uses the Allen density images as reference and the m/z 845.528 peak (the one most correlated with density) as target. 172 high-confidence lipids. 539 "lipizones" in a hierarchical clustering (dendrogram of 31 subclasses; four major lipidomic subdivisions). Processing used uMAIA, extended into EUCLID (Enhanced uMAIA for Clustering Lipizones, Imputation, and Differential analysis). Web: "Lipid Brain Atlas Explorer", a Python Dash app at lbae-v2.epfl.ch. [verified via search summaries; peak counts differ between preprint and Nature version] — [nature.com listing](https://www.nature.com/articles/s41586-026-11050-0); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1.full); [preLights](https://prelights.biologists.com/highlights/the-lipidomic-architecture-of-the-mouse-brain/); [Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/); [News-Medical, 23 Sep 2026](https://www.news-medical.net/news/20260923/New-3D-lipid-atlas-maps-the-chemistry-of-mouse-brains.aspx)
- Sister work (same labs): "Unified mass imaging maps the lipidome of vertebrate development", Nat. Methods 2025, DOI 10.1038/s41592-025-02771-7. A 4D (3D + developmental time) lipid atlas of whole zebrafish embryos with more than 100 lipids, from serially sectioned embryos at several stages, built with uMAIA (adaptive peak extraction, cross-section molecule matching, noise correction). This is the whole-organism counterpart of the brain atlas. [verified] — [PMC12446072](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12446072/); [EPFL news](https://actu.epfl.ch/news/mapping-the-lipid-blueprint-of-life-in-4d)

**Registration of MSI to the Allen atlas**
- Abdelmoula, Carreira … McDonnell, Dijkstra, "Automatic Registration of Mass Spectrometry Imaging Data Sets to the Allen Brain Atlas", Anal. Chem. 86(8):3947–3954 (2014), DOI 10.1021/ac500148a. It picks the best-matching ABA reference section, registers, and transfers annotations for inter-animal comparison. An SPIE 2014 precursor exists. [verified] — [TU Delft Pure](https://pure.tudelft.nl/portal/en/publications/automatic-registration-of-mass-spectrometry-imaging-data-sets-to-the-allen-brain-atlas(8fc18c3f-b2be-4fb5-9431-f50851215417).html); [SPIE](https://spiedigitallibrary.org/conference-proceedings-of-spie/9034/90343M/Automatic-registration-of-imaging-mass-spectrometry-data-to-the-Allen/10.1117/12.2043653.full)
- Škrášková et al., JASMS 2015: the same pipeline applied to ToF-SIMS lipid images of Mfp2-deficient mouse brain, for anatomical annotation of lipid deposits. [verified] — [Maastricht CRIS](https://cris.maastrichtuniversity.nl/en/publications/precise-anatomic-localization-of-accumulated-lipids-in-mfp2-defic/)
- Clifton, Anant … Fan, "STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping", Nat. Commun. 14:8123 (2023), DOI 10.1038/s41467-023-43915-7. LDDMM handles partially matched, nonlinearly distorted sections. It has a 3D-to-2D mode against Allen CCFv3 (the default atlas) and transfers region labels. It is a pip-installable Python toolkit (GitHub JEFworks-Lab/STalign), and the lipizone paper uses it for MSI. [verified] — [PubMed 38065970](https://pubmed.ncbi.nlm.nih.gov/38065970/); [GitHub](https://github.com/JEFworks-Lab/STalign)
- Recent (2025): multimodal MALDI-2/TIMS MSI of 5xFAD mouse brain with regions annotated against Allen CCFv3 (Nat. Commun. 2025, "Multimodal mass spectrometry imaging for plaque- and region-specific neurolipidomics in Alzheimer's disease mouse models"). [verified title/venue] — [Nat Commun](https://www.nature.com/articles/s41467-025-65956-w)

**Rendering stack that produces the atlas look**
- Claudi, Tyson, Petrucco, Margrie, Portugues & Branco, "Visualizing anatomically registered data with brainrender", eLife 10:e65751 (2021), DOI 10.7554/eLife.65751. Open-source Python for interactive rendering of data registered to atlases. It uses BrainGlobe AtlasAPI to load the Allen and other atlases without modification, mixes data types in one scene, and exports high-resolution figures and animated videos. `pip install brainrender`. Built on vedo/VTK [vedo dependency is from prior knowledge; not confirmed in the search snippet]. [verified] — [PubMed 33739286](https://pubmed.ncbi.nlm.nih.gov/33739286/); [GitHub brainglobe/brainrender](https://github.com/brainglobe/brainrender)

### Inferences
- **Recipe to reproduce the lipizone/atlas look in Python (NEW for SMILE-MSI):**
  1. `brainglobe-atlasapi`: `BrainGlobeAtlas("allen_mouse_25um")` gives the annotation volume, the reference template, the structure tree with official Allen RGB colours, and per-region OBJ meshes.
  2. Register each SMILE-MSI section to the atlas. The simplest route is a user-chosen AP position plus 2D affine/BSpline registration of the TIC or a density-correlated lipid image to the matching reference slice (SimpleITK, already a dependency). A better route is STalign (LDDMM, 3D-to-2D), as the lipizone paper did.
  3. Cluster all registered pixels (k-means/Leiden on PCA of lipid intensities) to get "lipizones", and give each cluster a colour.
  4. Render: a translucent CCF root mesh (opacity about 0.1–0.2), per-cluster point clouds (`pv.PolyData` points as spheres/gaussians) or per-cluster marching-cubes isosurfaces from the label volume, optionally Allen region meshes for context. brainrender does steps 1 and 4 almost directly (`scene.add_brain_region`, `Points`, `scene.render`, `Animation`), and pyvista does the same with more control.
  Difficulty: medium. Steps 1 and 4 take about 1–2 days with brainrender. Step 2 is the hard part (about 1–2 weeks to make robust). Step 3 is easy. The CCF only applies to mouse (rat and human atlases exist in BrainGlobe), so this should be an optional "brain atlas mode".
- A cheaper non-atlas variant of the same look needs no Allen registration. Cluster the existing SMILE-MSI stacked volume, mesh each cluster with marching cubes, and render the coloured meshes inside a translucent TIC-derived shell. It works for any organ. Difficulty: low–medium. It gives most of the visual impact.
- Web export of the atlas look: the LBAE (Dash) and MetaVision3D web servers show that the field ships atlases as web apps. Exporting glTF scenes (pyvista `export_gltf`/`export_html` via trame) gives a self-contained HTML 3D viewer.

### Gaps
- I could not access the lipizone paper's methods to confirm which renderer made its 3D figures (brainrender vs. napari vs. custom). Another researcher is covering that paper.
- I found no published whole-body (mouse/rat) 3D MSI study verified this session. Whole-body 3D MSI exists in the literature but no specific citation was confirmed, so treat it as a gap.

---

## Q4. Immersive VR/AR, mixed reality, 3D-printed or physical molecular models

### Takeaway
Immersive MSI is barely published. The only direct hit is a 2026 Advanced Science paper (Iakab et al.) that exports 3D MALDI volumes of organoid/3D cell-culture models into a mixed-reality/VR tool. AR exists in intraoperative probe MS (not imaging). I found no peer-reviewed 3D-printed MSI molecular map. VR/AR/3D printing would therefore be novel, but it is a niche feature; a glTF/STL export is the cheap enabler.

### Cited Findings
- Iakab et al., "From Sample to Mixed Reality: A Translational 3D MALDI Imaging Platform for Advanced 3D Spatial Omics Analysis of 3D Cell Culture Disease Models", Advanced Science 2026, DOI 10.1002/advs.202516098 (recent). The workflow exports 3D image volumes that load into a virtual-reality tool or can be analysed in R. Headset and software were not confirmed (full text blocked). [verified title/venue/abstract only] — [Wiley](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202516098)
- AR with mass spectrometry in surgery: an MS probe classifies tissue in seconds, and an AR display overlays red/blue (tumour/healthy) pixels on the surgical video using infrared tracking. This is probe MS, not MSI. [verified] — [Chemistry World](https://www.chemistryworld.com/news/mass-spectrometry-and-augmented-reality-guide-tumour-removal-in-real-time/4012263.article)
- AR for spectral imaging in general (combining spectral datasets with 3D models; not MS): Applied Sciences 2025. [verified] — [MDPI](https://doi.org/10.3390/app15126635)
- Searches for 3D-printed MSI physical models found none. The only "printed" item was the Vanderbilt group's early method development on printed images of brain slices. [verified negative result] — [Vanderbilt d3dims PDF](https://eecs.vuse.vanderbilt.edu/People/bobbyb/pubs/d3dims.pdf)

### Inferences
- **NEW and cheap: mesh export.** Write per-ion isosurfaces and cluster meshes as STL (3D printing, monochrome per part), PLY/OBJ with vertex colours, or glTF/GLB (VR headsets, PowerPoint 3D, web `<model-viewer>`, Blender). trimesh or pyvista can do this. Difficulty: low (about 1 day once isosurfaces exist). Multi-colour printing would mean one STL per ion/cluster.
- Native VR (OpenXR) in a Qt desktop app is high effort and low value. Exporting to glTF and letting users open it in existing viewers (Blender, Meta Quest browser via WebXR/three.js) is the pragmatic route.
- **Blender route for showcase renders:** import glTF/PLY, use Cycles path-tracing with emission/volume shaders, and animate a turntable. This gives the most "publication-cover" look, at moderate effort for users and no SMILE-MSI code beyond export.

### Gaps
- I found no peer-reviewed VR study for MALDI tissue MSI beyond Iakab 2026, and I could not confirm the VR software that paper used.
- I found no published 3D-printed MSI maps.

---

## Q5. Animated renders: rotations, fly-throughs, slice sweeps, m/z sweep movies, time-lapse

### Takeaway
Animation in published 3D MSI is mostly supplementary movies: turntable rotations of 3D volumes, segmentations and atlases, plus slice-by-slice sweeps. brainrender explicitly supports animated video export. The "4D" zebrafish lipid atlas adds a developmental-time axis. I found no peer-reviewed paper centred on m/z-sweep movies; they are a common talk/demo device but not documented as a method. All of these animations are cheap to add and new for SMILE-MSI.

### Cited Findings
- brainrender exports "high-resolution figures and animated videos" of atlas-registered scenes. [verified] — [PubMed 33739286](https://pubmed.ncbi.nlm.nih.gov/33739286/)
- The uMAIA zebrafish atlas is explicitly 4D (3D space plus developmental time) across stages, a natural time-lapse/morph animation subject. [verified] — [EPFL news](https://actu.epfl.ch/news/mapping-the-lipid-blueprint-of-life-in-4d)
- MetaVision3D's interpolation module exists specifically to improve 3D rendering continuity, which matters most for rotating and fly-through views. [verified] — [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.27.568931.full.pdf)

### Inferences
- **NEW: animation exporter (MP4/GIF).**
  - (a) Turntable: rotate the camera azimuth 0–360° (pyvista `open_movie` + `orbit_on_path`, or pyqtgraph `GLViewWidget.orbit()` + `grabFramebuffer()` piped to imageio-ffmpeg).
  - (b) Slice sweep: step through Z, or move a clipping plane through the volume.
  - (c) m/z sweep: step through m/z bins of one section (or volume) with a running m/z label and a spectrum cursor.
  - (d) Ion morph: cross-fade between lipids.
  - (e) Cluster "build-up": add lipizones one by one.
  Difficulty: low (about 2–3 days for all five with imageio-ffmpeg). Visual payoff is high, since it suits talks and social media.
- The m/z sweep is the one animation that needs no 3D engine at all, and SMILE-MSI's existing 2D ion-image code could produce it directly.

### Gaps
- I found no citable paper that presents m/z-sweep movies as a visualization method. It is plausibly common in supplementary videos, but this is unverified.

---

## Q6. Single-cell and subcellular 3D (ToF-SIMS, OrbiSIMS, NanoSIMS)

### Takeaway
Subcellular 3D MSI comes from SIMS depth profiling. Each sputter cycle is one image plane, and the main visual and algorithmic challenge is z-correction, because cells are not flat. The corrected stacks are shown as voxel volumes or projected onto a reconstructed 3D cell-surface model. The technique is out of scope for a MALDI app, but the "project chemistry onto a reconstructed surface" rendering idea transfers.

### Cited Findings
- Passarelli et al., "The 3D OrbiSIMS—label-free metabolic imaging with subcellular lateral resolution and high mass-resolving power", Nat. Methods 2017, DOI 10.1038/nmeth.4504. It combines a GCIB-SIMS with an Orbitrap: under 2 µm for biomolecules, under 200 nm for inorganics, and more than 240,000 resolving power at m/z 200. It visualized metabolites in 3D at subcellular resolution, including amiodarone in single macrophages and 29 sulfoglycosphingolipids and 45 glycerophospholipids in tissue. [verified] — [Nottingham ePrints](https://eprints.nottingham.ac.uk/48339/); [OA PDF](https://nottingham-repository.worktribe.com/OutputFile/895031)
- Robinson et al. 2012, ZCorrectorGUI (MATLAB). It corrects whole 3D ToF-SIMS data cubes for cell topography and was validated against AFM on NIH/3T3 fibroblasts. A 2025 method (Brunet, Gorman & Kraft) uses total-ion images per depth step to build a 3D cell-surface model, corrects voxel z, and projects the chemistry onto the cell model. This made ER–plasma-membrane junctions interpretable relative to surface ridges and valleys. [verified via search summary] — [PMC12467685](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12467685/)
- NanoSIMS depth correction (Kraft lab, Illinois): it reconstructs cell morphology per raster plane from secondary-ion and secondary-electron images, was validated against AFM and FIB-SEM, and resolved ¹⁸O-cholesterol and ¹⁵N-sphingolipid vesicles and membranes in 3D. [verified] — [Illinois Experts](https://experts.illinois.edu/en/publications/depth-correction-of-3d-nanosims-images-shows-intracellular-lipid-/)

### Inferences
- These features are not relevant to SMILE-MSI's MALDI serial-section pipeline. The 2.5D "drape chemistry on a reconstructed surface" rendering is the same machinery as Q2's surface draping.

### Gaps
- I did not verify specific t-MALDI-2 or MALDI-2 subcellular 3D work this session.

---

## Q7. Most visually striking published renders, how they were made, and implementation difficulty for SMILE-MSI

### Takeaway
The highest-impact visuals are, in order:
1. Atlas-registered multi-region coloured 3D brains (lipizones; MetaVision3D atlas).
2. 3D molecular cartography on human, plant or room meshes ('ili; Bouslimani 2015).
3. Multi-colour isosurfaces or segmentations inside a translucent organ shell (Andersson 2008; Trede 2012; Paine 2019).
4. Topographic draped MSI of non-flat objects (Kompauer 2017).

Items 1 and 3 are reachable in Python with pyvista or brainrender at low-to-medium effort. All four are new beyond SMILE-MSI's current single-ion volume, MIP and orthoslice views.

### Cited Findings
- The atlas look is built on Allen CCF registration plus clustering plus web/3D rendering: lipizone paper (STalign, uMAIA/EUCLID, Dash explorer) and MetaVision3D (web atlas). [verified] — [nature.com listing](https://www.nature.com/articles/s41586-026-11050-0); [MetaVision3D bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.27.568931.full.pdf)
- Atlas-registered rendering with video export in Python: brainrender. [verified] — [eLife/PubMed](https://pubmed.ncbi.nlm.nih.gov/33739286/)
- Surface cartography: 'ili (WebGL, browser) with shareable snapshots. [verified] — [GitHub](https://github.com/ElDeveloper/ili); [Nat Protoc record](https://nij.ojp.gov/index%2ephp/library/publications/3d-molecular-cartography-using-lc-ms-facilitated-optimus-and-ili-software)
- Open-source desktop 3D MSI with deformable reconstruction: M²aia. [verified] — [PMC8290197](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8290197/)
- 2D-focused interactive exploration tools such as MSI-VISUAL (bioRxiv 2025, "Truthful visualizations…") are complementary, not 3D. [verified] — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.18.643852.full.pdf)

### Inferences
Implementation matrix for SMILE-MSI. Effort is approximate developer time and is my judgement, not sourced.

| Method | Visual impact | New beyond current engine? | Python route | Difficulty |
|---|---|---|---|---|
| Multi-ion RGB / additive volume render (2–4 ions) | High | Yes | pyqtgraph GLVolumeItem RGBA (existing); napari multi-layer additive blending | Low |
| Per-ion isosurfaces (marching cubes) in translucent TIC shell | High | Yes | skimage `marching_cubes` (existing dep) → pyqtgraph `GLMeshItem` or pyvista `contour` | Low–Med |
| 3D clustering/segmentation label volume ("lipizone-like", no atlas) | Very high | Yes | sklearn KMeans/UMAP on voxels → per-label meshes, categorical colormap | Med |
| Allen CCF registration + atlas-region overlay (target look) | Very high | Yes | brainglobe-atlasapi + SimpleITK or STalign → brainrender/pyvista scene | Med–High |
| Z-interpolation / smoothing for anisotropic stacks | Medium (enabler) | Yes | scipy.ndimage zoom + Gaussian; shape-based interpolation | Low |
| Non-rigid section registration | Medium (enabler) | Yes | SimpleITK BSpline/Demons (existing dep) | Med |
| Turntable / slice-sweep / m/z-sweep / ion-morph movies | High | Yes | imageio-ffmpeg + pyqtgraph framebuffer, or pyvista `open_movie` | Low |
| Exploded-section 3D view (sections floating apart in Z) | Medium–High | Yes | pyqtgraph `GLImageItem` per section, or pyvista textured planes | Low |
| 2.5D surface draping / intensity-as-height | Medium–High | Yes | pyvista `warp_by_scalar`; pyqtgraph `GLSurfacePlotItem` | Low |
| Web export (self-contained HTML 3D viewer) | High (sharing) | Yes | pyvista `export_html` (trame/vtk.js) or glTF + three.js `<model-viewer>`; 'ili CSV+mesh export | Low–Med |
| Mesh export (STL/PLY/glTF) for 3D printing, VR, Blender | Medium (niche) | Yes | trimesh / pyvista save | Low |
| Blender path-traced hero renders | Very high (cover art) | Yes (via export) | glTF → Blender Cycles (user side) or bpy script | Med |
| Native VR/AR | Medium | Yes | OpenXR / WebXR | High (not recommended) |
| HSNE/UMAP RGB colouring of 3D volume | High | Yes | umap-learn → RGB per voxel → RGBA volume | Med |

- **Library choice.** pyvista (VTK) gives GPU ray-cast volume rendering with transfer functions, isosurfaces, clipping planes, movie export and HTML export in one API, and embeds in Qt via pyvistaqt. napari gives multi-layer 3D with additive blending and is quick for prototyping, but it is a separate viewer app. vispy is the low-level layer under napari, and pyqtgraph's GL is already used by SMILE-MSI but lacks transfer functions and good lighting. brainrender is the shortest path to the exact Allen-atlas look, but it pulls in vedo/VTK and is mouse/rat-brain specific.
- **Recommended priority for SMILE-MSI**, by impact per effort: (1) isosurfaces plus multi-ion overlay, (2) animation export, (3) 3D cluster label volumes, (4) glTF/HTML export, (5) optional Allen-CCF atlas mode via brainglobe-atlasapi for brain datasets.

### Gaps
- I could not confirm which specific renderer (Amira, ParaView, Imaris, Blender, brainrender) made each showcase figure, because full texts were inaccessible this session.
- I found no quantitative or user-study comparison of 3D MSI visualization styles. Impact rankings here are qualitative.
