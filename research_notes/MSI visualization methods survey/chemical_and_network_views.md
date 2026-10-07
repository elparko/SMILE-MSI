# Chemical and Network Views of MSI Data (beyond the ion image)

Scope: published visualizations showing the molecular/chemical and relational structure of spatial lipidomics/metabolomics data. Target: SMILE-MSI (Python MALDI-MSI lipid app). SMILE-MSI already has: co-localization ranking and matrix, a dendrogram view, a lipid tree, sum-composition in-silico annotation, region comparisons (AUC/FDR), PCA/NMF, a pixel UMAP, and a spatial transcriptomics co-mapping module. Each method below is tagged **[NEW]**, **[PARTIAL]** (an extension of an existing SMILE-MSI view) or **[EXISTS]**.
Difficulty (Python): Low = a few hundred lines with matplotlib/plotly/networkx; Med = needs a new data model or an interactive widget; High = a research-grade model or new annotation depth.
Research date: 2026-10-07. Network limits: biorxiv.org, pubmed.ncbi.nlm.nih.gov and neurosciencenews.com were blocked by the egress proxy, so some 2025–2026 items are cited from search snippets, indexes or secondary pages. Items marked "(from prior knowledge, not re-verified this session)" should be checked before citing.

---

## 1. Ion co-localization networks and ion-image "maps of maps"

### Takeaway
The main published basis for "ion-centric" views is: (a) a validated co-localization metric (ColocML, *Bioinformatics* 2020, not *BMC Bioinformatics*); (b) graph/community views of m/z images (Wüllems et al., *BMC Bioinformatics* 2019); and (c) learned ion-image embeddings (DeepION, *Anal. Chem.* 2024, recent), which support a UMAP where each point is an ion image. SMILE-MSI's coloc matrix and dendrogram already cover (a). The force-directed graph with thumbnail nodes and the ion-image UMAP gallery would be **NEW** and high-impact.

### Cited Findings
- **ColocML.** Ovchinnikova, Stuart, Rakhlin, Nikolenko, Alexandrov. "ColocML: machine learning quantifies co-localization between mass spectrometry images." *Bioinformatics* 36(10):3215–3224 (2020). DOI 10.1093/bioinformatics/btaa085. The brief's citation of *BMC Bioinformatics* is incorrect; the paper is in *Bioinformatics*. — [PMC7214035](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7214035/)
  - Gold standard: 42 imaging-MS experts from 9 labs ranked 2,210 ion-image pairs. The semi-supervised deep "Pi model" (Spearman 0.797 vs. experts) and **cosine similarity after median thresholding** (0.794) performed best. The authors applied the measures to 10,273 molecules from 3,685 public METASPACE datasets. Code: github.com/metaspace2020/coloc. — [PMC7214035](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7214035/); preprint title "ColocAI" — [bioRxiv 758425](https://www.biorxiv.org/content/10.1101/758425)
  - The METASPACE Python client exposes the same colocalization function. — [metaspace2020 docs, image_processing](https://metaspace2020.readthedocs.io/en/latest/content/apireference/image_processing.html)
  - *SMILE relevance:* [PARTIAL]. Median-thresholded cosine is a cheap, validated drop-in metric (Low difficulty) if SMILE-MSI currently uses only Pearson.
- **Community detection and network visualization of m/z images.** Wüllems et al. "Detection and visualization of communities in mass spectrometry imaging data." *BMC Bioinformatics* 20 (2019). DOI 10.1186/s12859-019-2890-6. They build a graph of m/z images connected by spatial similarity and find communities by modularity optimization, to suggest functional networks or pathway activity. They ship an online interactive visualization tool for exploring community substructure. — [PMC6549267](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549267/); [Springer DOI](https://link.springer.com/doi/10.1186/s12859-019-2890-6)
  - *SMILE relevance:* **[NEW]**. Proposed view: a force-directed graph (networkx spring/ForceAtlas2 layout) whose nodes are ion-image thumbnails, with edges drawn above a coloc threshold and nodes colored by Louvain/Leiden community or lipid class. Visual impact very high, scientific value high, difficulty Low–Med (thumbnail rendering in Qt/plotly is the only fiddly part).
- **A conference poster ("Mass spectrometry: from imaging to metabolic networks", Imperial College)** reports network analysis of co-localized ions in colorectal tumors and **module preservation analysis** between patients with and without metastatic recurrence. Treat this as low-tier evidence (poster). — [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/552863bd-8343-4142-bb54-112fc32fdb9b); [Technology Networks poster](https://www.technologynetworks.com/analysis/posters/mass-spectrometry-from-imaging-to-metabolic-networks-304451)
  - *SMILE relevance:* **[NEW]**. A "module preservation" comparison of coloc networks between two groups would be a novel group-contrast view.
- **DeepION (recent, 2024).** "DeepION: A Deep Learning-Based Low-Dimensional Representation Model of Ion Images for Mass Spectrometry Imaging." *Anal. Chem.* 96:3829–3836 (2024). DOI 10.1021/acs.analchem.3c05002. Contrastive self-supervised learning maps each ion image to a low-dimensional vector, so co-localized ions and isotope ions can be found by distance. MSI-specific augmentations simulate Poisson ion counts and random missing-value patterns. It outperformed other methods on rat brain MSI. — [PMC10918617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10918617/); [HKBU record](https://scholars.hkbu.edu.hk/en/publications/deepion-a-deep-learning-based-low-dimensional-representation-mode/)
  - Third-party pages describe a ResNet18 encoder producing 512-dimensional vectors and a GitHub repo gankLei-X/DeepION. I did not verify these details against the paper. — [claudskills page (unverified, low quality)](https://claudskills.com/skills/contrastive-learning-encoder-construction/SKILL.md)
  - *SMILE relevance:* **[NEW]**. Proposed "ion-image UMAP gallery": UMAP of per-ion embeddings, with each point drawn as its thumbnail. A cheap baseline embeds each ion image as a flattened, downsampled, median-thresholded vector (or uses 1 − coloc as the distance) and runs UMAP/t-SNE. DeepION-quality embeddings need PyTorch (High). The baseline is Low–Med. This is the "map of maps" figure and has very high visual impact.

### Inferences
- SMILE-MSI's existing dendrogram plus coloc matrix give a hierarchical-with-thumbnails view almost for free. Adding thumbnails at the dendrogram leaves is [PARTIAL] and Low difficulty.
- The most distinctive additions are (i) a thumbnail-node force graph with community coloring, and (ii) a thumbnail UMAP of ions rather than pixels. Both reuse the coloc matrix SMILE-MSI already computes.

### Gaps
- I could not confirm a dedicated published "ion-image UMAP gallery" figure in METASPACE itself. METASPACE shows ranked colocalized-ion lists in its web UI, but I did not verify a UMAP-of-ions view there.
- I could not verify DeepION's architecture details or the GitHub repository directly.

---

## 2. Lipid-class and chemical-space views (KMD, chain-length × unsaturation grids, ontology, van Krevelen)

### Takeaway
Referenced Kendrick mass defect (RKMD) is the best-documented MSI-specific chemical-space method. It annotates lipids and filters/reconstructs images by class, radyl chain length and unsaturation (Richardson et al., *Anal. Chem.* 2022). LION/web and Lipid Mini-On give ontology-term enrichment that can run on any ranked lipid list, including per-region AUC rankings. All of these are **NEW** for SMILE-MSI; the lipid tree covers only the class hierarchy.

### Cited Findings
- **RKMD annotation and class-based filtering of imaging MS.** Richardson LT, Neumann EK, Caprioli RM, Spraggins JM, Solouki T. "Referenced Kendrick Mass Defect Annotation and Class-Based Filtering of Imaging MS Lipidomics Experiments." *Anal. Chem.* 94(14):5504–5513 (2022). Filtering by lipid class, radyl carbon chain length and degree of unsaturation allows images to be reconstructed per structural feature. Demonstrated on MALDI IMS of human kidney. DOI not shown in the retrieved metadata; look it up on ACS/Crossref. — [NSF PAR record](https://par.nsf.gov/biblio/10352372); [PAR PDF](https://par.nsf.gov/servlets/purl/10352372)
  - The LIPID MAPS RKMD calculator and screening tools are online. — [LIPID MAPS RKMD tool](https://www.lipidmaps.org/tools/ms/kendrick_form.php)
  - An earlier MSI KMD-filter paper also exists: "Rapid visualization of chemically related compounds using Kendrick mass defect as a filter in mass spectrometry imaging" (Maastricht group; ChemRxiv preprint). I did not retrieve its journal, year or DOI. — [Maastricht CRIS](https://cris.maastrichtuniversity.nl/en/publications/rapid-visualization-of-chemically-related-compounds-using-kendric/); [ChemRxiv PDF](https://chemrxiv.org/articles/Rapid_Visualization_of_Chemically_Related_Compounds_Using_Kendrick_Mass_Defect_as_a_Filter_in_Mass_Spectrometry_Imaging/8206349/files/16369682.pdf)
  - Blanc et al. (*Anal. Chem.* 2021) used a KMD variant to display MSI data as maps that reveal unnatural isotopic profiles from stable-isotope tracers. This comes from a search summary; I did not retrieve the primary text. — [search-derived; see PMC9841243](https://pmc.ncbi.nlm.nih.gov/articles/PMC9841243)
  - Within a lipid class, species differing only in double-bond count lie on a diagonal line in an RKMD plot. A slope of −6.69965 was reported in an algal-lipid abstract and is specific to that KMD base. — [SIM abstract T95](https://sim.confex.com/sim/37th/webprogram/Paper29582.html)
  - *SMILE relevance:* **[NEW]**. Proposed view: a KMD (CH2 base) or RKMD scatter of all annotated peaks, with points colored by the region where each ion is most enriched (from existing AUC results) or by the coloc community, and sized by intensity. Lasso-selecting points builds a summed image. Visual impact high, scientific value high (it exposes homologous series and annotation errors), difficulty Low. RKMD needs class reference masses, which SMILE's sum-composition annotator already implies.
- **Chain-length × unsaturation ("lipid map") grids.** I found no single canonical MSI paper that defines this view. The RKMD work above shows images reconstructed by chain length and double bonds. — [NSF PAR](https://par.nsf.gov/biblio/10352372)
  - Bessler et al. imaged cardiolipin profiles in individual murine retinal layers at 10 µm and reported layer-specific PUFA (linoleic, arachidonic, DHA) differences. This is a natural use case for per-layer chain/unsaturation grids. — [Univ. Münster CRIS](https://cris-portal.uni-muenster.de/portal/en/publication/86383348) (source from a search summary; details not re-read)
  - *SMILE relevance:* **[NEW]**. Proposed view: for each class (PC, PE, PI, …), a heatmap with total carbons on the x-axis and double bonds on the y-axis, with cell color showing log2 fold change (region A vs. B) or the region-mean fraction of class signal. Small multiples are arranged per region or per class. SMILE's sum-composition annotation already gives C:DB, so difficulty is Low. Visual impact high: this is the classic "lipidome fingerprint" figure in bulk lipidomics. Isobars and isomers must be shown as caveats.
- **LION/web.** Molenaar MR, Jeucken A, Wassenaar TA, van de Lest CHA, Brouwers JF, Helms JB. *GigaScience* 8(6):giz061 (2019). DOI 10.1093/gigascience/giz061. The ontology links more than 50,000 lipid species to four branches: LIPID MAPS classification, chemical/physical properties (FA length and unsaturation, headgroup charge, intrinsic curvature, membrane fluidity, bilayer thickness), function, and subcellular component. Two modes: target-list (Fisher's exact test of a subset vs. the full set, e.g. a cluster or a thresholded list) and ranking mode (lipids ranked by a supplied value). — [PMC6541037](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6541037/)
- **Lipid Mini-On.** Clair G, Reehl S, Stratton KG, et al. *Bioinformatics* 35(21):4507–4508 (2019). It text-mines lipid names into ontology bins (LIPID MAPS class, chain length, etc.) and offers enrichment through a Shiny app with 5 statistical approaches. Distributed as the R package "Rodin", a Shiny app and an online tool. — [PNNL](https://www.pnnl.gov/publications/lipid-mini-mining-and-ontology-tool-enrichment-analysis-lipidomic-data); [LIPID MAPS tools overview 7.2](https://lipidmaps.org/resources/tools/overview/7-2)
  - *SMILE relevance:* **[NEW]**. Proposed view: run LION-style enrichment per region (ranking mode on AUC) or per coloc community (target-list mode). Show the result as a sunburst/treemap of ontology terms colored by enrichment −log10 q (plotly `sunburst`). A "region × ontology-term" heatmap is a compact alternative. LION's ontology file is downloadable (OBO), so a Python re-implementation is Med difficulty. Scientific value very high: it turns "these 40 ions" into "membrane fluidity / PUFA-PE enriched in white matter".
- **Van Krevelen diagrams with spatial coloring.** I found no MSI-lipidomics primary source in this session. This view is better suited to untargeted metabolomics/DOM with formula assignments than to lipids. **[NEW]** but lower priority for a lipid app.
- **Chord diagrams of lipid-class co-localization.** I found no published MSI example in this session. It can be derived by aggregating the coloc matrix by class pairs. **[NEW]**, Low difficulty (e.g., `mne-connectivity` circle plot, `pycirclize`, or plotly). Value is moderate; this is mainly an aesthetic summary.

### Inferences
- The "chemical-space" views (KMD, C:DB grids, ontology sunburst) are the biggest gap relative to SMILE-MSI's current feature list, and they are cheap because SMILE-MSI already holds sum-composition annotations.
- A coherent new "Chemistry" tab could link KMD scatter → C:DB grid → LION sunburst, all cross-filtered by region or coloc community.

### Gaps
- DOI for Richardson et al. 2022 not retrieved (expected format 10.1021/acs.analchem.1c05xxx; unverified, do not guess).
- No verified MSI paper presenting chord diagrams or van Krevelen views with spatial coloring.
- LipidSuite was named in the brief, but I did not retrieve a source for it.

---

## 3. Spatial pathway visualizations (pathway maps with embedded ion images)

### Takeaway
Pathway-level MSI figures typically place ion images beside pathway schematics: Sun et al., *PNAS* 2019 for tumor metabolic alterations, and a 2023 *Nat. Commun.* gastric-cancer follow-up with MSI plus spatial transcriptomics. I found no standard tool that renders Escher-style maps with ion-image nodes. For SMILE-MSI this is **NEW**, but it is lower priority for a lipid-focused app because lipid metabolism is better summarized by class-to-class conversion networks.

### Cited Findings
- **Sun C, …, He J, Abliz Z.** "Spatially resolved metabolomics to discover tumor-associated metabolic alterations." *PNAS* 116(1):52–57 (2019; online Dec 2018). DOI 10.1073/pnas.1808950116. Airflow-assisted ionization MSI of esophageal cancer was used to discover and visualize pathway-related metabolites and the metabolic enzymes linked to tumor metabolism. — [PMC6320512](https://pmc.ncbi.nlm.nih.gov/articles/PMC6320512)
- **Same group, Nat. Commun. 2023.** "Spatially resolved multi-omics highlights cell-specific metabolic remodeling and interactions in gastric cancer." Combines MSI of metabolites and lipids with spatial transcriptomics on the same section. — [PMC10172194](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10172194/)
- Wüllems et al. 2019 explicitly frame coloc communities as candidate "functional networks or pathway activity". — [PMC6549267](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549267/)
- HT SpaceM (Cell 2025, below) includes a **network analysis of metabolite co-abundance** that highlights coordinated pathways and metabolic hubs. — [PMC12853170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12853170/)
- Escher (King ZA et al., *PLoS Comput Biol* 2015, DOI 10.1371/journal.pcbi.1004321) is the standard web pathway-map builder with data overlays (from prior knowledge, not re-verified this session). It colors nodes by value but does not natively embed images.
- *SMILE relevance:* **[NEW]**. Proposed view for lipids: a "lipid class conversion network" (PA→DG→PC/PE, PE→PS, Cer→SM/HexCer, PC→LPC, etc.) laid out by hand as a fixed template. Each node is a small summed-class ion image, or a per-region bar, and edges are colored by the log-ratio of product to precursor per region. Difficulty Med (template plus networkx/matplotlib inset axes). Visual impact very high. Scientific value moderate to high: ratios such as LPC/PC are interpretable but MALDI ionization biases apply.

### Gaps
- I did not verify any MSI paper that renders a KEGG/Escher map with an embedded ion image at every node. Many papers do this manually in Illustrator. I also did not verify a METASPACE pathway overlay; METASPACE-related enrichment tools exist, but none were retrieved this session.

---

## 4. Region "fingerprint" visuals (radar, ridgelines, mirror spectra, barcodes, waterfalls)

### Takeaway
I found no primary MSI-methods papers that establish radar, ridgeline or "spectral barcode" plots as named methods. These are generic plot forms. They are easy to add (**NEW**, Low difficulty) but should be presented as conveniences, not literature-backed methods.

### Cited Findings
- None retrieved this session with verifiable MSI-specific sources.

### Inferences
- **Mirror spectra** (mean spectrum of region A pointing up, region B pointing down, with differential peaks highlighted using the existing AUC/FDR) would be the most useful of these. They link directly to the existing region comparison. Low difficulty.
- **Spectral barcode**: a 1D strip per region in which each annotated lipid is a vertical bar colored by z-score, ordered by class then by m/z. It is compact and attractive for comparing many regions. Low difficulty.
- **Ridgeline of mean spectra per region / cluster**: attractive, but it is redundant with the barcode for many regions.
- **Radar plots** are hard to read accurately (angular axes, area distortion). Prefer a region × class heatmap; mention radar only for a small number of class-level summaries.

### Gaps
- No citable MSI papers found for these specific forms; they would need a separate search if literature backing is required.

---

## 5. Spatial gradient and trajectory views (line profiles, depth/distance profiles, kymographs)

### Takeaway
Distance-from-boundary profiling has precedent: in mouse medulloblastoma, MSI showed a distance-dependent gradient of cancer-like lipid profiles extending about 1.2 mm beyond the tumor border, corroborated by LCM-LC-MS; in oral SCC DESI, a lipid panel decreases gradually from tumor to normal tissue. A "distance-to-ROI-boundary × lipid" heatmap (kymograph-style) would be **NEW** for SMILE-MSI and is Low–Med difficulty using the existing ROI masks and a distance transform.

### Cited Findings
- Mouse medulloblastoma MSI: "distance-dependent gradient of cancer-like lipid profiles in brain tissue within 1.2 mm of the cancer border", corroborated by laser capture microdissection plus LC-MS. The authors conclude that metabolic borders are not as sharp as morphometric ones. PubMed ID 33651938; full citation not retrieved because PubMed was blocked. — [PubMed 33651938 (snippet via search)](https://pubmed.ncbi.nlm.nih.gov/33651938/)
- Oral squamous cell carcinoma DESI: 14 lipid ions that gradually decrease from tumor to normal tissue carried high weights in a Lasso model with 92.6% accuracy for tumor vs. positive vs. negative margins. Source via search snippet; citation details unverified. — [Drexel record "Defining the molecular tumor margin regions"](https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Defining-the-molecular-tumor-margin-regions/991019319093304721)
- Axial MALDI-TOF of brain tumors with pLSA separated white matter, gray matter/infiltration zone and disseminated medulloblastoma components. — [LabRulez summary](https://lcms.labrulez.com/paper/31529)
- Kidney: a nephrotoxicity MSI study found inflammation-related SM, ceramide and sphingosine species up-regulated and colocalized in the renal cortex, medulla and pelvis respectively. This was found via search snippet; I did not identify the specific paper.

### Inferences
- Implementation: compute `scipy.ndimage.distance_transform_edt` from an ROI boundary (signed: inside negative, outside positive). Bin pixels by distance and plot a heatmap with distance on the x-axis, lipids on the y-axis (ordered by peak position or class) and binned mean z-score as color. This gives a kymograph-like "gradient fingerprint". Ridgelines of selected lipids along the same axis add visual appeal. For cortical depth or kidney cortex→medulla, the same machinery works with a user-drawn reference curve (distance to the pial surface or capsule).
- "Pseudo-spatial trajectory" (diffusion pseudotime on pixel embeddings, e.g. scanpy DPT on SMILE's existing UMAP/PCA, mapped back onto tissue) is a **NEW**, Med-difficulty option. I found no specific MSI paper for it this session.

### Gaps
- No verified MSI paper presenting kymograph-style distance heatmaps under that name. Medulloblastoma paper's authors, journal and DOI not retrieved.

---

## 6. Single-cell MSI and spatial multi-omics joint views

### Takeaway
SpaceM (*Nat. Methods* 2021) and HT SpaceM (*Cell* 2025, recent) established single-cell MSI views: UMAP/PAGA of per-cell metabolic profiles linked to microscopy phenotypes, and co-abundance networks. SMA (*Nat. Biotechnol.*, online 2023, print 2024) plus the 2025–2026 integration frameworks SpatialMETA and MAGPIE (*Nat. Commun.*) define MSI+ST joint views. SMILE-MSI's ST co-mapping covers part of this. Cross-modal latent embeddings and per-cell MSI are **NEW** but heavy.

### Cited Findings
- **SpaceM.** Rappez L, Stadler M, Triana S, et al. "SpaceM reveals metabolic states of single cells." *Nat. Methods* 18:799–805 (2021). DOI 10.1038/s41592-021-01198-0. More than 100 metabolites from more than 1,000 cells per hour, with fluorescence and morpho-spatial readouts. Fig. 2B is a UMAP of single-cell profiles (740 metabolites, 2,840 cells) showing two subpopulations with different lipid-droplet levels. Fig. 3B uses PAGA with Leiden clustering on 740 metabolites to show homeostatic, intermediate and steatotic states in 29,738 hepatocytes. — [PMC7611214](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611214); [Fig. 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611214/figure/F2); [Fig. 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611214/figure/F3)
  - SpaceM data can be loaded and plotted with scverse SpatialData (Python). — [SpatialData SpaceM tutorial](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/technology_spacem.html)
- **HT SpaceM (recent).** Preprint: "HT SpaceM: A High-Throughput and Reproducible Method for Small-Molecule Single-Cell Metabolomics" (bioRxiv 10.1101/2024.10.24.620114). Published version: "HT SpaceM enables high-throughput mapping of metabolic diversity at the single-cell level" (*Cell*, 2025; the exact issue date is unclear, and a commentary record lists January 2026). 135 ions across 78,500 cells in 72 samples, 73 of them validated by LC-MS/MS. NCI-60 extension: more than 42,000 cells, 202 ion features. Annotation runs through METASPACE, and the paper includes a **metabolite co-abundance network** highlighting pathways and hubs. — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2024.10.24.620114.full.pdf); [PMC12853170](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12853170/). Figures are summarized via a commentary; the *Cell* paper itself was not accessed.
- **SMA.** Vicari M, …, Andrén PE, Lundeberg J. "Spatial multimodal analysis of transcriptomes and metabolomes in tissues." *Nat. Biotechnol.* 42:1046–1050 (2024; online 4 Sep 2023). DOI 10.1038/s41587-023-01937-y. MALDI-MSI plus Visium on the same section (Visium-slide compatible), plus histology. Demonstrated with dopamine in 6-OHDA mouse brain (Parkinson's model) and human post-mortem striatum (RRST protocol). — [PubMed 37667091](https://pubmed.ncbi.nlm.nih.gov/37667091/); [SciLifeLab news](https://www.scilifelab.se/news/spatial-multimodal-analysis-combining-mass-spectrometry-and-next-generation-sequencing-for-advanced-brain-studies); dataset [SciLifeLab figshare](https://figshare.scilifelab.se/articles/dataset/Spatial_Multimodal_Analysis_SMA_-_Spatial_Transcriptomics/22778920)
- **SpatialMETA (recent).** Liu lab (ZJU-UoE Institute), *Nat. Commun.* (2025). "Integrating cross-sample and cross-modal data for spatial transcriptomics and metabolomics with SpatialMETA." A conditional VAE with modality-specific decoders and losses aligns ST and spatial metabolomics to a unified resolution, finds cross-modal spatial patterns, and includes visualization functions. Python docs available. — [SpatialMETA docs](https://spatialmeta.readthedocs.io/en/latest/); [ZJE news](https://zje.zju.edu.cn/zje/2025/1009/c77599a3088949/page.htm); [Zenodo code](https://zenodo.org/records/16750012)
- **MAGPIE (recent).** Co-registers spatial transcriptomics, metabolomics (MALDI and DESI) and morphology from the same or consecutive sections. *Nat. Commun.*, DOI 10.1038/s41467-025-68003-w (PMC version January 2026). — [search-derived metadata; PMC12780049 is a related integrative-analysis paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12780049/). Mark the MAGPIE DOI as "from search summary, verify".
- *SMILE relevance:* Joint MSI+ST views are **[PARTIAL]** (the co-mapping module exists). Possible **NEW** additions: (i) a gene–lipid cross-correlation bipartite network or heatmap computed on co-registered spots; (ii) a joint latent UMAP (SpatialMETA-style) colored by modality-specific clusters. (i) is Low–Med difficulty; (ii) is High. Per-cell MSI views (SpaceM-style segmentation, per-cell UMAP/PAGA) are **[NEW]** and High difficulty, and probably out of scope for tissue MALDI-MSI.

### Gaps
- I could not open the HT SpaceM *Cell* paper to describe its exact figures (DOI not captured), or the MAGPIE paper directly.

---

## 7. Visually famous MSI / lipidomics figures (lipid atlases 2023–2026), and atlas-style chemical/relational views

> Coordinator note: the user's reference paper is "Lipizones". Another researcher is dissecting it in full, so only bibliographic anchors are given here. Its aesthetic is the TARGET: hierarchical cluster trees whose colors map to tissue territories, plus lipid-programme heatmaps. The atlas-style views below are prioritized for that reason.

### Takeaway
The flagship recent example is the **3D mouse-brain lipid atlas** of Fusar Bassini, La Manno, D'Angelo et al. (EPFL La Manno lab; bioRxiv October 2025; *Nature* 2026, recent). Its key unit is **539 hierarchically defined "lipizones"**. Related atlas-style precedents are: MLIBRA's hierarchical segmentation by iterative bipartitioning; a classic color-coded kidney dendrogram (cortex/medulla/pelvis); a Vanderbilt kidney lipid atlas over more than 100,000 functional tissue units; and a 2025/2026 explainable-ML mouse-brain "lipid landscapes" study over 123 annotated regions. For SMILE-MSI, the highest-value atlas views are **NEW**: (a) a hierarchical *pixel* taxonomy (tree/icicle) whose node colors are the tissue-map colors, and (b) a "territory × lipid-programme" heatmap with class/C:DB side annotations.

### Cited Findings
- "The lipidomic architecture of the mouse brain." Fusar Bassini L, …, D'Angelo G, La Manno G. bioRxiv 10.1101/2025.10.13.682018 (October 2025); *Nature* (2026), DOI 10.1038/s41586-026-11050-0 (DOI from search summary, verify). 172 lipids at quasi-cellular resolution, 109 sections, 11 mice aged 8 weeks, whole-brain volume. 539 lipizones, many matching known cell types and regions while others give new relational links (e.g., nuclei connected to their projection targets). White matter shown as a "rich patchwork" of biochemical zones. Also covers sex and inter-individual differences and pregnancy. — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2025.10.13.682018.full.pdf); [preLights summary](https://prelights.biologists.com/?p=42883); [News-Medical (Sep 23 2026)](https://www.news-medical.net/news/20260923/New-3D-lipid-atlas-maps-the-chemistry-of-mouse-brains.aspx); [Neuroscience News](https://neurosciencenews.com/lipid-brain-atlas-31242/)
  - Background project: MLIBRA (Mouse LIpid Brain Atlas), Swiss Data Science Center, started June 2023. — [datascience.ch/projects/mlibra](https://datascience.ch/projects/mlibra)
- Other recent spatial lipidomics in brain: Opielka et al., *J. Lipid Res.* 2025. AP-MALDI-Orbitrap imaging of cuprizone de/remyelination annotated 154 (corpus callosum) and 133 (cortex) lipids at sum-composition level, with 60% LC-MS/MS validated; long-chain Cer/HexCer decreased in demyelinated regions. — [LIST research portal](https://researchportal.list.lu/publications/detail/spatial-lipidomics-reveals-demyelination-and-remyelination-dynamics-in-the-mouse-brain). Shafer et al., *JASMS* 2024: sex and Western-diet effects on hippocampus, cortex and corpus callosum lipids (83 lipids). — [PMC11544704](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11544704/)
- Human kidney "molecular histology" by IMS of lipids (Spraggins/Caprioli group) is a related high-impact figure style. — [PMC8922278](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8922278/)
- *SMILE relevance:* **[NEW]/[PARTIAL]**. A "lipizone"-style **hierarchical pixel taxonomy** would show the tissue at several cut levels of a pixel dendrogram. It can be drawn as a tree or icicle whose leaves are colored to match the tissue map, with each node annotated by its top marker lipids. This differs from SMILE-MSI's existing ion dendrogram, which clusters ions rather than pixels. Difficulty Med (bisecting k-means or Ward on PCA/NMF scores; plotly icicle linked to the tissue map). Visual impact very high: this is the atlas aesthetic.

#### Atlas-style precedents (beyond Lipizones)
- **MLIBRA (Mouse LIpid Brain Atlas)**, the Swiss Data Science Center project behind the La Manno work. Its project description specifies "automatic hierarchical segmentation of brain slice MALDI readings" by **iterative bipartitioning plus a generative probabilistic model**, covering hundreds of lipids across the whole adult mouse brain, with the aim of a first whole-organ 3D metabolic atlas. — [datascience.ch/projects/mlibra](https://www.datascience.ch/projects/mlibra); EPFL talk "Atlassing brain lipids" — [EPFL memento](https://memento.epfl.ch/event/dr-la-manno-lab-gioele-la-manno-atlassing-brain--2)
  - Implication: the territory tree is a *divisive* binary tree (repeated 2-way splits), not agglomerative Ward. Python: recursive 2-cluster GMM/k-means on PCA/NMF scores with a stopping rule (minimum size or BIC), which yields a binary tree directly.
- **Color-coded dendrogram precedent (classic).** A review figure shows hierarchical clustering of a mouse-kidney MALDI dataset. Its three main branches are colored cortex (blue), medulla (green) and pelvis (red), with further subdivision of medulla and cortex branches, and the colors are mapped back onto the tissue. This is the pre-2015 antecedent of the lipizone aesthetic. The PMC ID and figure come from a search summary (likely PMC2742436 Fig. 4, ~2009); verify before citing. — [PMC2742436 Fig. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC2742436/figure/F4)
  - A ChemComm supplement describes overlaying spectra from dendrogram branches on the optical image to delineate benign vs. tumor regions. — [RSC suppdata c4cc08331h](https://www.rsc.org/suppdata/cc/c4/c4cc08331h/c4cc08331h1.pdf)
- **"The brain's lipid landscapes uncovered by mass spectrometry imaging and explainable machine learning"** (bioRxiv 10.1101/2025.09.12.675752, recent). It appears to be published as "Mass spectrometry imaging-based explainable machine learning reveals the biochemical landscapes of the mouse brain" (PubMed 42078793, 2026). Search summaries describe a graph-based explainable-ML framework on negative-mode MSI over **123 anatomically defined regions**, with an "MSI-ATLAS" polygon annotation of regions and subregions. Journal and DOI were not retrieved. — [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2025.09.12.675752.full.pdf); [PubMed 42078793](https://pubmed.ncbi.nlm.nih.gov/42078793/)
  - SMILE relevance: an **anatomy-ontology-guided** variant (Allen CCF region tree → lipid heatmap) complements data-driven lipizones. NEW; Med difficulty if registered to an atlas, Low if using user ROIs arranged in a user-defined hierarchy.
- **Vanderbilt/Delft human kidney lipid atlas.** MALDI IMS plus interpretable ML across more than 100,000 functional tissue units (glomeruli, tubules, etc.) in 29 donor kidneys. This is a "functional-tissue-unit × lipid" atlas, i.e. the segmentation unit is anatomy-defined objects rather than pixels. — [Vanderbilt news](https://medschool.vanderbilt.edu/basic-sciences/kidney-atlas-maps-molecular-landscape-unlocking-clues-to-renal-health-and-disease); related "High-Resolution Human Kidney Molecular Histology by IMS of Lipids" — [PMC8922278](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8922278/)
- **NMF "lipid programmes".** Pathirage et al. (*PLOS One* 2024) applied NMF to lipid MSI from a visceral-pain model. NMF components are interpretable as sets of lipid ions, i.e. "programmes". — [PMC11466421](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466421/). A 2024 KU Leuven PhD thesis (M. Nijs) reports Poisson-noise NMF variants (KL divergence) performing best for MSI. — [KU Leuven event page](https://homes.esat.kuleuven.be/~sistawww/it/stadius_redesign/event.php?id=2419). I found no primary paper using the exact phrase "lipid programs" for brain MALDI cell types.

#### Atlas-style views to build in SMILE-MSI (inference)
1. **Territory tree ↔ tissue map with shared colors** (NEW, top priority). Use a divisive binary tree on pixel embeddings (reuse SMILE's PCA/NMF). Assign colors hierarchically so that sibling territories get nearby hues: split the hue wheel recursively per branch and vary lightness by depth. This "hue-by-branch" scheme is what makes lipizone-style figures readable. Render the tree as a dendrogram, icicle or sunburst (plotly). Clicking a node highlights its pixels, and a depth slider re-colors the map at that cut level. Difficulty Med.
2. **Lipid-programme heatmap** (NEW; SMILE has NMF, but not this layout). Rows are territories ordered by tree leaf order, with the tree drawn alongside. Columns are lipids ordered by NMF programme, then class, then C:DB. Cells show z-scored mean intensity. Top annotation bars show lipid class color, total carbons and double bonds; left bars show territory colors. The heatmap is a clustermap with the *pixel* tree on the rows and the *ion* tree or programme blocks on the columns, which unites SMILE's existing ion dendrogram with the new pixel tree. Python: seaborn `clustermap` with `row_linkage` supplied plus `row_colors`/`col_colors`, or PyComplexHeatmap for multi-track annotations. Difficulty Low–Med.
3. **Territory × chemistry summaries**: for each territory node, show a compact C:DB grid (section 2) or a LION term sunburst (section 2) as a "chemical identity card". Cards can be laid out at tree leaves for a small-multiples atlas page. Difficulty Med.
4. **Relational links between distant territories**: Lipizones' "postal code" insight is that spatially separated regions share a lipid identity. A direct view of this is a *territory similarity graph* (nodes are territories positioned at their tissue centroids over a faint section image; edges are lipid-profile similarity above a threshold). This is a spatial "connectome-like" overlay. NEW, Low difficulty, very high visual impact.

### Gaps
- Exact figure designs and the clustering algorithm of the lipid atlas could not be read (bioRxiv blocked); the descriptions above come from secondary summaries. These are deferred to the researcher dissecting that paper. The final *Nature* DOI and date need verification.
- The kidney color-coded dendrogram figure reference (PMC2742436) is search-derived; I did not confirm its authors or year.

---

## Priority summary for SMILE-MSI (inference, based on the above)

Re-prioritized for the user's target aesthetic (Lipizones-style atlas views). Rows A–C come first; the original rows 1–12 follow.

| # | View | Status | Impact | Value | Python difficulty | Key source |
|---|------|--------|--------|-------|-------------------|-----------|
| A | Divisive territory tree with hue-by-branch colors, linked to the tissue map (depth slider) | NEW | Very high | High | Med | Lipizones 2025/26; MLIBRA; classic kidney dendrogram |
| B | Territory × lipid-programme clustermap (pixel tree rows, NMF/ion-tree columns, class/C:DB tracks) | NEW (uses existing NMF and ion dendrogram) | Very high | Very high | Low–Med | Pathirage 2024 (NMF); Lipizones |
| C | Territory similarity graph over the section (distant regions sharing lipid identity) | NEW | Very high | High | Low | Lipizones "postal code" concept |
| 1 | Thumbnail-node coloc force graph + communities | NEW | Very high | High | Low–Med | Wüllems 2019; ColocML 2020 |
| 2 | Ion-image UMAP gallery ("map of maps") | NEW | Very high | High | Low (baseline) / High (DeepION) | DeepION 2024 |
| 3 | KMD/RKMD scatter colored by region/community, lasso → image | NEW | High | High | Low | Richardson 2022 |
| 4 | Class-wise carbons × double-bond heatmaps per region | NEW | High | High | Low | RKMD 2022; bulk lipidomics convention |
| 5 | LION-style ontology enrichment sunburst per region/community | NEW | High | Very high | Med | LION/web 2019; Lipid Mini-On 2019 |
| 6 | Distance-to-boundary lipid heatmap (kymograph) + ridgelines | NEW | High | High | Low–Med | Medulloblastoma gradient (PMID 33651938) |
| 7 | Hierarchical pixel taxonomy ("lipizones") icicle linked to map | NEW | Very high | High | Med | Fusar Bassini et al. 2025/2026 |
| 8 | Lipid class conversion network with ion-image nodes | NEW | Very high | Moderate | Med | Sun 2019 PNAS (concept); Escher |
| 9 | Mirror spectra / spectral barcode per region | NEW | Moderate | Moderate | Low | none MSI-specific found |
| 10 | Gene–lipid bipartite correlation network (on co-mapped ST) | PARTIAL | High | High | Low–Med | SMA 2024; SpatialMETA 2025 |
| 11 | Dendrogram with thumbnail leaves; cosine-after-median-threshold coloc | PARTIAL | Moderate | Moderate | Low | ColocML 2020 |
| 12 | Chord diagram of class-level coloc | NEW | Moderate | Low–Mod | Low | none found |
