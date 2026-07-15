---
title: 'SMILE MSI: an open-source desktop workspace for mass spectrometry imaging with in-silico lipid annotation'
tags:
  - Python
  - mass spectrometry imaging
  - MALDI
  - lipidomics
  - spatial omics
  - bioinformatics
  - imzML
authors:
  - name: "Parker <SURNAME>" # TODO: replace <SURNAME> with your full name as it should appear in print
    orcid: 0000-0000-0000-0000 # TODO: add your ORCID (register free at https://orcid.org)
    corresponding: true
    affiliation: 1
affiliations:
  - name: "Independent Researcher, <COUNTRY>" # TODO: replace with your institution + country, e.g. "Department of X, University of Y, Country"
    index: 1
    # ror: 00000000 # TODO (optional): add your institution's ROR id from https://ror.org
date: 21 June 2026
bibliography: paper.bib
---

# Summary

`SMILE MSI` (Spatial Mass Imaging of Lipid Environments; Python package `smile_msi`)
is an open-source desktop workspace and Python library for analysing **mass
spectrometry imaging (MSI)** data, with an emphasis on lipids. In an MSI experiment a
mass spectrometer records a complete mass spectrum at every pixel of a thin tissue
section, yielding a data cube in which the spatial distribution of any ion can be
rendered as an image. `SMILE MSI` loads such datasets in the open imzML/ibd standard
[@schramm2012imzml], lets the user explore ion images and spectra interactively,
extracts the spatial and statistical structure in the data (segmentation,
co-localization, region comparison), and assigns a putative lipid identity to every
detected mass-to-charge (*m/z*) feature — entirely on the user's own computer, with no
curated database file and no network connection.

The software is two things at once: a point-and-click desktop application, built on Qt
for Python, for researchers who do not program; and an importable, scriptable engine
(`import smile_msi`) for those who do. The same analysis functions back both
interfaces, and an in-application Python console exposes them as pre-bound commands so
that workflows can be saved, replayed across a cohort, and even drafted by an AI
assistant. \autoref{fig:overview} shows a representative analysis of the bundled
synthetic dataset: ion images of three lipid markers, a red/green/blue overlay, an
unsupervised segmentation that recovers the underlying tissue compartments, and the
mean spectrum with in-silico identifications.

![A complete `SMILE MSI` analysis of the bundled synthetic tissue section, produced
entirely through the public Python API. (a--c) Ion images of a white-matter, a
gray-matter, and a lesion-enriched lipid, each labelled with the identity returned by
the in-silico annotator. (d) Three-channel overlay of the same ions. (e) Unsupervised
segmentation recovering the tissue compartments, with the number of regions chosen
automatically. (f) The mean spectrum annotated with the recovered lipid identities.
\label{fig:overview}](figure.png)

# Statement of need

Lipids are spatially organized in tissue, and MSI is a primary tool for mapping that
organization. Turning a raw imzML file into an annotated, statistically defensible list
of discriminating lipids, however, usually requires stitching together several tools: a
viewer for ion images, a separate environment for segmentation and statistics, and a
third step to translate *m/z* values into molecular identities. These pieces frequently
live in different languages, behind a commercial licence, or as a web service to which
data must be uploaded. For a wet-lab scientist without programming experience the path
is fragmented and steep.

`SMILE MSI` packages that entire path — load, explore, find structure, compare,
identify, report — into one locally installed application that also exposes its engine
as a clean library. Identification draws on an in-silico database of several thousand
lipid species enumerated across the major classes (glycerophospholipids, sphingolipids
including sulfatides and gangliosides, sterols and cholesteryl esters, and fatty
acids), reported at sum-composition level following community shorthand nomenclature
[@liebisch2020nomenclature]. Each candidate is corroborated by isotope-pattern [@senko1995] and
adduct consistency and carries a target--decoy false-discovery-rate estimate
[@palmer2017fdr; @elias2007targetdecoy], so an identity arrives with a confidence
context rather than as a bare mass match. Region statistics are computed *exactly* from
per-pixel values — rank-based receiver-operating-characteristic area under the curve
with Mann--Whitney *p*-values [@mann1947] and Benjamini--Hochberg false-discovery-rate
control [@benjamini1995] — rather than from parametric approximations of group means.
Because everything runs locally, unpublished or clinically sensitive data never leaves
the machine.

# State of the field

Several established tools cover parts of this space. Cardinal
[@bemis2015cardinal; @bemis2023cardinal] is a powerful R/Bioconductor package for
statistical MSI analysis, but it is driven through R code. METASPACE [@palmer2017fdr]
provides gold-standard false-discovery-rate-controlled metabolite annotation, but as a
web service to which data is uploaded. MSiReader [@robichaud2013msireader] is a free
viewer and analysis interface that runs on the proprietary MATLAB platform. SCiLS Lab
is a widely used commercial desktop package. Rule-based lipid-identification workflows
such as that of @koelmel2017lipidmatch target chromatography-coupled tandem mass
spectrometry rather than imaging.

`SMILE MSI`'s contribution is the combination, not any single feature: a free,
cross-platform, **local** desktop application — with pre-built Windows and macOS bundles
that need no Python installation — that unifies interactive imaging, exact spatial
statistics, and false-discovery-rate-aware in-silico lipid annotation, while remaining a
small, scriptable Python library. Every quantitative method is reimplemented from its
primary literature rather than ported from an existing tool, which keeps the codebase
licence-clean (Apache-2.0) and documented against its sources.

# Software design

The package separates a dependency-light analysis **engine** from an optional Qt
**GUI**. A base install yields the importable library and a synthetic-data demo, while
the `[gui]` extra adds the desktop application (PySide6 and pyqtgraph). The imzML
backend is *lazy* and *out-of-core*: large `.ibd` files are read on demand rather than
loaded into memory, so datasets larger than RAM remain workable on a laptop. A single
registry of analysis steps backs the GUI, the in-application script console, and a
declarative "flow" format, which guarantees that a result is identical however it was
invoked.

Two choices specifically target reproducibility. First, the lipid database is generated
in silico by enumerating sum compositions from class backbone formulas, so there is no
external file to version or lose and the mass of every candidate is reproducible from
first principles. Second, analyses run under named, versioned **profiles** that capture
tolerances, normalization, peak-picking and segmentation settings, and a random seed;
every result records the active profile together with an auto-generated provenance
report that cites the methods and literature actually used. Stochastic steps thread an
explicit seed so that a given input and profile reproduce the same output on a given
machine. The numerical core builds on NumPy [@harris2020numpy], SciPy
[@virtanen2020scipy], scikit-learn [@pedregosa2011scikitlearn], pandas
[@mckinney2010pandas], and Matplotlib [@hunter2007matplotlib]; spatial structure is
surfaced with Moran's *I* autocorrelation [@moran1950], spatially aware segmentation
[@alexandrov2011segmentation], silhouette-validated cluster counts
[@rousseeuw1987silhouettes], and manifold embeddings via UMAP [@mcinnes2018umap]. The
project is covered by an automated test suite (over 550 tests) run in continuous
integration.

# Research impact

`SMILE MSI` is designed to lower the barrier between a raw imaging dataset and a
publishable result for the large group of MSI practitioners who are not programmers.
Packaging the full pipeline as signed-folder desktop bundles, bundling a synthetic demo
section so that every panel can be exercised without proprietary data, and emitting a
provenance report with each analysis together make a reproducible MSI workflow
accessible without writing code or uploading data to a third party. The parallel Python
API and in-application scripting console make the same engine available for
batch/cohort processing and for integration into larger analysis pipelines, including
AI-assisted ones. The methods documentation maps every quantitative step to its source
in the literature, which we hope also makes the package useful as a teaching reference
for spatial-statistics and lipid-annotation methods in MSI.

# AI usage disclosure

Generative AI coding assistants were used during development of `SMILE MSI` to help
write and refactor code, tests, and documentation, and to assist in drafting this
paper. All AI-assisted output was
reviewed, tested, and edited by the author, who takes responsibility for the
correctness of the software and the content of this manuscript. <!-- TODO: adjust this
paragraph so it precisely reflects how AI was (or was not) used in your project. -->

# Acknowledgements

<!-- TODO: acknowledge any financial support (grants and grant numbers), institutions,
collaborators, or testers. If there was no specific funding, state e.g. "This research
received no specific grant from any funding agency." -->

# References
