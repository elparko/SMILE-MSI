# JOSS paper

Source for the *Journal of Open Source Software* submission describing SMILE MSI.

- `paper.md` — the manuscript (Markdown + YAML front matter)
- `paper.bib` — the bibliography (BibTeX)
- `figure.png` — Figure 1, regenerated from the bundled synthetic dataset with
  `make_figure.py`

## Regenerate the figure

```bash
pip install -e .            # the analysis engine (no GUI extra needed)
python paper/make_figure.py # -> paper/figure.png
```

## Preview the compiled PDF locally

JOSS compiles papers with [Inara](https://github.com/openjournals/inara). The
easiest local preview is the official Docker image:

```bash
docker run --rm -v "$PWD/paper":/data -u "$(id -u):$(id -g)" \
  openjournals/inara -o pdf,crossref paper.md
```

This writes `paper/paper.pdf`. Alternatively, push the paper to a branch and open a
preview through the [JOSS Pre-Review](https://joss.theoj.org) process.

## Before submitting

Search `paper.md` for `TODO` and fill in the author name, ORCID, affiliation/country,
the AI-usage disclosure (so it matches your actual process), and the acknowledgements
(funding). JOSS requires the software to be released under an OSI-approved licence
(this project is Apache-2.0) with a documented test suite and contribution guide — all
already present in the repository root.
