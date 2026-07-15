"""Retired Component-analysis tab.

The interactive component-analysis tab this module built (PCA/NMF/UMAP/t-SNE with a
linked feature-space scatter) was retired in the plan-24 move to the Analyze gallery
— every decomposition now runs through the analysis registry. ``ComponentsMixin`` is
kept as an empty shell until its slot is dropped from ``MainWindow``'s base list.
"""
from __future__ import annotations


class ComponentsMixin:
    """Retired Components tab builder — see the module docstring. Kept as an empty
    base only so ``MainWindow`` still imports until its slot is dropped."""
