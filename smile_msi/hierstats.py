"""Hierarchical (nested) cohort statistics — subject-level inference for MSI group tests.

Where :mod:`smile_msi.spatial` compares two pixel regions **within** one slide and
:mod:`smile_msi.cohort` compares two groups of samples **across** slides, this module
handles the design that sits between them: several *compartments* measured in every
*subject*, with subjects split into two groups. Pixels within a compartment are
sub-measurements of one biological sample, so the unit of replication is the **subject**,
not the pixel — and a compartment's three values within one subject are correlated,
so they are not three independent observations either.

    group  (normal | treated)
      └─ subject      ← the unit of replication (a nerve, a donor, an animal)
           └─ compartment   ← repeated measure within the subject
                └─ pixel    ← pseudoreplicate; summarized away before any test

Feeding raw pixels to a per-pixel t-test inflates n by 3-4 orders of magnitude and turns
between-subject variation into "significance". Cardinal's ``meansTest`` (Bemis et al. 2019,
doi:10.1093/bioinformatics/btv146) established the fix for MSI: summarize each replicate to
one mean profile and test *those*. This module keeps that stance and adds the two things a
pseudobulk rank test cannot do at small n:

* :func:`moderated_t_two_group` — limma-style empirical-Bayes variance shrinkage
  (Smyth 2004, doi:10.2202/1544-6115.1027). Borrowing variance across all features buys
  real power at n = 3 vs 6, where the exact rank test is arithmetically incapable of
  surviving FDR (see :func:`bh_feasibility`).
* :func:`nested_mixed_model` — a per-feature linear mixed model
  ``intensity ~ group * compartment + (1 | subject)``, so the subject random intercept
  encodes the within-subject correlation and the interaction terms answer "does the
  group difference differ by compartment".

Both consume a ``(n_observations × n_features)`` matrix whose rows are already
**pseudobulk** — one row per subject × compartment. Building that matrix from pixels is
:func:`smile_msi.cohort.pseudobulk_table`'s job; this module stays pure numpy/pandas so it
is unit-testable without Qt, imzML, or a real cohort.

The small-n honesty guard, :func:`bh_feasibility`, is the piece that is easy to omit and
expensive to omit: an unbalanced 3-vs-6 rank test *passes* the usual "≥2 per group" gate,
returns well-formed q-values, and cannot reject anything after BH across a few hundred
features. Without the guard that reads as "no biological difference" when it means "no
possible answer".
"""
from __future__ import annotations

import warnings

import numpy as np

from .spatial import _bh_fdr, _fold_tau

# Degrees of freedom are capped rather than left infinite: ``scipy.stats.t`` with df = inf is
# the standard normal, but numpy's inf propagates into the DataFrame column and prints as "inf"
# in the GUI table. A df this large is numerically indistinguishable from the normal limit.
_DF_INF = 1e10


# --------------------------------------------------------------------------- #
# Small-n feasibility — what CAN this design detect?
# --------------------------------------------------------------------------- #
def exact_rank_p_floor(n_a: int, n_b: int) -> float:
    """Smallest two-sided p an exact Mann-Whitney/Wilcoxon rank-sum test can return for
    group sizes ``n_a`` vs ``n_b``.

    With no ties the rank test's null distribution has exactly ``C(n_a+n_b, n_a)`` equally
    likely label assignments. Complete separation — the best any feature can do — puts one
    assignment in each tail, so the one-sided exact p is ``1/C(N, n_a)`` and the two-sided p
    is twice that. No amount of effect size beats this floor: it is a property of the
    *design*, not the data.

    The consequence people miss: 3-vs-6 gives ``2/C(9,3) = 2/84 ≈ 0.024``. That clears an
    uncorrected 0.05, so a "≥2 per group" gate waves it through — but see
    :func:`bh_feasibility` for what happens next.
    """
    from math import comb

    n_a, n_b = int(n_a), int(n_b)
    if n_a < 1 or n_b < 1:
        return 1.0
    return float(min(1.0, 2.0 / comb(n_a + n_b, n_a)))


def bh_feasibility(n_a: int, n_b: int, n_features: int, alpha: float = 0.05) -> dict:
    """Can an exact rank test on ``n_a`` vs ``n_b`` survive BH-FDR across ``n_features``?

    Benjamini-Hochberg rejects the most significant feature only when
    ``p_(1) · n_features / 1 ≤ alpha``. Substituting the design's exact p floor
    (:func:`exact_rank_p_floor`) for ``p_(1)`` gives the best case the design admits, so
    the whole contrast is decidable in advance:

        ``max_features = floor(alpha / p_floor)``

    Returns ``{p_floor, max_features, feasible, alpha, n_a, n_b, n_features}`` and, when
    infeasible, a ``reason`` string fit to show the user. A 3-vs-6 contrast over 500
    features has ``p_floor = 0.024`` and ``max_features = 2`` — it cannot return a single
    FDR-significant hit, whatever the biology. The remedy is not a different rank test; it
    is a test that borrows information across features (:func:`moderated_t_two_group`) or
    models the design (:func:`nested_mixed_model`), each of which trades an assumption for
    the power the permutation count refuses to supply.
    """
    floor = exact_rank_p_floor(n_a, n_b)
    max_features = int(np.floor(alpha / floor)) if floor > 0 else 0
    feasible = bool(n_features <= max_features)
    out = {"p_floor": floor, "max_features": max_features, "feasible": feasible,
           "alpha": float(alpha), "n_a": int(n_a), "n_b": int(n_b),
           "n_features": int(n_features)}
    if not feasible:
        out["reason"] = (
            f"an exact rank test on {n_a} vs {n_b} cannot return p below {floor:.3g}, so across "
            f"{n_features} features nothing can reach q≤{alpha:g} (BH admits at most "
            f"{max_features} feature{'' if max_features == 1 else 's'} at this n). Use the "
            f"moderated t-test or the mixed model, or report effect sizes without inference.")
    return out


# --------------------------------------------------------------------------- #
# Empirical-Bayes variance shrinkage (limma / Smyth 2004)
# --------------------------------------------------------------------------- #
def _trigamma_inverse(x: np.ndarray) -> np.ndarray:
    """Solve ``trigamma(y) = x`` for ``y``, elementwise (Newton on ``1/y``-scaled steps).

    Mirrors limma's ``trigammaInverse``. The asymptotes are handled in closed form —
    ``trigamma(y) ≈ 1/y`` for tiny ``y`` and ``≈ 1/sqrt(x)`` for huge ``x`` — because
    Newton is ill-conditioned there.
    """
    from scipy.special import polygamma

    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    big, small = x > 1e7, x < 1e-6
    out[big] = 1.0 / np.sqrt(x[big])
    out[small] = 1.0 / x[small]
    mid = ~(big | small | np.isnan(x))
    if not mid.any():
        return out
    xm = x[mid]
    y = 0.5 + 1.0 / xm                      # limma's starting value
    for _ in range(50):
        tri = polygamma(1, y)
        # Newton step on trigamma(y) - x = 0, written via the tetragamma derivative.
        dif = tri * (1.0 - tri / xm) / polygamma(2, y)
        y = y + dif
        if np.max(-dif / y) < 1e-8:
            break
    out[mid] = y
    return out


def fit_f_dist(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Moment-match a scaled inverse-chi-square prior to the per-feature residual variances.

    Given ``s2[g] ~ s2_prior · chisq(df_prior) / df_prior`` scaled by the true variance,
    Smyth (2004) §6 matches the first two moments of ``log s2`` — whose sampling mean and
    variance are ``digamma(df/2) - log(df/2)`` and ``trigamma(df/2)`` — to the observed
    spread. Returns ``(df_prior, s2_prior)``.

    When the observed spread of ``log s2`` is no wider than the sampling noise alone
    (``evar ≤ 0``), the features are consistent with a *single* common variance: the prior
    carries infinite weight, ``df_prior = inf``, and every feature shrinks all the way to
    ``s2_prior``. In that branch the prior variance is the **arithmetic mean** of the
    observed ``s2`` (limma's ``fitFDist``), not ``exp(emean)`` — the log-scale bias correction
    that ``emean`` carries is only unbiased for the *scale* of the F distribution, so reusing
    it here would inflate the common variance by ``exp(log(df/2) - digamma(df/2))`` (≈ 16% at
    df = 7) and silently make every test conservative.

    Zero and non-finite variances are excluded from the fit (``log 0``), exactly as limma's
    ``fitFDist`` does.
    """
    from scipy.special import digamma, polygamma

    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    n = int(ok.sum())
    if n == 0:
        return float("inf"), float("nan")
    if n == 1:
        return float("inf"), float(s2[ok][0])
    z = np.log(s2[ok])
    dfo = df[ok]
    e = z - digamma(dfo / 2.0) + np.log(dfo / 2.0)     # unbiased for log(s2_prior)
    emean = float(e.mean())
    evar = float(((e - emean) ** 2).sum() / (n - 1)) - float(np.mean(polygamma(1, dfo / 2.0)))
    if evar > 0:
        df_prior = float(2.0 * _trigamma_inverse(np.array([evar]))[0])
        s2_prior = float(np.exp(emean + digamma(df_prior / 2.0) - np.log(df_prior / 2.0)))
    else:
        df_prior = float("inf")
        s2_prior = float(np.mean(s2[ok]))
    return df_prior, s2_prior


def squeeze_var(s2: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Shrink per-feature residual variances toward a fitted prior (limma's ``squeezeVar``).

    ``s2_post = (df_prior · s2_prior + df · s2) / (df_prior + df)`` — a precision-weighted
    blend of each feature's own noisy variance and the variance typical of the assay. The
    posterior is what makes a moderated t usable at n = 3: a feature that happened to draw a
    tiny residual variance can no longer manufacture a huge t.

    Returns ``(s2_post, df_prior, s2_prior)``. Features with a non-finite or zero ``s2``
    still receive the prior (they contributed nothing to fitting it).
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    df_prior, s2_prior = fit_f_dist(s2, df)
    if not np.isfinite(s2_prior):
        return np.full(s2.shape, np.nan), df_prior, s2_prior
    if not np.isfinite(df_prior):
        return np.full(s2.shape, s2_prior), df_prior, s2_prior
    s2_use = np.where(np.isfinite(s2) & (s2 > 0), s2, 0.0)
    df_use = np.where(np.isfinite(df) & (df > 0), df, 0.0)
    s2_post = (df_prior * s2_prior + df_use * s2_use) / (df_prior + df_use)
    return s2_post, df_prior, s2_prior


def moderated_t_two_group(XA: np.ndarray, XB: np.ndarray, ids=None, id_col: str = "mz",
                          min_n: int = 2):
    """Per-feature moderated (empirical-Bayes) two-sample t-test, B versus A.

    ``XA``/``XB`` are ``(n_replicates × n_features)`` **pseudobulk** matrices — one row per
    subject, never per pixel. NaN cells (a subject missing that feature) drop out per
    feature, so ``n_A``/``n_B`` are per-feature counts.

    Each feature contributes a pooled residual variance ``s2`` on ``df = n_A + n_B - 2``.
    Those are shrunk toward a common prior by :func:`squeeze_var`, and the moderated
    statistic ``t = (mean_B - mean_A) / (sqrt(s2_post) · sqrt(1/n_A + 1/n_B))`` is referred to
    ``t`` on ``df + df_prior`` degrees of freedom (Smyth 2004, eq. 9-10). The extra
    ``df_prior`` degrees of freedom are exactly the power that pooling across features buys.

    Returns a DataFrame with ``{id_col}, n_A, n_B, mean_A, mean_B, effect, s2_raw, s2_post,
    t, df_total, p_value, q_value, tested``. ``effect`` is the difference of means on the
    matrix's own scale — a log2 fold-change when the caller passed log2-transformed
    intensities, which is the intended use. BH runs over the tested features only, matching
    :func:`smile_msi.cohort.group_comparison`, so sparse untestable features cannot pad the
    denominator and hide a real hit. ``df.attrs`` carries ``df_prior``, ``s2_prior``,
    ``test`` and ``n_testable``.
    """
    import pandas as pd
    from scipy.stats import t as t_dist

    XA = np.atleast_2d(np.asarray(XA, dtype=float))
    XB = np.atleast_2d(np.asarray(XB, dtype=float))
    if XA.shape[1] != XB.shape[1]:
        raise ValueError("XA and XB must have the same number of feature columns")
    nfeat = XA.shape[1]
    min_n = max(2, int(min_n))
    if ids is None:
        ids = np.arange(nfeat, dtype=float)
    ids = np.asarray(ids)
    if ids.size != nfeat:
        raise ValueError(f"ids has {ids.size} entries but the matrices have {nfeat} features")

    nA = np.zeros(nfeat, dtype=int)
    nB = np.zeros(nfeat, dtype=int)
    mA = np.full(nfeat, np.nan)
    mB = np.full(nfeat, np.nan)
    s2 = np.full(nfeat, np.nan)
    dfr = np.zeros(nfeat)
    unscaled = np.full(nfeat, np.nan)          # sqrt(1/nA + 1/nB)
    tested = np.zeros(nfeat, dtype=bool)
    for j in range(nfeat):
        a = XA[:, j]; a = a[np.isfinite(a)]
        b = XB[:, j]; b = b[np.isfinite(b)]
        nA[j], nB[j] = a.size, b.size
        if a.size:
            mA[j] = float(a.mean())
        if b.size:
            mB[j] = float(b.mean())
        if a.size < min_n or b.size < min_n:
            continue
        d = a.size + b.size - 2
        # Pooled (Student) residual variance — the moderation acts on this, so an unequal-
        # variance Welch estimate would have no single df to shrink against.
        sp = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / d
        s2[j] = float(sp)
        dfr[j] = float(d)
        unscaled[j] = float(np.sqrt(1.0 / a.size + 1.0 / b.size))
        tested[j] = True

    s2_post, df_prior, s2_prior = squeeze_var(np.where(tested, s2, np.nan),
                                              np.where(tested, dfr, np.nan))
    effect = mB - mA
    tval = np.full(nfeat, np.nan)
    pval = np.full(nfeat, np.nan)
    df_total = np.full(nfeat, np.nan)
    if np.isfinite(s2_prior):
        dft = np.where(np.isfinite(df_prior), dfr + df_prior, _DF_INF)
        se = np.sqrt(s2_post) * unscaled
        with np.errstate(divide="ignore", invalid="ignore"):
            tj = effect / se
        good = tested & np.isfinite(tj)
        tval[good] = tj[good]
        df_total[good] = dft[good]
        pval[good] = 2.0 * t_dist.sf(np.abs(tj[good]), dft[good])
        # A feature whose posterior variance is still exactly zero (prior and data both zero)
        # yields t = ±inf / nan; it carries no usable inference, so it stays untested rather
        # than entering the BH denominator with a fabricated p.
        tested = good
    else:
        # No feature had an estimable variance, so no prior exists and nothing is testable.
        # Leaving `tested` set here would run BH over a column of NaN p-values.
        tested = np.zeros(nfeat, dtype=bool)

    qval = np.full(nfeat, np.nan)
    if tested.any():
        qval[tested] = _bh_fdr(pval[tested])

    df = pd.DataFrame({id_col: ids, "n_A": nA, "n_B": nB, "mean_A": mA, "mean_B": mB,
                       "effect": effect, "s2_raw": s2, "s2_post": s2_post, "t": tval,
                       "df_total": df_total, "p_value": pval, "q_value": qval,
                       "tested": tested})
    df.attrs["test"] = "moderated t (empirical-Bayes, limma/Smyth 2004)"
    df.attrs["df_prior"] = df_prior
    df.attrs["s2_prior"] = s2_prior
    df.attrs["n_features"] = int(nfeat)
    df.attrs["n_testable"] = int(tested.sum())
    return df


# --------------------------------------------------------------------------- #
# Per-feature linear mixed model:  y ~ group * compartment + (1 | subject)
# --------------------------------------------------------------------------- #
def _nested_design(group_bin: np.ndarray, comp_idx: np.ndarray, n_comp: int):
    """Fixed-effects design for ``y ~ group * compartment`` with treatment coding.

    Columns, in order: ``Intercept``, ``group``, ``comp[k]`` for ``k = 1..K-1``, then
    ``group:comp[k]`` for the same ``k``. Compartment 0 is the reference, so the ``group``
    coefficient is the group effect *in the reference compartment* — a simple effect, not a
    main effect. Building the matrix here rather than through a patsy formula keeps the
    column order known, which is what makes the contrast vectors below readable and testable.
    """
    n = group_bin.size
    dummies = np.zeros((n, max(0, n_comp - 1)))
    for k in range(1, n_comp):
        dummies[:, k - 1] = (comp_idx == k).astype(float)
    inter = dummies * group_bin[:, None]
    return np.column_stack([np.ones(n), group_bin, dummies, inter])


def _bw_dfs(n_obs: int, n_sub: int, n_fixed: int, n_between: int = 2):
    """Between-within denominator degrees of freedom (SAS ``ddfm=betwithin``; nlme's default).

    statsmodels' ``MixedLM`` reports Wald **z** p-values, which are anti-conservative when the
    number of subjects is small — precisely our case. Splitting the residual df by whether a
    contrast varies *between* subjects (group) or *within* them (compartment, interaction)
    recovers the classical repeated-measures df and refers the statistic to ``t``/``F``
    instead of the normal. For 9 subjects × 3 compartments this gives 7 df for the group
    effect and 14 for the interaction, versus the normal's implicit infinity.

    This is an approximation, not Satterthwaite — it is conservative relative to the z-test,
    which is the direction an honest small-n analysis should err in.
    """
    df_between = max(1.0, float(n_sub - n_between))
    df_within = max(1.0, float(n_obs - n_sub - (n_fixed - n_between)))
    return df_between, df_within


def _wald(beta, vcov, contrast, df):
    """Two-sided Wald t-test of ``contrast · beta = 0`` → ``(effect, se, t, p)``."""
    from scipy.stats import t as t_dist

    c = np.asarray(contrast, dtype=float)
    eff = float(c @ beta)
    var = float(c @ vcov @ c)
    if not np.isfinite(var) or var <= 0:
        return eff, np.nan, np.nan, np.nan
    se = float(np.sqrt(var))
    tv = eff / se
    return eff, se, tv, float(2.0 * t_dist.sf(abs(tv), df))


def _wald_joint(beta, vcov, R, df_den):
    """Joint Wald F-test of ``R · beta = 0`` → p, on ``(rank(R), df_den)`` df."""
    from scipy.stats import f as f_dist

    R = np.atleast_2d(np.asarray(R, dtype=float))
    if R.size == 0:
        return np.nan
    rb = R @ beta
    mid = R @ vcov @ R.T
    try:
        sol = np.linalg.solve(mid, rb)
    except np.linalg.LinAlgError:
        return np.nan
    q = int(np.linalg.matrix_rank(R))
    if q == 0:
        return np.nan
    stat = float(rb @ sol) / q
    if not np.isfinite(stat) or stat < 0:
        return np.nan
    return float(f_dist.sf(stat, q, df_den))


def nested_mixed_model(X: np.ndarray, group, compartment, subject, group_a: str, group_b: str,
                       ids=None, id_col: str = "mz", compartments=None,
                       min_subjects_per_group: int = 2, df_method: str = "bw",
                       raw: np.ndarray | None = None):
    """Per-feature ``intensity ~ group * compartment + (1 | subject)`` mixed model.

    ``X`` is the ``(n_observations × n_features)`` pseudobulk matrix — one row per
    subject × compartment, on the **modelling scale** (log2 intensity; see
    :func:`smile_msi.cohort.pseudobulk_table`). ``group``, ``compartment`` and ``subject``
    are row-aligned label arrays. ``raw``, when given, is the same matrix on the *raw*
    intensity scale and is used only to report a τ-regularized ``log2_fc`` consistent with
    every other fold metric in the app (:func:`smile_msi.spatial._fold_tau`).

    The subject random intercept is what makes the three compartments of one nerve count as
    one nerve. Without it the model would treat 27 profiles as 27 independent samples and
    reproduce the pseudoreplication it exists to prevent.

    Per feature the model yields, via Wald contrasts on the fixed effects:

    * ``p_interaction`` — joint test of all ``group:compartment`` terms. "Does the
      normal-vs-treated difference *itself* differ across compartments?"
    * ``p_group`` — the group effect averaged over compartments (the mean simple effect,
      which is the interpretable main effect under treatment coding).
    * ``effect__<c>`` / ``se__<c>`` / ``p__<c>`` — the group's **simple effect within each
      compartment** ``c``, i.e. normal vs treated in the endoneurium alone.

    Each family of p-values is BH-adjusted independently across features (``q_interaction``,
    ``q_group``, ``q__<c>``), because they answer different questions and are reported
    separately. Denominator df follow :func:`_bw_dfs` unless ``df_method='z'``.

    Features whose model fails to converge, or whose surviving rows can't estimate the design
    (a group short of ``min_subjects_per_group`` subjects, or a compartment gone), come back
    all-NaN with ``converged=False`` and are excluded from every BH denominator.

    ``df.attrs`` records ``compartments``, ``ref_compartment``, ``n_subjects``,
    ``n_observations``, ``df_between``, ``df_within``, ``test`` and ``n_testable``.
    """
    import pandas as pd
    try:
        import statsmodels.api as sm
    except ImportError as exc:                  # pragma: no cover - hard dependency
        raise ImportError(
            "the nested mixed model needs statsmodels; install it with "
            "`uv pip install statsmodels` (it ships with SMILE-MSI by default)") from exc

    X = np.atleast_2d(np.asarray(X, dtype=float))
    group = np.asarray(group, dtype=object)
    compartment = np.asarray(compartment, dtype=object)
    subject = np.asarray(subject, dtype=object)
    n_obs, nfeat = X.shape
    if not (group.size == compartment.size == subject.size == n_obs):
        raise ValueError("group / compartment / subject must be row-aligned with X")
    if ids is None:
        ids = np.arange(nfeat, dtype=float)
    ids = np.asarray(ids)
    if ids.size != nfeat:
        raise ValueError(f"ids has {ids.size} entries but X has {nfeat} feature columns")

    keep = np.isin(group, [group_a, group_b])
    if not keep.any():
        raise ValueError(f"no rows labelled {group_a!r} or {group_b!r}")
    X, group, compartment, subject = X[keep], group[keep], compartment[keep], subject[keep]
    raw = raw[keep] if raw is not None else None

    if compartments is None:
        compartments = sorted({str(c) for c in compartment})
    compartments = [str(c) for c in compartments]
    n_comp = len(compartments)
    if n_comp < 1:
        raise ValueError("need at least one compartment")
    comp_pos = {c: k for k, c in enumerate(compartments)}
    in_comp = np.array([str(c) in comp_pos for c in compartment])
    X, group, compartment, subject = (X[in_comp], group[in_comp], compartment[in_comp],
                                      subject[in_comp])
    raw = raw[in_comp] if raw is not None else None

    group_bin = (group == group_b).astype(float)          # 0 = A (reference), 1 = B
    comp_idx = np.array([comp_pos[str(c)] for c in compartment], dtype=int)
    n_fixed = 1 + 1 + 2 * (n_comp - 1)
    tau = _fold_tau(raw) if raw is not None and raw.size else 1.0

    # Contrast vectors over the fixed-effect columns of _nested_design.
    #   [0] intercept · [1] group · [2 : 2+K-1] comp dummies · [2+K-1 : ] interactions
    i0 = 2 + (n_comp - 1)
    simple = []                                            # group effect within compartment k
    for k in range(n_comp):
        c = np.zeros(n_fixed)
        c[1] = 1.0
        if k > 0:
            c[i0 + (k - 1)] = 1.0
        simple.append(c)
    c_group = np.mean(np.vstack(simple), axis=0)           # mean simple effect over compartments
    R_inter = np.zeros((n_comp - 1, n_fixed))
    for k in range(1, n_comp):
        R_inter[k - 1, i0 + (k - 1)] = 1.0

    cols: dict[str, np.ndarray] = {
        "p_interaction": np.full(nfeat, np.nan), "p_group": np.full(nfeat, np.nan),
        "effect_group": np.full(nfeat, np.nan), "se_group": np.full(nfeat, np.nan),
    }
    for c in compartments:
        for pre in ("effect", "se", "t", "p", "log2_fc", "n_A", "n_B"):
            cols[f"{pre}__{c}"] = np.full(nfeat, np.nan)
    converged = np.zeros(nfeat, dtype=bool)
    df_between = df_within = np.nan

    for j in range(nfeat):
        y = X[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        yj, gj, cj, sj = y[ok], group_bin[ok], comp_idx[ok], subject[ok]
        # Estimability: both groups need enough SUBJECTS (not rows — three compartments of one
        # nerve are one nerve), and every modelled compartment must still be present.
        subs_a = {s for s, g in zip(sj, gj) if g == 0.0}
        subs_b = {s for s, g in zip(sj, gj) if g == 1.0}
        if min(len(subs_a), len(subs_b)) < max(2, int(min_subjects_per_group)):
            continue
        if len({int(k) for k in cj}) < n_comp:
            continue
        D = _nested_design(gj, cj, n_comp)
        if np.linalg.matrix_rank(D) < D.shape[1]:
            continue                                       # collinear (e.g. a group×comp cell empty)
        n_sub = len({str(s) for s in sj})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            # convergence / singular-RE chatter
                res = sm.MixedLM(yj, D, groups=sj).fit(reml=True)
        except Exception:  # noqa: BLE001 — a single ill-conditioned feature must not kill the run
            continue
        if not bool(getattr(res, "converged", True)):
            continue
        beta = np.asarray(res.fe_params, dtype=float)
        # cov_params() spans fixed effects AND the random-effect variance parameters; the Wald
        # contrasts touch fixed effects only, so take the leading block.
        V = np.asarray(res.cov_params(), dtype=float)[:n_fixed, :n_fixed]
        if beta.size != n_fixed or not np.all(np.isfinite(V)):
            continue

        dfb, dfw = _bw_dfs(int(ok.sum()), n_sub, n_fixed)
        if df_method == "z":
            dfb = dfw = _DF_INF
        df_between, df_within = dfb, dfw

        cols["p_interaction"][j] = (_wald_joint(beta, V, R_inter, dfw)
                                    if n_comp > 1 else np.nan)
        eff, se, _t, p = _wald(beta, V, c_group, dfb)
        cols["effect_group"][j], cols["se_group"][j], cols["p_group"][j] = eff, se, p
        for k, cname in enumerate(compartments):
            eff, se, tv, p = _wald(beta, V, simple[k], dfb)
            cols[f"effect__{cname}"][j] = eff
            cols[f"se__{cname}"][j] = se
            cols[f"t__{cname}"][j] = tv
            cols[f"p__{cname}"][j] = p
            sel = ok & (comp_idx == k)
            cols[f"n_A__{cname}"][j] = len({s for s, g, m in zip(subject, group_bin, sel)
                                            if m and g == 0.0})
            cols[f"n_B__{cname}"][j] = len({s for s, g, m in zip(subject, group_bin, sel)
                                            if m and g == 1.0})
            if raw is not None:
                ra = raw[sel & (group_bin == 0.0), j]
                rb = raw[sel & (group_bin == 1.0), j]
                ra, rb = ra[np.isfinite(ra)], rb[np.isfinite(rb)]
                if ra.size and rb.size:
                    cols[f"log2_fc__{cname}"][j] = float(
                        np.log2((rb.mean() + tau) / (ra.mean() + tau)))
        converged[j] = True

    data = {id_col: ids, "converged": converged}
    data.update(cols)
    df = pd.DataFrame(data)
    # One BH family per question. The interaction, the averaged group effect and each
    # compartment's simple effect answer different hypotheses, so pooling them into one
    # denominator would be neither more nor less correct — just uninterpretable.
    q_of = {"p_interaction": "q_interaction", "p_group": "q_group"}
    q_of.update({f"p__{c}": f"q__{c}" for c in compartments})
    for pcol, qcol in q_of.items():
        p = df[pcol].to_numpy(dtype=float)
        q = np.full(nfeat, np.nan)
        ok = np.isfinite(p)
        if ok.any():
            q[ok] = _bh_fdr(p[ok])
        df[qcol] = q

    df.attrs["test"] = ("linear mixed model  y ~ group * compartment + (1 | subject), "
                        f"Wald contrasts on {'between-within' if df_method == 'bw' else 'normal'} df")
    df.attrs["compartments"] = compartments
    df.attrs["ref_compartment"] = compartments[0]
    df.attrs["a_label"], df.attrs["b_label"] = group_a, group_b
    df.attrs["n_subjects"] = len({str(s) for s in subject})
    df.attrs["n_observations"] = int(X.shape[0])
    df.attrs["df_between"], df.attrs["df_within"] = df_between, df_within
    df.attrs["df_method"] = df_method
    df.attrs["n_features"] = int(nfeat)
    df.attrs["n_testable"] = int(converged.sum())
    if not converged.any():
        # Name the structural cause. "No feature could be fitted" over 231 rows of NaN is the
        # least actionable thing this function can say, and a missing compartment (the A/B
        # groups covering only one of them) is by far the commonest reason.
        seen = {str(c) for c in compartment}
        absent = [c for c in compartments if c not in seen]
        if absent:
            why = (f"compartment(s) {', '.join(absent)} are absent from the "
                   f"{group_a}/{group_b} rows, so the group × compartment design is "
                   f"rank-deficient — check that the group labels don't embed the compartment")
        elif len({str(s) for s in subject}) < 2 * max(2, int(min_subjects_per_group)):
            why = (f"only {len({str(s) for s in subject})} subjects resolved — check the "
                   f"subject source")
        else:
            why = (f"the design needs ≥{max(2, int(min_subjects_per_group))} subjects per group "
                   f"and every compartment present in both groups")
        df.attrs["warning"] = f"no feature could be fitted: {why}."
    return df


# --------------------------------------------------------------------------- #
# Competitive set testing with inter-feature correlation (camera; Wu & Smyth 2012)
# --------------------------------------------------------------------------- #
def standardized_residuals(Y: np.ndarray, D: np.ndarray):
    """Column-unit-normalized OLS residuals of ``Y`` ``(n_obs × n_feat)`` on design ``D``.

    Residuals are taken against the **fixed-effects** design only. The subject random intercept
    is deliberately *not* removed: between-subject variation is exactly what couples the ions of
    one lipid class (a nerve rich in myelin is rich in every sulfatide at once), and that
    coupling is the thing :func:`camera` needs to measure. Projecting it away would return
    ``r̄ ≈ 0`` and reinstate the very over-confidence the correlation adjustment exists to fix.

    Returns ``(R, usable)`` where ``R``'s columns have unit norm and ``usable`` marks the
    features with a non-degenerate residual (a feature perfectly fitted by the design has none).
    """
    Y = np.asarray(Y, dtype=float)
    Q, _ = np.linalg.qr(np.asarray(D, dtype=float))
    R = Y - Q @ (Q.T @ Y)
    nrm = np.linalg.norm(R, axis=0)
    usable = np.isfinite(nrm) & (nrm > 1e-12)
    R = R.copy()
    R[:, usable] /= nrm[usable]
    R[:, ~usable] = 0.0
    return R, usable


def inter_feature_correlation(resid_unit: np.ndarray, idx) -> float:
    """Mean pairwise correlation among the columns of ``resid_unit`` selected by ``idx``.

    With unit-norm columns each pairwise correlation is an inner product, so the mean over the
    ``m(m-1)`` ordered off-diagonal pairs is ``(‖Σ cᵢ‖² − m) / (m(m−1))`` — one matrix-vector
    product instead of an ``m × m`` correlation matrix.
    """
    idx = np.asarray(idx)
    m = int(idx.size)
    if m < 2:
        return 0.0
    s = resid_unit[:, idx].sum(axis=1)
    return float((float(s @ s) - m) / (m * (m - 1)))


def rank_sum_test_with_correlation(stat: np.ndarray, in_set: np.ndarray,
                                   correlation: float = 0.0, df: float = np.inf):
    """Wilcoxon rank-sum p-values ``(less, greater)`` for ``in_set`` versus the rest, with the
    null variance inflated for correlation among the set's members.

    Under independence ``Var(U) = n₁n₂(n+1)/12``. Wu & Smyth (2012) derive the variance when the
    set's members share an average correlation ``ρ`` — the ``arcsin`` expression below, which
    reduces exactly to the classical formula at ``ρ = 0``. Faithful to limma's
    ``rankSumTestWithCorrelation``, including the tie correction and continuity correction.

    Ref: Wu, D. & Smyth, G.K. (2012). Camera: a competitive gene set test accounting for
    inter-gene correlation. Nucleic Acids Res. 40(17):e133. doi:10.1093/nar/gks461
    """
    from scipy.stats import rankdata
    from scipy.stats import t as t_dist

    stat = np.asarray(stat, dtype=float)
    in_set = np.asarray(in_set, dtype=bool)
    n = stat.size
    n1 = int(in_set.sum())
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return 1.0, 1.0
    r = rankdata(stat)
    U = n1 * n2 + n1 * (n1 + 1) / 2.0 - float(r[in_set].sum())
    mu = n1 * n2 / 2.0
    if correlation == 0.0 or n1 == 1:
        sigma2 = n1 * n2 * (n + 1) / 12.0
    else:
        c = float(np.clip(correlation, -1.0, 1.0))
        sigma2 = (np.arcsin(1.0)
                  + (n2 - 1) * np.arcsin(0.5)
                  + (n1 - 1) * (n2 - 1) * np.arcsin(c / 2.0)
                  + (n1 - 1) * np.arcsin((c + 1.0) / 2.0))
        sigma2 = n1 * n2 * sigma2 / (2.0 * np.pi)
    # Ties shrink the null variance; without this a lipidome with many equal statistics
    # (rounded, or clipped at a detection floor) would test anti-conservatively.
    _, counts = np.unique(r, return_counts=True)
    if counts.max() > 1:
        adj = float((counts * (counts + 1) * (counts - 1)).sum()) / (n * (n + 1) * (n - 1))
        sigma2 *= (1 - adj)
    if sigma2 <= 0:
        return 1.0, 1.0
    z_lower = (U + 0.5 - mu) / np.sqrt(sigma2)
    z_upper = (U - 0.5 - mu) / np.sqrt(sigma2)
    p_less = float(t_dist.sf(z_upper, df))
    p_greater = float(t_dist.cdf(z_lower, df))
    return p_less, p_greater


def camera(stat, sets: dict, resid_unit=None, inter_feature_cor: float | None = None,
           use_ranks: bool = True, min_vif: float = 1.0, df_residual: float = np.inf,
           min_size: int = 3):
    """Competitive set test adjusted for inter-feature correlation (Wu & Smyth 2012).

    Asks, for each set, whether its features' statistics are shifted **relative to the rest of
    the measured features** — not whether the set differs from zero. The distinction matters
    whenever a nuisance shifts everything at once: a per-slide scale offset moves the set and
    the background equally and cancels here, where a self-contained test would report it as a
    finding.

    The correction it exists for: a lipid class is a chain-length series, whose members are
    produced by the same enzymes and rise and fall together. Treating ``m`` correlated ions as
    ``m`` independent observations shrinks the p-value by roughly ``sqrt(VIF)``, where
    ``VIF = 1 + (m-1)·r̄``. At ``m = 19`` and a modest ``r̄ = 0.2`` that is a factor of 2.1 —
    the difference between ``p = 0.03`` and ``p = 0.30``. This is pixel-vs-subject
    pseudoreplication moved one level over, from observations to features.

    ``r̄`` is estimated per set from ``resid_unit`` (:func:`standardized_residuals`), or fixed
    for every set via ``inter_feature_cor`` (limma's ``camera`` defaults to a 0.01 preset,
    because the estimate is itself noisy). ``min_vif=1.0`` floors the inflation at "no
    adjustment": a negative ``r̄`` would make the test *more* significant than the independent
    case, and a homologous series cannot plausibly be negatively correlated, so an estimate
    below zero is noise rather than evidence. Set ``min_vif=0`` to let it through.

    ``use_ranks=True`` (default) runs :func:`rank_sum_test_with_correlation`; the statistics of
    a small-n contrast have heavy tails and a rank test does not care. ``False`` runs limma's
    parametric two-sample t on the statistics, which is more powerful when they are well-behaved.

    Returns one row per set with ``n_features, direction, mean_stat_in, mean_stat_out, r_bar,
    r_bar_used, vif, n_effective, p_value, p_uncorrected, q_value`` — ``p_uncorrected`` is the
    same test at ``r̄ = 0``, reported so the size of the adjustment is visible rather than
    implied. BH runs across the sets.
    """
    import pandas as pd
    from scipy.stats import t as t_dist

    stat = np.asarray(stat, dtype=float)
    G = stat.size
    finite = np.isfinite(stat)
    rows = []
    var_pooled = float(np.var(stat[finite], ddof=1)) if finite.sum() > 1 else np.nan

    def _p(idx_mask, rbar):
        m = int(idx_mask.sum())
        m2 = int(finite.sum()) - m
        if m < 2 or m2 < 2:
            return np.nan
        if use_ranks:
            p_less, p_greater = rank_sum_test_with_correlation(
                stat[finite], idx_mask[finite], correlation=rbar,
                df=min(df_residual, G - 2))
            return float(min(1.0, 2 * min(p_less, p_greater)))
        vif = 1.0 + (m - 1) * rbar
        if vif <= 0 or not np.isfinite(var_pooled) or var_pooled <= 0:
            return np.nan
        d = stat[finite & idx_mask].mean() - stat[finite & ~idx_mask].mean()
        se = np.sqrt(var_pooled * (vif / m + 1.0 / m2))
        return float(2 * t_dist.sf(abs(d / se), min(df_residual, G - 2)))

    for name, idx in sets.items():
        mask = np.zeros(G, dtype=bool)
        mask[np.asarray(idx, dtype=int)] = True
        mask &= finite
        m = int(mask.sum())
        if m < min_size:
            continue
        if inter_feature_cor is not None:
            r_bar = float(inter_feature_cor)
        elif resid_unit is not None:
            r_bar = inter_feature_correlation(resid_unit, np.flatnonzero(mask))
        else:
            r_bar = 0.0
        # Floor the inflation, not the correlation itself — we still report the raw estimate.
        r_floor = (min_vif - 1.0) / (m - 1) if m > 1 else 0.0
        r_used = max(r_bar, r_floor)
        vif = 1.0 + (m - 1) * r_used
        mean_in = float(stat[mask].mean())
        mean_out = float(stat[finite & ~mask].mean())
        rows.append({
            "set": name, "n_features": m,
            "direction": "down" if mean_in < mean_out else "up",
            "mean_stat_in": mean_in, "mean_stat_out": mean_out,
            "r_bar": r_bar, "r_bar_used": r_used, "vif": vif, "n_effective": m / vif,
            "p_value": _p(mask, r_used), "p_uncorrected": _p(mask, 0.0),
        })
    df = pd.DataFrame(rows, columns=["set", "n_features", "direction", "mean_stat_in",
                                     "mean_stat_out", "r_bar", "r_bar_used", "vif",
                                     "n_effective", "p_value", "p_uncorrected"])
    q = np.full(len(df), np.nan)
    if len(df):
        ok = np.isfinite(df["p_value"].to_numpy(dtype=float))
        if ok.any():
            q[ok] = _bh_fdr(df["p_value"].to_numpy(dtype=float)[ok])
    df["q_value"] = q
    df.attrs["test"] = ("competitive set test, correlation-adjusted "
                        f"({'rank-sum' if use_ranks else 'parametric t'}; Wu & Smyth 2012)")
    df.attrs["use_ranks"] = use_ranks
    df.attrs["min_vif"] = min_vif
    df.attrs["inter_feature_cor"] = inter_feature_cor
    df.attrs["n_background"] = int(finite.sum())
    return df.sort_values("p_value", kind="stable", na_position="last", ignore_index=True)


# --------------------------------------------------------------------------- #
# Stratified (per-compartment) two-group testing
# --------------------------------------------------------------------------- #
def stratified_comparison(X: np.ndarray, group, compartment, subject, group_a: str,
                          group_b: str, ids=None, id_col: str = "mz", compartments=None,
                          method: str = "modt", min_n: int = 2,
                          raw: np.ndarray | None = None):
    """Two-group test **within each compartment separately**, one subject per row.

    The simpler framing of :func:`nested_mixed_model`: for each compartment, take the
    subjects that have it, compare group A vs B across those subjects, and BH-adjust across
    features *within that compartment*. Because the compartments come from the same subjects
    the three contrasts are not independent of each other — interpret them jointly, and use
    :func:`nested_mixed_model` when the question is explicitly "does the difference differ
    between compartments".

    ``method``: ``'modt'`` (moderated t, the default and the only one with real power at
    n = 3 vs 6), ``'welch'``, ``'student'``, or ``'mwu'``. The rank option exists so the
    honest comparison can be shown rather than argued about — pair it with
    :func:`bh_feasibility`, which will usually explain why it returns nothing.

    Returns one row per feature, wide over compartments: ``effect__<c>``, ``p__<c>``,
    ``q__<c>``, ``log2_fc__<c>``, ``n_A__<c>``, ``n_B__<c>``. ``df.attrs['per_compartment']``
    maps each compartment to that stratum's own attrs (``df_prior``, ``n_testable``, and the
    :func:`bh_feasibility` verdict for its actual group sizes).
    """
    import pandas as pd

    X = np.atleast_2d(np.asarray(X, dtype=float))
    group = np.asarray(group, dtype=object)
    compartment = np.asarray(compartment, dtype=object)
    subject = np.asarray(subject, dtype=object)
    nfeat = X.shape[1]
    if ids is None:
        ids = np.arange(nfeat, dtype=float)
    ids = np.asarray(ids)
    if ids.size != nfeat:
        raise ValueError(f"ids has {ids.size} entries but X has {nfeat} feature columns")
    if compartments is None:
        compartments = sorted({str(c) for c in compartment})
    compartments = [str(c) for c in compartments]

    out = {id_col: ids}
    per_comp: dict[str, dict] = {}
    tau = _fold_tau(raw) if raw is not None and raw.size else 1.0
    for cname in compartments:
        sel = np.array([str(c) == cname for c in compartment])
        a = sel & (group == group_a)
        b = sel & (group == group_b)
        XA, XB = X[a], X[b]
        n_a = len({str(s) for s in subject[a]})
        n_b = len({str(s) for s in subject[b]})
        if method == "modt":
            sub = moderated_t_two_group(XA, XB, ids=ids, id_col=id_col, min_n=min_n)
            eff, p, q = (sub["effect"].to_numpy(), sub["p_value"].to_numpy(),
                         sub["q_value"].to_numpy())
            attrs = {"df_prior": sub.attrs["df_prior"], "s2_prior": sub.attrs["s2_prior"],
                     "n_testable": sub.attrs["n_testable"], "test": sub.attrs["test"]}
        else:
            from .spatial import _auc_mwu, _two_group_p

            _auc, mwu_p = _auc_mwu(XA, XB)
            p, test_name = _two_group_p(XA, XB, method, mwu_p)
            p = np.asarray(p, dtype=float)
            ok = (np.isfinite(p) & (np.sum(np.isfinite(XA), axis=0) >= min_n)
                  & (np.sum(np.isfinite(XB), axis=0) >= min_n))
            p = np.where(ok, p, np.nan)
            q = np.full(nfeat, np.nan)
            if ok.any():
                q[ok] = _bh_fdr(p[ok])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column → NaN mean
                eff = np.nanmean(XB, axis=0) - np.nanmean(XA, axis=0)
            attrs = {"n_testable": int(np.isfinite(p).sum()), "test": test_name}
        out[f"effect__{cname}"] = eff
        out[f"p__{cname}"] = p
        out[f"q__{cname}"] = q
        out[f"n_A__{cname}"] = np.full(nfeat, n_a)
        out[f"n_B__{cname}"] = np.full(nfeat, n_b)
        fc = np.full(nfeat, np.nan)
        if raw is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ra, rb = np.nanmean(raw[a], axis=0), np.nanmean(raw[b], axis=0)
            fc = np.log2((rb + tau) / (ra + tau))
        out[f"log2_fc__{cname}"] = fc
        attrs["n_A"], attrs["n_B"] = n_a, n_b
        attrs["feasibility"] = bh_feasibility(n_a, n_b, nfeat)
        per_comp[cname] = attrs

    df = pd.DataFrame(out)
    df.attrs["compartments"] = compartments
    df.attrs["a_label"], df.attrs["b_label"] = group_a, group_b
    df.attrs["per_compartment"] = per_comp
    df.attrs["method"] = method
    df.attrs["test"] = next(iter(per_comp.values()))["test"] if per_comp else ""
    df.attrs["n_features"] = int(nfeat)
    df.attrs["n_subjects"] = len({str(s) for s in subject})
    df.attrs["stratified"] = True
    # The rank path is the one that silently cannot reject; surface it once, up front.
    if method == "mwu":
        bad = [c for c, a in per_comp.items() if not a["feasibility"]["feasible"]]
        if bad:
            df.attrs["warning"] = per_comp[bad[0]]["feasibility"]["reason"]
    return df
