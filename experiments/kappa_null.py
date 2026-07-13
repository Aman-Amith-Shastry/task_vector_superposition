"""
Exact permutation null for the linear-superposition metric κ.

The linearity hypothesis predicts that every mixed-ratio contrast-vector centroid
lands at *its own* convex-combination-predicted position, i.e. κ_r ≈ 1 for each
mixed ratio r. To show this is not a generic property of any displacement, we
test the *specificity* of the label↔prediction correspondence with a permutation
test that mirrors the significance logic of the injection experiment (Sec. 5),
but is enumerated exactly because the permutable set is small.

For the K mixed ratios with a well-defined predicted displacement (the three
pure ratios define the basis and are held fixed; the 1-1-1 ratio has
d_pred = 0 and is excluded), we score each observed displacement d_act(r)
against a *permuted* prediction d_pred(π(r)):

    S(π) = mean_r  <d_act(r), d_pred(π(r))> / <d_pred(π(r)), d_pred(π(r))>

The identity permutation reproduces the ordinary mean κ (S_obs). Under a random
relabeling most centroids are paired with the wrong prediction, dragging S(π)
toward 0 (and negative for near-opposite corners of the simplex). With K = 6
mixed ratios there are 6! = 720 permutations, so we enumerate the null exactly:

    p = #{π : S(π) >= S_obs} / K!            (identity included  ⇒  p >= 1/K!)

This is the exact-enumeration analogue of the 10,000-draw Monte-Carlo
permutation p used for injection (there K = 10 ⇒ 10! ≫ 1e4 forces sampling).
"""

import itertools
import numpy as np


def _sin_angle(a, b):
    """sin of the angle between two vectors; nan if either is ~0."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cos * cos)))


def split_half_consistency(
    ratio_samples,
    pure_ratios,
    n_splits=500,
    seed=0,
    scale_floor=0.05,
    n_boot=1000,
    ci=(2.5, 97.5),
):
    """
    Calibrate the orthogonal residual of the linear prediction against a
    sampling-noise floor, per mixed ratio, using split-half consistency.

    The linearity claim is that the mixed-ratio displacement d_act = μ_r − center
    points along the convex-combination-predicted displacement d_pred. κ pins the
    component *along* d_pred; the complement is the orthogonal residual, which
    equals sin θ(d_act, d_pred) once scaled out. Rather than compare that residual
    to 0 (unachievable in finite samples — d_act is a noisy centroid), we compare
    it to the best agreement the data itself can resolve: split the n samples for
    ratio r into two independent halves, form d^(1) = μ_r^(1) − center and
    d^(2) = μ_r^(2) − center, and take sin θ(d^(1), d^(2)). No prediction can be
    expected to align with d_act better than an independent estimate of d_act
    aligns with itself, so this angle is the noise floor.

    The pure-task centroids (hence `center` and the prediction basis) are held at
    their full-sample values; only ratio r's own centroid is re-estimated per
    split, isolating the sampling noise in μ_r that the residual is compared to.

    Parameters
    ----------
    ratio_samples : dict {ratio_tuple: ndarray (n, dim)}
        Per-sample contrast vectors for ALL ratios (pures + mixed), one layer.
    pure_ratios : list of ratio_tuple
        The three pure ratios, in the order used to build d_pred.
    n_splits : int
        Number of random half-splits per ratio (default 500).
    seed : int
        RNG seed for reproducible splits.
    scale_floor : float
        A mixed ratio is skipped when ||d_pred|| <= scale_floor * (mean pure
        radius), matching the κ guard (excludes 1-1-1).
    n_boot : int
        Number of full-sample bootstrap resamples used to put a confidence band
        on the (full-sample) residual sin_pred. Resamples the n mixed samples
        with replacement, recomputes d_act, and re-scores sin θ(d_act*, d_pred);
        this reflects the sampling variance of the REPORTED magnitude, not the
        floor. Set to 0 to skip (the sin_pred_ci_* keys are then omitted, and
        older result files without them plot exactly as before). Default 1000.
    ci : (float, float)
        Lower/upper percentiles for the bootstrap band (default 2.5/97.5, a 95%
        percentile interval — robust to sin θ's [0,1] boundedness and skew near
        the floor, unlike a symmetric mean ± SD band).

    Returns
    -------
    dict or None
        None if no mixed ratio has a well-defined prediction. Otherwise:
          per_ratio : {ratio_str: {
                sin_pred        : sin θ(d_act, d_pred), full-sample prediction angle
                sin_pred_ci_lo  : lower bootstrap percentile of sin_pred (band on
                                  the reported magnitude; present iff n_boot > 0)
                sin_pred_ci_hi  : upper bootstrap percentile of sin_pred
                sin_floor_mean  : mean sin θ(d^(1), d^(2)) over splits (noise floor)
                sin_floor_std   : std of the floor over splits
                sin_pred_half   : mean sin θ(half-sample d_act, d_pred) — the
                                  prediction angle at the SAME n/2 sample size as
                                  the floor, so the comparison is size-matched
                ratio_to_floor  : sin_pred_half / sin_floor_mean  (≤ ~1 ⇒ prediction
                                  is as accurate as the data can resolve). Uses the
                                  size-matched (half-sample) residual so numerator
                                  and denominator are both built from n/2 samples.
                n_samples       : n used
          }}
          sin_pred_mean      : mean sin_pred over mixed ratios
          sin_floor_mean     : mean noise floor over mixed ratios
          sin_pred_half_mean : mean size-matched prediction angle over ratios
          ratio_to_floor_mean: mean sin_pred / floor over ratios (best case)
          ratio_to_floor_half_mean : mean sin_pred_half / floor over ratios
                               (the size-matched ratio; ~1 ⇒ residual as small
                               as the data can resolve at equal sample size)
          within_floor       : bool, sin_pred_half_mean <= sin_floor_mean. The
                               REPORTED pass/fail. Uses the size-matched residual
                               so both sides are built from n/2 samples and pay
                               the same sqrt(2) sampling-noise penalty — a fair
                               comparison.
          within_floor_full  : bool, sin_pred_mean <= sin_floor_mean. Best-case
                               margin only: the full-sample residual (noise
                               ~1/sqrt(n)) against the half-sample floor (noise
                               ~sqrt(2)/sqrt(n)). Generous by construction, so a
                               pass here is the easy direction; reported to show
                               the residual clears the floor even before the
                               floor's sample-size handicap is removed.
    """
    pures = [tuple(p) for p in pure_ratios]
    pure_c = {p: np.asarray(ratio_samples[p], dtype=float).mean(0) for p in pures}
    center = np.mean([pure_c[p] for p in pures], axis=0)
    scale = float(np.mean([np.linalg.norm(pure_c[p] - center) for p in pures]))
    rng = np.random.default_rng(seed)

    per_ratio = {}
    for ratio, X in ratio_samples.items():
        rt = tuple(ratio)
        if rt in pures:
            continue
        X = np.asarray(X, dtype=float)
        total = float(sum(rt))
        d_pred = np.sum(
            [(n / total) * pure_c[p] for n, p in zip(rt, pures)], axis=0
        ) - center
        if np.linalg.norm(d_pred) <= scale_floor * scale:
            continue  # undefined prediction (e.g. 1-1-1)

        d_act = X.mean(0) - center
        sin_pred = _sin_angle(d_act, d_pred)

        n = X.shape[0]
        half = n // 2
        floor, half_pred = [], []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            a = X[perm[:half]].mean(0) - center
            b = X[perm[half:2 * half]].mean(0) - center
            floor.append(_sin_angle(a, b))
            # size-matched prediction angle: half-sample d_act vs d_pred
            half_pred.append(_sin_angle(a, d_pred))
            half_pred.append(_sin_angle(b, d_pred))
        floor = np.asarray(floor)
        half_pred = np.asarray(half_pred)
        fmean = float(np.nanmean(floor))
        rec = {
            "sin_pred":       sin_pred,
            "sin_floor_mean": fmean,
            "sin_floor_std":  float(np.nanstd(floor)),
            "sin_pred_half":  float(np.nanmean(half_pred)),
            # size-matched residual (n/2) over the n/2 floor, so both sides pay the
            # same sampling-noise penalty; the full-sample sin_pred is kept above.
            "ratio_to_floor": float(np.nanmean(half_pred) / fmean) if fmean > 0 else float("nan"),
            "n_samples":      int(n),
        }

        # Confidence band on the REPORTED (full-sample) residual: resample the n
        # mixed samples with replacement and re-score sin θ(d_act*, d_pred). This
        # captures the sampling variance of the magnitude we plot, distinct from
        # the floor. Percentile interval (robust to sin θ's boundedness/skew).
        if n_boot > 0:
            boot = np.empty(n_boot)
            for bi in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot[bi] = _sin_angle(X[idx].mean(0) - center, d_pred)
            lo, hi = np.nanpercentile(boot, ci)
            rec["sin_pred_ci_lo"] = float(lo)
            rec["sin_pred_ci_hi"] = float(hi)

        per_ratio["-".join(map(str, rt))] = rec

    if not per_ratio:
        return None

    sp = np.array([v["sin_pred"] for v in per_ratio.values()])
    fl = np.array([v["sin_floor_mean"] for v in per_ratio.values()])
    sph = np.array([v["sin_pred_half"] for v in per_ratio.values()])
    return {
        "per_ratio":           per_ratio,
        "sin_pred_mean":       float(np.nanmean(sp)),
        "sin_floor_mean":      float(np.nanmean(fl)),
        "sin_pred_half_mean":  float(np.nanmean(sph)),
        "ratio_to_floor_mean": float(np.nanmean(sp / fl)),
        "ratio_to_floor_half_mean": float(np.nanmean(sph / fl)),
        # Reported pass/fail uses the size-matched residual (n/2 vs n/2 floor).
        "within_floor":        bool(np.nanmean(sph) <= np.nanmean(fl)),
        # Best-case margin: full-sample residual vs half-sample floor.
        "within_floor_full":   bool(np.nanmean(sp) <= np.nanmean(fl)),
    }


def kappa_permutation_null(d_pairs):
    """
    Exact permutation null for mean κ over the mixed ratios.

    Parameters
    ----------
    d_pairs : list of (d_act, d_pred)
        Displacement-from-tri-center vectors for each mixed ratio with a
        well-defined predicted displacement (||d_pred|| not ~ 0). d_act is the
        actual centroid displacement, d_pred the convex-combination-predicted
        displacement, both in the original contrast-vector space.

    Returns
    -------
    dict or None
        None if fewer than two permutable ratios. Otherwise:
          S_obs      : mean κ under the matched (identity) pairing
          p_value    : exact permutation p (fraction of the K! permutations
                       whose mean scored κ is >= S_obs; identity counted)
          n_perm     : K! (size of the exact null)
          null_mean  : mean of S(π) over all permutations
          null_std   : std of S(π) over all permutations
          null_p2_5  : 2.5th percentile of the null (for a shaded band)
          null_p97_5 : 97.5th percentile of the null
    """
    K = len(d_pairs)
    if K < 2:
        return None  # nothing to permute

    d_act  = [np.asarray(a, dtype=float) for a, _ in d_pairs]
    d_pred = [np.asarray(p, dtype=float) for _, p in d_pairs]
    denom  = [float(np.dot(p, p)) for p in d_pred]  # ||d_pred(j)||^2
    # kappa_matrix[i, j] = <d_act(i), d_pred(j)> / ||d_pred(j)||^2
    #   diagonal (i == j) is the ordinary κ for mixed ratio i.
    kappa_matrix = np.array([
        [float(np.dot(d_act[i], d_pred[j])) / denom[j] for j in range(K)]
        for i in range(K)
    ])

    S_obs = float(np.mean(np.diag(kappa_matrix)))

    # Enumerate all K! label permutations exactly.
    idx = np.arange(K)
    stats = np.array([
        kappa_matrix[idx, perm].mean()
        for perm in itertools.permutations(range(K))
    ])

    p_value = float(np.mean(stats >= S_obs))  # identity gives equality ⇒ counted

    return {
        "S_obs":      S_obs,
        "p_value":    p_value,
        "n_perm":     int(stats.size),
        "null_mean":  float(stats.mean()),
        "null_std":   float(stats.std()),
        "null_p2_5":  float(np.percentile(stats, 2.5)),
        "null_p97_5": float(np.percentile(stats, 97.5)),
    }
