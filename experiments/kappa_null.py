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
          rho        : per-ratio normalized orthogonal residual (list, same
                       order as d_pairs); ρ_i = ||d_act(i) - κ_i·d_pred(i)|| /
                       ||d_pred(i)||, i.e. how far the actual displacement lies
                       off the predicted line, in units of ||d_pred||. κ pins
                       only the component *along* the prediction and is blind to
                       this residual, so ρ is its complement: κ_i ≈ 1 with
                       ρ_i ≈ 0 means the centroid sits *on* the predicted point.
          rho_mean   : mean of rho over the mixed ratios
          rho_max    : max of rho over the mixed ratios
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

    # Normalized orthogonal residual per mixed ratio. Decompose the actual
    # displacement as d_act = κ·d_pred + d_perp (d_perp ⟂ d_pred); ρ = ||d_perp||
    # / ||d_pred||. Together with κ this fully locates d_act in the (d_act,d_pred)
    # plane: ||d_act||^2 / ||d_pred||^2 = κ^2 + ρ^2.
    kappa_diag = np.diag(kappa_matrix)
    rho = [
        float(np.linalg.norm(d_act[i] - kappa_diag[i] * d_pred[i]) / np.sqrt(denom[i]))
        for i in range(K)
    ]

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
        "rho":        rho,
        "rho_mean":   float(np.mean(rho)),
        "rho_max":    float(np.max(rho)),
    }
