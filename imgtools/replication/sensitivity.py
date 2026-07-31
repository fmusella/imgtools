"""Position sensitivity of replication to sub-nuclear localization."""

import numpy as np
from scipy.stats import false_discovery_control


def _fit_nested_models(
    y: np.ndarray,
    T: np.ndarray,
    L: np.ndarray,
) -> dict:
    """Fit the reduced and full position-sensitivity models for one locus.

    The full model is:

        y = mu + alpha * T + sum(beta_lambda * D_lambda) + eps

    where L is categorical and D contains its one-hot-encoded columns. The
    reduced model excludes localization:

        y = mu_r + alpha_r * T + eps

    Args:
        y (np.ndarray): Binary replication states, shape (N,).
        T (np.ndarray): S-phase pseudo-times, shape (N,).
        L (np.ndarray): Nuclear locale assignments, shape (N,).

    Ordinary least squares is written in matrix form as y = design @ parameters.
    The reduced design therefore contains the two columns [1, T], while the
    full design contains [1, T, D]. The column of ones multiplies the intercept.

    Returns:
        dict: Fitted parameters, predictions, reduced-model residuals, residual
        sums of squares and the one-hot-encoded locale matrix D.
    """
    N = len(y)
    locales, L_index = np.unique(L, return_inverse=True)
    K = len(locales)

    # The reduced model contains only an intercept and pseudo-time. Thus,
    # design_reduced has the two columns [1, T] and parameters [mu_r, alpha_r].
    design_reduced = np.column_stack([np.ones(N), T])
    parameters_reduced, *_ = np.linalg.lstsq(
        design_reduced,
        y,
        rcond=None,
    )
    mu_r, alpha_r = parameters_reduced
    yhat_reduced = design_reduced @ parameters_reduced
    residual_reduced = y - yhat_reduced
    RSS_reduced = float(np.sum(residual_reduced ** 2))

    # One-hot encode L in the N x (K - 1) matrix D. The intercept already
    # describes the first locale, so D contains one column for every other
    # locale and estimates beta relative to that reference.
    D = np.zeros((N, K - 1))
    for k in range(1, K):
        D[:, k - 1] = (L_index == k).astype(float)

    # The full design contains [1, T, D], corresponding to the parameters
    # [mu, alpha, beta]. Least squares finds the values minimizing RSS_full.
    design_full = np.column_stack([np.ones(N), T, D])
    parameters_full, *_ = np.linalg.lstsq(
        design_full,
        y,
        rcond=None,
    )
    mu = parameters_full[0]
    alpha = parameters_full[1]
    beta = parameters_full[2:]
    yhat_full = design_full @ parameters_full
    RSS_full = float(np.sum((y - yhat_full) ** 2))

    # The permutation calculation uses all K locales rather than a reference
    # locale. Here rows are locales and columns are the N allele observations.
    locale_membership = np.zeros((K, N))
    locale_membership[L_index, np.arange(N)] = 1.0

    return {
        'mu': mu,
        'alpha': alpha,
        'beta': beta,
        'mu_r': mu_r,
        'alpha_r': alpha_r,
        'design_reduced': design_reduced,
        'yhat_reduced': yhat_reduced,
        'yhat_full': yhat_full,
        'residual_reduced': residual_reduced,
        'RSS_reduced': RSS_reduced,
        'RSS_full': RSS_full,
        'locale_membership': locale_membership,
    }


def _calculate_sensitivity(fit: dict) -> float:
    """Calculate position sensitivity from the two nested model fits.

    Args:
        fit (dict): Output of _fit_nested_models for one locus.

    Returns:
        float: Square root of the reduced-model residual variation explained
        by adding locale to the model.
    """
    # Because the models are nested, RSS_reduced - RSS_full is exactly the
    # variation explained by adding locale to the pseudo-time-only model.
    explained_variation = fit['RSS_reduced'] - fit['RSS_full']

    # Divide by all variation left after correcting for pseudo-time, then take
    # the square root to obtain the position-sensitivity score in the Methods.
    sensitivity_squared = explained_variation / fit['RSS_reduced']
    return float(np.sqrt(max(sensitivity_squared, 0.0)))


def _permutation_null(
    fit: dict,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calculate the position-sensitivity permutation null for one locus.

    The Methods describe shuffling L and refitting the full model 10,000 times.
    This function performs the same calculation in batches. After removing the
    reduced model from y and D, the locale contribution for one permutation is:

        locale_SS = g.T @ pinv(G) @ g

    where g contains the sums of reduced-model residuals in each shuffled
    locale, and G is the locale design after removing the intercept and T.
    This is the partial-regression form of the same nested least-squares model.

    Args:
        fit (dict): Output of _fit_nested_models for one locus.
        n_permutations (int): Number of locale-label permutations.
        rng (np.random.Generator): Generator shared across loci.

    Returns:
        np.ndarray: Null position-sensitivity values, shape (n_permutations,).
    """
    locale_membership = fit['locale_membership']
    locale_counts = locale_membership.sum(axis=1)
    N = locale_membership.shape[1]

    # QR decomposition gives an orthonormal basis spanning [1, T]. It will be
    # used below to remove the intercept and pseudo-time from locale membership.
    reduced_basis, _ = np.linalg.qr(fit['design_reduced'])

    # Each row gives one random reordering of the N observations. Shuffling the
    # observations relative to fixed locales is equivalent to shuffling L.
    permutations = np.argsort(
        rng.random((n_permutations, N), dtype=np.float32),
        axis=1,
    )

    # For every shuffle, sum the reduced-model residuals within each locale.
    # These B x K values form g in the equation in the docstring.
    permuted_residuals = fit['residual_reduced'][permutations]
    locale_residual_sums = permuted_residuals @ locale_membership.T

    # Projecting locale membership onto the [1, T] basis measures the part of
    # the locale design that is already explained by the reduced model.
    projected_locales = np.stack([
        reduced_basis[permutations, column] @ locale_membership.T
        for column in range(reduced_basis.shape[1])
    ], axis=-1)

    # Subtract that projection from the raw locale cross-products. locale_gram
    # is G: the B x K x K residualized locale design for all permutations.
    locale_gram = (
        np.diag(locale_counts)[None]
        - projected_locales @ projected_locales.transpose(0, 2, 1)
    )

    # Apply g.T @ pinv(G) @ g to every permutation simultaneously. The
    # pseudoinverse also handles the redundant all-K-columns encoding of D.
    explained_null = np.einsum(
        'bi,bij,bj->b',
        locale_residual_sums,
        np.linalg.pinv(locale_gram),
        locale_residual_sums,
    )
    # Convert each permuted locale sum of squares into a sensitivity score using
    # the same denominator and square root as the observed sensitivity.
    return np.sqrt(
        np.maximum(explained_null, 0.0) / fit['RSS_reduced']
    )


def calculate_position_sensitivity(
    repliprob: np.ndarray,
    pseudotime: np.ndarray,
    locales: np.ndarray,
    cell_states: np.ndarray,
    repliprob_thresh: float = 0.9,
    n_permutations: int = 10_000,
    min_observations: int = 10,
    min_locales: int = 3,
    missing_locale: int = -1,
    random_seed: int = 42,
) -> dict:
    """Calculate position sensitivity and its significance for every locus.

    Replication probabilities are binarized as replicated when they are above
    repliprob_thresh and non-replicated when they are below
    1 - repliprob_thresh. Intermediate values are ignored. For each locus,
    locales with fewer than min_observations classified alleles are excluded,
    and the locus is analyzed only if at least min_locales remain.

    Args:
        repliprob (np.ndarray): RepTile ML replication probabilities, shape
            (ncells, nloci, ncopies).
        pseudotime (np.ndarray): Per-cell S-phase pseudo-time, shape (ncells,).
        locales (np.ndarray): Preprocessed locale assignments with the same
            shape as repliprob.
        cell_states (np.ndarray): Per-cell G1, S or G2 labels, shape (ncells,).
        repliprob_thresh (float): Probability threshold for calling a locus
            replicated. Its complement is the non-replicated threshold.
        n_permutations (int): Number of locale-label permutations per locus.
        min_observations (int): Minimum classified alleles required per locale.
        min_locales (int): Minimum eligible locales required per locus.
        missing_locale (int): Value identifying missing or excluded locales.
        random_seed (int): Seed shared across the genome-wide calculation.

    Returns:
        dict: Per-locus arrays named sensitivity, pvalue and qvalue. Loci that
        fail an inclusion gate contain NaN in all three arrays.
    """
    # Validate the array shapes and the replication-state threshold.
    if (
        repliprob.shape != locales.shape
        or repliprob.ndim != 3
    ):
        raise ValueError(
            'repliprob and locales must have equal '
            '(ncells, nloci, ncopies) shapes'
        )
    ncells, nloci, ncopies = repliprob.shape
    if len(pseudotime) != ncells or len(cell_states) != ncells:
        raise ValueError('pseudotime and cell_states must have one value per cell')
    if not 0.5 < repliprob_thresh < 1:
        raise ValueError('repliprob_thresh must be between 0.5 and 1')
    if n_permutations < 1:
        raise ValueError('n_permutations must be at least 1')

    # Restrict every input to S-phase cells while preserving cell/copy order.
    S_cells = np.asarray(cell_states).astype(str) == 'S'
    repliprob_S = repliprob[S_cells]
    L_S = locales[S_cells]
    T_S = pseudotime[S_cells]

    # Initialize the genome-wide outputs and the shared random-number generator.
    sensitivity = np.full(nloci, np.nan)
    pvalue = np.full(nloci, np.nan)
    rng = np.random.default_rng(random_seed)

    for i in range(nloci):
        # Flatten the S-cell and chromosome-copy axes for locus i.
        repliprob_i = repliprob_S[:, i, :].ravel()
        L = L_S[:, i, :].ravel()
        T = np.repeat(T_S, ncopies)

        # Convert confident replication probabilities to binary states. Values
        # between the two thresholds remain NaN and are therefore ignored.
        replistate = np.full(repliprob_i.shape, np.nan)
        replistate[repliprob_i > repliprob_thresh] = 1
        replistate[repliprob_i < 1 - repliprob_thresh] = 0

        # Keep only alleles with both a replication state and a nuclear locale.
        valid = ~np.isnan(replistate) & (L != missing_locale)
        y = replistate[valid]
        T = T[valid]
        L = L[valid]

        # Remove under-sampled locales, then require enough remaining locales.
        locale_ids, locale_counts = np.unique(L, return_counts=True)
        eligible_locales = locale_ids[locale_counts >= min_observations]
        if len(eligible_locales) < min_locales:
            continue
        keep = np.isin(L, eligible_locales)
        y = y[keep]
        T = T[keep]
        L = L[keep]

        # Skip loci with no replication-state variation.
        if y.mean() <= 0.0 or y.mean() >= 1.0:
            continue

        # Skip saturated fits. The full model spends one parameter on the
        # intercept, one on pseudo-time and K - 1 on locale, so it needs
        # N - K - 1 > 0 residual degrees of freedom for the variance
        # decomposition to mean anything. Without this the full model can
        # interpolate the data, driving RSS_full to zero and sensitivity to 1.
        if len(y) - len(eligible_locales) - 1 <= 0:
            continue

        # Fit the nested models and exclude degenerate reduced-model fits.
        fit = _fit_nested_models(y, T, L)
        if fit['RSS_reduced'] <= 0:
            continue

        # Calculate the observed continuous position-sensitivity score.
        sensitivity[i] = _calculate_sensitivity(fit)

        # Build the permutation null and calculate the upper-tail p-value.
        sensitivity_null = _permutation_null(
            fit,
            n_permutations,
            rng,
        )
        pvalue[i] = (
            np.sum(sensitivity_null >= sensitivity[i]) + 1
        ) / (n_permutations + 1)

    # Correct finite permutation p-values across all analyzed loci using the
    # Benjamini-Hochberg implementation provided by SciPy.
    qvalue = np.full(nloci, np.nan)
    finite = np.isfinite(pvalue)
    qvalue[finite] = false_discovery_control(
        pvalue[finite],
        method='bh',
    )

    return {
        'sensitivity': sensitivity,
        'pvalue': pvalue,
        'qvalue': qvalue,
    }
