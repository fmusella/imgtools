import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer


# ---------------------------------------------------------------------------
# Model densities and log-likelihood.
# These are defined OUTSIDE the class (like simulate_rt in synchronizer.py) so
# they are plain picklable functions that scipy.optimize.minimize can call, and
# so the model is easy to read/test in isolation.
#
# Doubling-constrained mixture on the per-cell total spot count X = M + E, where
# M is the copy-number contribution and E is the cell-to-cell variability (<E>=0):
#     G1 ~ N(mu,   sigma)
#     S  ~ U(mu, 2mu) * N(0, sigma)  (convolution -> "smoothed boxcar")
#     G2 ~ N(2mu,  sigma)
# The three component means are tied to a single location mu by the DNA-doubling
# constraint (G2 = 2*G1), and the S plateau spans exactly [mu, 2mu]. The closed
# form of the S density is
#     fS(x) = (1/mu) * [ Phi((x-mu)/sigmaS) - Phi((x-2mu)/sigmaS) ],
# with Phi the standard-normal CDF.
# ---------------------------------------------------------------------------


def component_densities(x, mu, sigma1, sigmaS, sigma2):
    """ Evaluate the three per-phase densities fG1, fS, fG2 at x.

    The means are pinned to the doubling constraint (mu for G1, 2*mu for G2),
    while the widths are passed explicitly so the same routine serves both the
    shared-sigma model (sigma1 == sigmaS == sigma2) and the stage-specific model.

    Args:
        x (np.array): per-cell total spot counts, shape (ncell,).
        mu (float): G1 (2C) mean; the G2 (4C) mean is 2*mu.
        sigma1 (float): standard deviation of the G1 Gaussian.
        sigmaS (float): standard deviation of the Gaussian smoothing the S plateau.
        sigma2 (float): standard deviation of the G2 Gaussian.

    Returns:
        tuple (np.array, np.array, np.array): the G1, S and G2 densities at x.
    """
    d1 = norm.pdf(x, mu, sigma1)
    dS = (norm.cdf((x - mu) / sigmaS) - norm.cdf((x - 2.0 * mu) / sigmaS)) / mu
    d2 = norm.pdf(x, 2.0 * mu, sigma2)
    return d1, dS, d2


def unpack_params(p, sigma_mode):
    """ Map the unconstrained optimizer vector p to the model parameters.

    We optimize in an unconstrained reparameterization so the constraints
    (sigma > 0, pi_k >= 0, sum_k pi_k = 1) are satisfied by construction:
        mu     = p[0] * 1e4                (the 1e4 factor is numerical conditioning)
        sigma  = exp(p[i]) * 1e3           (log link -> sigma > 0)
        weights = softmax([p[j], p[k], 0]) (softmax with the third logit pinned to
                                            0 to remove the shift-invariance; leaves
                                            two degrees of freedom for three weights)
    In 'shared' mode a single sigma is used for all three phases (4 free params:
    mu, sigma, and two weight logits). In 'stagewise' mode each phase has its own
    sigma (6 free params: mu, sigma1, sigmaS, sigma2, and two weight logits).

    Args:
        p (np.array): unconstrained parameter vector (length 4 or 6).
        sigma_mode (str): 'shared' or 'stagewise'.

    Returns:
        tuple (mu, sigma1, sigmaS, sigma2, w): floats and the weight array w (3,).
    """
    mu = p[0] * 1e4
    if sigma_mode == 'shared':
        sigma1 = sigmaS = sigma2 = np.exp(p[1]) * 1e3
        logits = (p[2], p[3])
    else:  # 'stagewise'
        sigma1 = np.exp(p[1]) * 1e3
        sigmaS = np.exp(p[2]) * 1e3
        sigma2 = np.exp(p[3]) * 1e3
        logits = (p[4], p[5])
    # Softmax over [logit_G1, logit_S, 0], stabilized by subtracting the max
    ww = np.exp(np.array([logits[0], logits[1], 0.0]) - max(logits[0], logits[1], 0.0))
    w = ww / ww.sum()
    return mu, sigma1, sigmaS, sigma2, w


def negloglik(p, x, sigma_mode):
    """ Negative log-likelihood of the mixture for the count vector x.

    L = sum_c log( piG1 fG1(xc) + piS fS(xc) + piG2 fG2(xc) ), and we return -L.

    Args:
        p (np.array): unconstrained parameter vector.
        x (np.array): per-cell total spot counts.
        sigma_mode (str): 'shared' or 'stagewise'.

    Returns:
        float: the negative log-likelihood (a large sentinel if parameters are invalid).
    """
    mu, s1, sS, s2, w = unpack_params(p, sigma_mode)
    if mu <= 0 or min(s1, sS, s2) <= 0:
        return 1e18
    d1, dS, d2 = component_densities(x, mu, s1, sS, s2)
    mix = w[0] * d1 + w[1] * dS + w[2] * d2
    return -np.sum(np.log(np.clip(mix, 1e-300, None)))


class CellCycleMixtureModel(CellCycleSynchronizer):
    """ Assign cells to G1, S and G2 by a doubling-constrained mixture-model MLE.

    The synchronization uses only the per-cell total spot count (a proxy for DNA
    content): xc = sum over all loci and copies of the feature matrix. It does NOT
    use the replication-timing signal, so 'rt_file' is optional (requires_rt = False).

    We model the per-cell count as X = M + E, the sum of a copy-number contribution
    M and a cell-to-cell variability term E with <E> = 0. Copy number is fixed in
    G1 (2C) and G2 (4C) and increases through S, giving the per-phase densities
        G1 ~ N(mu, sigma)
        S  ~ U(mu, 2mu) * N(0, sigma)  ->  fS(x) = (1/mu)[Phi((x-mu)/sigmaS) - Phi((x-2mu)/sigmaS)]
        G2 ~ N(2mu, sigma)
    with the means tied by the DNA-doubling constraint (G2 = 2*G1). By default a
    single sigma is shared across the cycle ('shared', 4 free parameters). Setting
    sigma_mode = 'stagewise' fits one sigma per phase (sigma1, sigmaS, sigma2;
    6 free parameters), which we use to check that the shared-sigma assumption is
    justified (the E14 data give sigmaG2/sigmaG1 ~ 1.03).

    Fitting minimizes -L with the Nelder-Mead algorithm in scipy.optimize.minimize,
    restarting from n_restarts random initializations and keeping the lowest -L.
    The constraints (sigma > 0, pi_k >= 0, sum pi_k = 1) are enforced by an
    unconstrained reparameterization (log link for sigma, softmax for the weights);
    see unpack_params.

    Cells are then assigned by the quantification rule (Adjusted Classify-and-Count):
    the S count is pinned to N*piS (the naive argmax under-counts S, because the tall
    narrow G1/G2 Gaussians win at the plateau edges). We rank cells by the posterior
    P(S|xc), call the top round(N*piS) of them S, and split the rest into G1/G2 by
    argmax(P(G1|xc), P(G2|xc)).

    Inherits from CellCycleSynchronizer.

    --- Attributes (inherit from CellCycleSynchronizer) ---
    scf (SingleCellFeature): SingleCellFeature object.
    index (Index): Index of the SingleCellFeature.
    config (dict): Configuration dictionary for the synchronization method.
    rt_file (str or None): Path to the replication timing file (optional, unused by this class).
    usechroms (list): List of chromosome strings to be used in the synchronization.
    feature (str): Name of the feature to be used in the synchronization.
    matrix (np.array): Matrix of the feature for the chromosomes specified in usechroms.
    states_ (np.array): Array of strings with the states of the cells, updated in the run method.

    --- Attributes (specific to CellCycleMixtureModel) ---
    sigma_mode (str): 'shared' (single sigma, 4 params) or 'stagewise' (3 sigmas, 6 params).
    n_restarts (int): number of random restarts of the optimizer.
    random_seed (int): seed of the random-number generator used for the restarts.
    x_ (np.array): per-cell total spot counts used for the fit, shape (ncell,).
    mu_ (float): fitted G1 mean (the G2 mean is 2*mu_).
    sigma_ (float or np.array): fitted sigma. A scalar in 'shared' mode, or the
                                array [sigma1, sigmaS, sigma2] in 'stagewise' mode.
    weights_ (np.array): fitted mixing weights (piG1, piS, piG2), the population fractions.
    posterior_ (np.array): posterior P(phase|xc), shape (ncell, 3), columns G1/S/G2.
    nS_ (int): number of cells assigned to S by the quantification rule (round(N*piS)).
    nll_ (float): negative log-likelihood at the optimum.
    aic_ (float): Akaike information criterion.
    bic_ (float): Bayesian information criterion.

    --- Methods (for users) ---
    run: Fit the mixture model by MLE and assign the cells to G1, S and G2.
    """

    # This synchronizer uses only the per-cell total spot count, not the RT signal,
    # so 'rt_file' is optional.
    requires_rt = False

    # Order of the phases in the posterior columns and the weight array.
    PHASES = ['G1', 'S', 'G2']

    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleMixtureModel object.
        Inherits from CellCycleSynchronizer.

        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary. Uses 'feature' and 'usechroms'
                (required by the base class), and the optional keys 'sigma_mode'
                ('shared' by default), 'n_restarts' (60 by default) and
                'random_seed' (0 by default).
            initial_states (np.array, optional): Initial states of the cells.
                            If None, the states are initialized randomly.
        """

        super().__init__(scf, config, initial_states)

        # Read and validate the parameters specific to the mixture model
        self.check_config()

    def check_config(self) -> None:
        """ Check the configuration dictionary for the mixture-model synchronizer.

        Reads (with defaults) the optional keys:
        - sigma_mode (str): 'shared' (default) or 'stagewise'.
        - n_restarts (int): number of random restarts (default 60).
        - random_seed (int): RNG seed for the restarts (default 0).

        Stores self.sigma_mode, self.n_restarts and self.random_seed.
        """

        # sigma_mode: 'shared' (single sigma) or 'stagewise' (one sigma per phase)
        sigma_mode = self.config.get('sigma_mode', 'shared')
        if sigma_mode not in ('shared', 'stagewise'):
            raise ValueError(f"sigma_mode must be 'shared' or 'stagewise', got '{sigma_mode}'.")
        self.sigma_mode = sigma_mode

        # Number of random restarts of the Nelder-Mead optimizer
        n_restarts = self.config.get('n_restarts', 60)
        if not isinstance(n_restarts, int) or n_restarts <= 0:
            raise ValueError("n_restarts must be a positive integer.")
        self.n_restarts = n_restarts

        # Seed of the random-number generator used to draw the restarts
        self.random_seed = self.config.get('random_seed', 0)

    def get_total_spot(self) -> np.array:
        """ Get the per-cell total spot count used as the model input.

        The base class builds self.matrix by summing the copy axis of the feature
        matrix and keeping the columns whose chromosome is in usechroms. Summing
        that matrix over the loci gives the per-cell total spot count xc.

        Note: to reproduce a total that spans the whole genome (all autosomes AND
        chrX), usechroms must list every chromosome present in the SCF; the '#'
        shortcut expands to the autosomes only and therefore excludes chrX.

        Returns:
            np.array, shape (ncell,): per-cell total spot count.
        """
        return np.nansum(self.matrix, axis=1).astype(float)

    def fit(self, x: np.array):
        """ Fit the mixture by maximum likelihood with multi-restart Nelder-Mead.

        Minimizes -L (negloglik) in the unconstrained reparameterization, from
        self.n_restarts random initializations, and keeps the lowest -L solution.
        The first restart starts from a fixed, sensible base point; the others
        perturb it with Gaussian noise drawn from a seeded RNG (reproducible).

        Args:
            x (np.array): per-cell total spot counts.

        Returns:
            scipy.optimize.OptimizeResult: the best (lowest -L) optimization result.
        """

        rng = np.random.default_rng(self.random_seed)

        # Base initialization and optimizer settings depend on the number of free
        # parameters (4 in 'shared' mode, 6 in 'stagewise' mode).
        if self.sigma_mode == 'shared':
            # mu ~ 39k, sigma ~ 7k, ~equal weights
            base = np.array([3.9, np.log(7.0), 0.0, 0.3])
            maxiter = 20000
        else:
            # each of the three sigmas initialized ~6k
            base = np.array([3.9, np.log(6.0), np.log(6.0), np.log(6.0), 0.0, 0.3])
            maxiter = 40000
        ndim = len(base)

        best = None
        for k in range(self.n_restarts):
            p0 = base + (0 if k == 0 else rng.normal(0, 0.4, ndim))
            r = minimize(negloglik, p0, args=(x, self.sigma_mode), method='Nelder-Mead',
                         options=dict(maxiter=maxiter, xatol=1e-7, fatol=1e-7))
            if best is None or r.fun < best.fun:
                best = r
        return best

    def run(self) -> None:
        """ Fit the mixture model by MLE and assign each cell to G1, S or G2.

        Steps:
        1. Build the per-cell total spot count xc from the feature matrix.
        2. Fit the doubling mixture by multi-restart Nelder-Mead (self.fit).
        3. Compute the posterior P(phase|xc) for every cell.
        4. Assign cells with the quantification rule: rank cells by P(S|xc), call
           the top round(N*piS) of them S, and split the rest into G1/G2 by
           argmax(P(G1|xc), P(G2|xc)) (ties go to G2).
        The fitted parameters and the posterior are stored on the object.
        """

        # 1. Per-cell total spot count
        x = self.get_total_spot()
        N = len(x)

        # 2. Maximum-likelihood fit
        best = self.fit(x)
        mu, s1, sS, s2, w = unpack_params(best.x, self.sigma_mode)

        # 3. Posterior probability of each phase for every cell
        d1, dS, d2 = component_densities(x, mu, s1, sS, s2)
        post = np.vstack([w[0] * d1, w[1] * dS, w[2] * d2]).T
        post /= post.sum(1, keepdims=True)

        # 4. Quantification rule (Adjusted Classify-and-Count).
        # Pin the S count to the MLE weight, then split the remainder by argmax.
        nS = int(round(N * w[1]))
        order_S = np.argsort(post[:, 1])[::-1]
        is_S = np.zeros(N, dtype=bool)
        is_S[order_S[:nS]] = True
        # Remaining cells: G2 if P(G2) >= P(G1), else G1
        states = np.where(post[:, 2] >= post[:, 0], 'G2', 'G1').astype('U20')
        states[is_S] = 'S'

        # Number of free parameters, for the information criteria
        k_params = 4 if self.sigma_mode == 'shared' else 6
        nll = best.fun

        # Store the results on the object
        self.states_ = states
        self.x_ = x
        self.mu_ = mu
        self.sigma_ = s1 if self.sigma_mode == 'shared' else np.array([s1, sS, s2])
        self.weights_ = w
        self.posterior_ = post
        self.nS_ = nS
        self.nll_ = nll
        self.aic_ = 2 * k_params + 2 * nll
        self.bic_ = k_params * np.log(N) + 2 * nll
