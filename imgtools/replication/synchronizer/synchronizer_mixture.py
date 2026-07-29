"""Doubling-constrained mixture model for cell-cycle synchronization.

G1 and G2 spot counts follow N(mu, sigma) and N(2mu, sigma), while S follows
U(mu, 2mu) convolved with N(0, sigma). The default model shares sigma across
phases; ``sigma_mode='stagewise'`` fits one width per phase.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer


def component_densities(x, mu, sigma1, sigmaS, sigma2):
    """Evaluate the G1, S and G2 probability densities at ``x``."""
    d1 = norm.pdf(x, mu, sigma1)
    dS = (norm.cdf((x - mu) / sigmaS) -
          norm.cdf((x - 2.0 * mu) / sigmaS)) / mu
    d2 = norm.pdf(x, 2.0 * mu, sigma2)
    return d1, dS, d2


def unpack_params(p, sigma_mode):
    """Transform unconstrained optimizer values into model parameters."""
    # Scaling mu and sigma improves numerical conditioning.
    mu = p[0] * 1e4
    if sigma_mode == 'shared':
        sigma1 = sigmaS = sigma2 = np.exp(p[1]) * 1e3
        logits = (p[2], p[3])
    else:
        sigma1 = np.exp(p[1]) * 1e3
        sigmaS = np.exp(p[2]) * 1e3
        sigma2 = np.exp(p[3]) * 1e3
        logits = (p[4], p[5])

    # Softmax of [logit_G1, logit_S, 0], stabilized before exponentiation.
    ww = np.exp(np.array([logits[0], logits[1], 0.0]) -
                max(logits[0], logits[1], 0.0))
    w = ww / ww.sum()
    return mu, sigma1, sigmaS, sigma2, w


def negloglik(p, x, sigma_mode):
    """Return the negative log-likelihood of the phase mixture."""
    mu, s1, sS, s2, w = unpack_params(p, sigma_mode)
    if mu <= 0 or min(s1, sS, s2) <= 0:
        return 1e18
    d1, dS, d2 = component_densities(x, mu, s1, sS, s2)
    mix = w[0] * d1 + w[1] * dS + w[2] * d2
    return -np.sum(np.log(np.clip(mix, 1e-300, None)))


class CellCycleMixtureModel(CellCycleSynchronizer):
    """Assign cells to G1, S and G2 from their total spot counts.

    The model is fitted by multi-restart Nelder-Mead maximum likelihood. Its
    constraints are enforced by a log link for sigma and a softmax for phase
    fractions. The number of S cells is fixed to ``round(N * pi_S)``; cells are
    ranked by posterior S probability, and the remainder are assigned to G1 or
    G2 by their larger posterior.

    Optional config keys are ``sigma_mode`` (``'shared'`` or ``'stagewise'``),
    ``n_restarts`` (default 60) and ``random_seed`` (default 0).
    """

    requires_rt = False
    PHASES = ['G1', 'S', 'G2']

    def __init__(
        self,
        scf: SingleCellFeature,
        config: dict,
        initial_states: np.array = None,
    ) -> None:
        """Initialize the synchronizer and validate mixture-specific options."""
        super().__init__(scf, config, initial_states)
        self.check_config()

    def check_config(self) -> None:
        """Read and validate mixture-specific configuration."""
        sigma_mode = self.config.get('sigma_mode', 'shared')
        if sigma_mode not in ('shared', 'stagewise'):
            raise ValueError(
                f"sigma_mode must be 'shared' or 'stagewise', got '{sigma_mode}'."
            )
        self.sigma_mode = sigma_mode

        n_restarts = self.config.get('n_restarts', 60)
        if not isinstance(n_restarts, int) or n_restarts <= 0:
            raise ValueError("n_restarts must be a positive integer.")
        self.n_restarts = n_restarts
        self.random_seed = self.config.get('random_seed', 0)

    def get_total_spot(self) -> np.array:
        """Return total spot count per cell over the configured chromosomes."""
        return np.nansum(self.matrix, axis=1).astype(float)

    def fit(self, x: np.array):
        """Fit the model with reproducible multi-restart Nelder-Mead."""
        rng = np.random.default_rng(self.random_seed)

        if self.sigma_mode == 'shared':
            base = np.array([3.9, np.log(7.0), 0.0, 0.3])
            maxiter = 20000
        else:
            base = np.array([
                3.9, np.log(6.0), np.log(6.0), np.log(6.0), 0.0, 0.3
            ])
            maxiter = 40000
        ndim = len(base)

        best = None
        for k in range(self.n_restarts):
            p0 = base + (0 if k == 0 else rng.normal(0, 0.4, ndim))
            r = minimize(
                negloglik,
                p0,
                args=(x, self.sigma_mode),
                method='Nelder-Mead',
                options=dict(maxiter=maxiter, xatol=1e-7, fatol=1e-7),
            )
            if best is None or r.fun < best.fun:
                best = r
        return best

    def run(self) -> None:
        """Fit the model, assign phases and store fitted results."""
        x = self.get_total_spot()
        N = len(x)

        best = self.fit(x)
        mu, s1, sS, s2, w = unpack_params(best.x, self.sigma_mode)

        d1, dS, d2 = component_densities(x, mu, s1, sS, s2)
        post = np.vstack([w[0] * d1, w[1] * dS, w[2] * d2]).T
        post /= post.sum(1, keepdims=True)

        # Adjusted Classify-and-Count fixes the S count to the fitted fraction.
        nS = int(round(N * w[1]))
        order_S = np.argsort(post[:, 1])[::-1]
        is_S = np.zeros(N, dtype=bool)
        is_S[order_S[:nS]] = True
        states = np.where(post[:, 2] >= post[:, 0], 'G2', 'G1').astype('U20')
        states[is_S] = 'S'

        k_params = 4 if self.sigma_mode == 'shared' else 6
        nll = best.fun

        self.states_ = states
        self.x_ = x
        self.mu_ = mu
        self.sigma_ = (
            s1 if self.sigma_mode == 'shared' else np.array([s1, sS, s2])
        )
        self.weights_ = w
        self.posterior_ = post
        self.nS_ = nS
        self.nll_ = nll
        self.aic_ = 2 * k_params + 2 * nll
        self.bic_ = k_params * np.log(N) + 2 * nll
