"""Doubling-constrained mixture model for cell-cycle synchronization.

G1 and G2 spot counts follow N(mu, sigma) and N(2mu, sigma), while S follows
U(mu, 2mu) convolved with N(0, sigma). The default model shares sigma across
phases; sigma_mode='stagewise' fits one width per phase.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer


def component_densities(
    total_spots: np.ndarray,
    mu: float,
    sigma_G1: float,
    sigma_S: float,
    sigma_G2: float,
) -> tuple:
    """Evaluate the G1, S and G2 probability densities.

    Args:
        total_spots (np.ndarray): Per-cell total spot counts.
        mu (float): Mean of the G1 component; the G2 mean is 2 * mu.
        sigma_G1 (float): Standard deviation of the G1 component.
        sigma_S (float): Gaussian smoothing width of the S component.
        sigma_G2 (float): Standard deviation of the G2 component.

    Returns:
        tuple: G1, S and G2 density arrays, in that order.
    """
    f_G1 = norm.pdf(total_spots, mu, sigma_G1)
    f_S = (
        norm.cdf((total_spots - mu) / sigma_S) -
        norm.cdf((total_spots - 2.0 * mu) / sigma_S)
    ) / mu
    f_G2 = norm.pdf(total_spots, 2.0 * mu, sigma_G2)
    return f_G1, f_S, f_G2


def reparametrization(params: np.ndarray, sigma_mode: str) -> tuple:
    """Convert unconstrained parameters into valid model parameters.

    params[0] stores mu in units of 10,000 spots. The following one
    (shared mode) or three (stagewise mode) parameters store
    log(sigma / 1,000); exponentiation therefore ensures positive sigmas.

    The final two parameters are the log-odds of G1 and S relative to G2:
    logit_G1 = log(pi_G1 / pi_G2) and
    logit_S = log(pi_S / pi_G2). We set the G2 logit to zero because G2 is
    the reference. Applying softmax to [logit_G1, logit_S, 0] converts
    these two unconstrained values into the three positive phase fractions
    pi_G1, pi_S and pi_G2, which sum to one.

    Args:
        params (np.ndarray): Unconstrained parameter vector.
        sigma_mode (str): Either 'shared' or 'stagewise'.

    Returns:
        tuple: mu, the three phase sigmas and pi_G1, pi_S, pi_G2.
    """
    # Scaling mu and sigma places optimizer coordinates near order one.
    mu = params[0] * 1e4
    if sigma_mode == 'shared':
        sigma_G1 = sigma_S = sigma_G2 = np.exp(params[1]) * 1e3
        logit_G1, logit_S = params[2], params[3]
    else:
        sigma_G1 = np.exp(params[1]) * 1e3
        sigma_S = np.exp(params[2]) * 1e3
        sigma_G2 = np.exp(params[3]) * 1e3
        logit_G1, logit_S = params[4], params[5]

    # Subtracting the largest logit prevents overflow without changing softmax.
    logits = np.array([logit_G1, logit_S, 0.0])
    unnormalized_pi = np.exp(logits - max(
        logit_G1, logit_S, 0.0
    ))
    pi_G1, pi_S, pi_G2 = unnormalized_pi / unnormalized_pi.sum()
    return mu, sigma_G1, sigma_S, sigma_G2, pi_G1, pi_S, pi_G2


def negloglik(
    params: np.ndarray,
    total_spots: np.ndarray,
    sigma_mode: str,
) -> float:
    """Return the negative log-likelihood of the phase mixture.

    Args:
        params (np.ndarray): Unconstrained parameter vector.
        total_spots (np.ndarray): Per-cell total spot counts.
        sigma_mode (str): Either 'shared' or 'stagewise'.

    Returns:
        float: Negative log-likelihood, or a large value for invalid parameters.
    """
    mu, sigma_G1, sigma_S, sigma_G2, pi_G1, pi_S, pi_G2 = (
        reparametrization(params, sigma_mode)
    )
    if mu <= 0 or min(sigma_G1, sigma_S, sigma_G2) <= 0:
        return 1e18
    f_G1, f_S, f_G2 = component_densities(
        total_spots, mu, sigma_G1, sigma_S, sigma_G2
    )
    mixture_density = (
        pi_G1 * f_G1 +
        pi_S * f_S +
        pi_G2 * f_G2
    )
    # The floor prevents numerical underflow from producing log(0) = -inf.
    return -np.sum(np.log(np.clip(mixture_density, 1e-300, None)))


class CellCycleMixtureModel(CellCycleSynchronizer):
    """Assign cells to G1, S and G2 from their total spot counts.

    The model is fitted by multi-restart Nelder-Mead maximum likelihood. Its
    constraints are enforced by a log link for sigma and a softmax for phase
    fractions. The number of S cells is fixed to round(N * pi_S); cells are
    ranked by posterior S probability, and the remainder are assigned to G1 or
    G2 by their larger posterior.

    Optional config keys are sigma_mode ('shared' or 'stagewise'),
    n_restarts (default 60) and random_seed (default 0).
    """

    requires_rt = False
    PHASES = ['G1', 'S', 'G2']

    def __init__(
        self,
        scf: SingleCellFeature,
        config: dict,
        initial_states: np.ndarray = None,
    ) -> None:
        """Initialize the synchronizer.

        Args:
            scf (SingleCellFeature): Input single-cell feature object.
            config (dict): Synchronizer configuration.
            initial_states (np.ndarray, optional): Initial cell-cycle labels.

        Returns:
            None.
        """
        super().__init__(scf, config, initial_states)
        self.check_config()

    def check_config(self) -> None:
        """Read and validate mixture-specific configuration.

        Returns:
            None.
        """
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

    def get_total_spot(self) -> np.ndarray:
        """Return total spot count over the configured chromosomes.

        Returns:
            np.ndarray: Per-cell total spot counts.
        """
        return np.nansum(self.matrix, axis=1).astype(float)

    def fit(self, total_spots: np.ndarray):
        """Fit the model with reproducible multi-restart Nelder-Mead.

        Args:
            total_spots (np.ndarray): Per-cell total spot counts.

        Returns:
            scipy.optimize.OptimizeResult: Fit with the lowest negative
            log-likelihood across restarts.
        """
        random_generator = np.random.default_rng(self.random_seed)

        if self.sigma_mode == 'shared':
            initial_params = np.array([3.9, np.log(7.0), 0.0, 0.3])
            maxiter = 20000
        else:
            initial_params = np.array([
                3.9, np.log(6.0), np.log(6.0), np.log(6.0), 0.0, 0.3
            ])
            maxiter = 40000
        n_params = len(initial_params)

        best_fit = None
        for restart_idx in range(self.n_restarts):
            start_params = initial_params + (
                0 if restart_idx == 0
                else random_generator.normal(0, 0.4, n_params)
            )
            fit_result = minimize(
                negloglik,
                start_params,
                args=(total_spots, self.sigma_mode),
                method='Nelder-Mead',
                options=dict(maxiter=maxiter, xatol=1e-7, fatol=1e-7),
            )
            if best_fit is None or fit_result.fun < best_fit.fun:
                best_fit = fit_result
        return best_fit

    def run(self) -> None:
        """Fit the model, assign phases and store fitted results.

        Returns:
            None.
        """
        total_spots = self.get_total_spot()
        n_cells = len(total_spots)

        best_fit = self.fit(total_spots)
        mu, sigma_G1, sigma_S, sigma_G2, pi_G1, pi_S, pi_G2 = (
            reparametrization(best_fit.x, self.sigma_mode)
        )

        f_G1, f_S, f_G2 = component_densities(
            total_spots, mu, sigma_G1, sigma_S, sigma_G2
        )
        posterior = np.vstack([
            pi_G1 * f_G1,
            pi_S * f_S,
            pi_G2 * f_G2,
        ]).T
        posterior /= posterior.sum(1, keepdims=True)

        # Adjusted Classify-and-Count fixes the S count to the fitted fraction.
        n_S_cells = int(round(n_cells * pi_S))
        s_probability_order = np.argsort(posterior[:, 1])[::-1]
        s_mask = np.zeros(n_cells, dtype=bool)
        s_mask[s_probability_order[:n_S_cells]] = True
        states = np.where(
            posterior[:, 2] >= posterior[:, 0], 'G2', 'G1'
        ).astype('U20')
        states[s_mask] = 'S'

        n_model_params = 4 if self.sigma_mode == 'shared' else 6
        negative_log_likelihood = best_fit.fun

        self.states_ = states
        self.x_ = total_spots
        self.mu_ = mu
        self.sigma_ = (
            sigma_G1 if self.sigma_mode == 'shared'
            else np.array([sigma_G1, sigma_S, sigma_G2])
        )
        self.weights_ = np.array([pi_G1, pi_S, pi_G2])
        self.posterior_ = posterior
        self.nS_ = n_S_cells
        self.nll_ = negative_log_likelihood
        self.aic_ = 2 * n_model_params + 2 * negative_log_likelihood
        self.bic_ = (
            n_model_params * np.log(n_cells) + 2 * negative_log_likelihood
        )
