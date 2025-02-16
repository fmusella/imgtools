import numpy as np
from functools import partial
from scipy.optimize import minimize


def GMM_solve(
    stat: dict,
    p = None, eps = None, beta = None,
    p_err = np.nan, eps_err = np.nan, beta_err = np.nan
):
    """ Implements the solutions of the Generalized Method of Moments (GMM)
    for the statistical model underlying the SimulatedRepliSeq class.
    
    Depending on the input parameters, it uses different equations.
    
    The shape of the output will match the input one.

    Args:
        stat: the summary statistics dictionary, with keys:
            - 'n': average number of spots,
            - 'n_var': variance of the number of spots,
            - 'f': fraction of zeroes.
            - 'f_var': variance of the fraction of zeroes.
        p: replication probability,
        eps: detection efficiency,
        beta: overcounting rate,
        p_err: error on p,
        eps_err: error on eps,
        beta_err: error on beta.

    Returns:
        Depending on the input parameters, it returns:
            - eps, beta, eps_err, beta_err if p is provided,
            - p, beta, p_err, beta_err if eps is provided,
            - p, eps, p_err, eps_err if beta is provided,
            - p, p_err if eps and beta are provided.
    """
    
    # Get the data from the stat dictionary
    n = stat['n']
    n_var = stat['n_var']
    f = stat['f']
    f_var = stat['f_var']
    nf_cov = stat['nf_cov']
    
    # 1) 2) AND 3) KNOWN P (G1, G2 or S) - GET: EPS, BETA
    if p is not None and eps is None and beta is None:
        
        # 1) G1
        if p == 'G1':
            eps, beta, eps_err, beta_err = G1_get_eps_beta(n, n_var, f, f_var, nf_cov)
        
        # 2) G2
        elif p == 'G2':
            eps, beta, eps_err, beta_err = G2_get_eps_beta(n, n_var, f, f_var, nf_cov)
        
        # 3) S
        else:
            eps, beta, eps_err, beta_err = know_p_get_eps_beta(n, n_var, f, f_var, nf_cov, p, p_err**2)
        
        return eps, beta, eps_err, beta_err

    # 4) KNOWN: EPS, GET: P, BETA
    elif eps is not None and p is None and beta is None:
        p, beta, p_err, beta_err = know_eps_get_p_beta(n, n_var, f, f_var, nf_cov, eps, eps_err**2)
        return p, beta, p_err, beta_err
    
    # 5) KNOWN: BETA, GET: P, EPS
    elif beta is not None and p is None and eps is None:
        p, eps, p_err, eps_err = know_beta_get_p_eps(n, n_var, f, f_var, nf_cov, beta, beta_err**2)
    
    # 6) KNOWN: EPS, BETA, GET: P
    elif eps is not None and beta is not None and p is None:
        p, p_err = know_eps_beta_get_p(n, n_var, f, f_var, nf_cov, eps, beta, eps_err**2, beta_err**2)



# 1) G1 - GET: EPS, BETA
def G1_get_eps_beta(n, n_var, f, f_var, nf_cov):
    
    # Get eps
    eps = 1 - f
    
    # Get beta
    beta = n / eps
    
    # Get error on eps
    eps_var = f_var
    eps_err = eps_var ** 0.5
    
    # Get error on beta
    db_dn = 1 / eps
    db_df = n / eps ** 2
    beta_var = db_dn ** 2 * n_var + db_df ** 2 * f_var + 2 * db_dn * db_df * nf_cov
    beta_err = beta_var ** 0.5
    
    return eps, beta, eps_err, beta_err

# 2) G2 - GET: EPS, BETA
def G2_get_eps_beta(n, n_var, f, f_var, nf_cov):
    
    # Get eps
    eps = 1 - f ** 0.5
    
    # Get beta
    beta = n / (2 * eps)
    
    # Get error on eps
    eps_var = f_var / (4 * f)
    eps_err = eps_var ** 0.5
    
    # Get error on beta
    db_dn = 1 / (2 * eps)
    db_df = n / (4 * f ** 0.5 * eps ** 2)
    beta_var = db_dn ** 2 * n_var + db_df ** 2 * f_var + 2 * db_dn * db_df * nf_cov
    beta_err = beta_var ** 0.5
    
    return eps, beta, eps_err, beta_err

# 3) KNOW: P - GET: EPS, BETA
def know_p_get_eps_beta(n, n_var, f, f_var, nf_cov, p, p_var):
        
    # Auxiliary variable
    Y = (1 + p) ** 2 - 4 * p * (1 - f)
    
    # Get eps
    eps = (1 + p - Y ** 0.5) / (2 * p)
    
    # Get beta
    beta = n / ((1 + p) * eps)
    
    # Get error on eps
    de_df = - 1 / Y ** 0.5
    de_dp = ((2 * f - 1) * p + 1 - Y ** 0.5) / (2 * p ** 2 * Y ** 0.5)
    eps_var = de_df ** 2 * f_var + de_dp ** 2 * p ** 2
    eps_err = eps_var ** 0.5
    
    # Get error on beta
    db_dn = 1 / ((1 + p) * eps)
    db_dp = - n / (eps * (1 + p) ** 2)
    db_df = n / ((1 + p) * eps ** 2 * Y ** 0.5)
    beta_var = db_dn ** 2 * n_var + db_df ** 2 * f_var + db_dp ** 2 * p_var + 2 * db_dn * db_df * nf_cov
    beta_err = beta_var ** 0.5
    
    return eps, beta, eps_err, beta_err

# 4) KNOW: EPS - GET: P, BETA
def know_eps_get_p_beta(n, n_var, f, f_var, nf_cov, eps, eps_var):
    
    # Get p
    p = (1 - eps - f) / (eps * (1 - eps))
    
    # Get beta
    beta = n / ((1 + p) * eps)
    
    # Get error on p
    dp_de = - ((1 - eps) ** 2 - f * (1 - 2 * eps)) / (eps ** 2 * (1 - eps) ** 2)
    dp_df = - 1 / (eps * (1 - eps))
    p_var = dp_de ** 2 * eps_var + dp_df ** 2 * f_var
    p_err = p_var ** 0.5
    
    # Get error on beta
    db_dn = 1 / ((1 + p) * eps)
    db_de = - n / ((1 + p) * eps ** 2)
    db_df = n / (eps ** 2 * (1 - eps) * (1 + p) ** 2)
    beta_var = db_dn ** 2 * n_var + db_df ** 2 * f_var + db_de ** 2 * eps_var + 2 * db_dn * db_df * nf_cov
    beta_err = beta_var ** 0.5
    
    return p, beta, p_err, beta_err

# 5) KNOW: BETA - GET P, EPS
def know_beta_get_p_eps(n, n_var, f, f_var, nf_cov, beta, beta_var):
    
    # Auxiliary variables
    D = n / beta
    X = 1 - 4 * (f + D - 1) / D ** 2
    
    # Get eps
    eps = (D / 2) * (1 + X ** 0.5)
    # We can see from equations that X becomes negative for G2 cells.
    # So for these cases we use the G2 formula
    eps = np.where(np.isnan(eps), 1 - f ** 0.5, eps)
    
    # Get p
    p = n / (eps * beta) - 1
    
    # Get error on eps
    de_dD = 0.5 + (D - 2) / (2 * D * X ** 0.5)
    de_dn = (1 / beta) * de_dD
    de_df = - 1 / (D * X ** 0.5)
    de_db = - (n / beta ** 2) * de_dD
    eps_var = de_dn ** 2 * n_var + de_df ** 2 * f_var + de_db ** 2 * beta_var + 2 * de_dn * de_df * nf_cov
    eps_err = eps_var ** 0.5
    
    # Get error on p
    dp_dn = 1 / (eps * beta)
    dp_df = 1 / (eps ** 2 * X ** 0.5)
    dp_db = - n / (eps * beta ** 2)
    p_var = dp_dn ** 2 * n_var + dp_df ** 2 * f_var + dp_db ** 2 * beta_var + 2 * dp_dn * dp_df * nf_cov
    p_err = p_var ** 0.5
    
    return p, eps, p_err, eps_err

# 6) KNOW: EPS, BETA - GET: P
def know_eps_beta_get_p(n, n_var, f, f_var, nf_cov, eps, beta, eps_var, beta_var):
    
    # Get p in the first way
    p1 = (1 - eps - f) / (eps * (1 - eps))
    
    # Get p in the second way
    p2 = n / (eps * beta) - 1
    
    # Get p as the average of the two
    p = (p1 + p2) / 2
    
    # Get error on p1 (same as in know_eps_get_p_beta)
    dp1_de = - ((1 - eps) ** 2 - f * (1 - 2 * eps)) / (eps ** 2 * (1 - eps) ** 2)
    dp1_df = - 1 / (eps * (1 - eps))
    p1_var = dp1_de ** 2 * eps_var + dp1_df ** 2 * f_var
    p1_err = p1_var ** 0.5
    
    # Get error on p2
    # We assume it's the same as the error on p1
    p2_var = p1_var
    p2_err = p2_var ** 0.5
    
    # Get error on p
    # We assume cov(p1, p2) = cov(p1, p1) = var(p1).
    # So at the end we simply have var(p) = var(p1)
    p_var = p1_var
    p_err = p_var ** 0.5
    
    # If the types are float, we use a numerical method to refine p
    # We assume that the error doesn't change
    if isinstance(n, float):
        def root_func(x: float, p1: float, p2: float) -> float:
            return np.sqrt((x - p1) ** 2 + (x - p2) ** 2)
        f = partial(root_func, p1=p1, p2=p2)
        p = minimize(f, (p1 + p2) / 2).x[0]
    
    return p, p_err
