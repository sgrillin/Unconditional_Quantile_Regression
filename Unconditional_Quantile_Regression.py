import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.api as sm

# Function to estimate quantiles
def estimate_quantiles(y, tau):
    """Estimates the quantile tau of y"""
    return np.percentile(y, tau * 100)

# Kernel Density Estimator function
def kernel_density_estimate(y, q_tau):
    """Estimate the density at q_tau using Gaussian kernel density estimator"""
    kde = gaussian_kde(y)
    return kde.evaluate(q_tau)[0]

# RIF function for the given quantile
def rif_quantile(y, tau):
    """Compute the RIF for the quantile tau"""
    q_tau = estimate_quantiles(y, tau)
    f_q_tau = kernel_density_estimate(y, q_tau)

    # Indicator function for y <= q_tau
    indicator = (y <= q_tau).astype(int)

    # Compute the RIF as per the formula
    rif = q_tau + (tau - indicator) / f_q_tau
    return rif


def fit_rif_regression(y, X, tau, error_type='none', cluster_groups=None, n_bootstraps=1000):
    """
    Fit an OLS regression model for the RIF of a given quantile tau using statsmodels.

    Parameters:
        y: Outcome variable (array-like)
        X: Covariates (DataFrame or array-like)
        tau: Quantile (float)
        error_type: Type of standard error estimation ('none', 'cluster', 'bootstrap')
        cluster_groups: Grouping variable for cluster-robust standard errors (if error_type='cluster')
        n_bootstraps: Number of bootstrap samples (if error_type='bootstrap')

    Returns:
        model: The fitted OLS model
        se: Estimated standard errors (either cluster-robust, bootstrapped, or regular)
    """
    # Compute RIF-transformed outcome
    rif_y = rif_quantile(y, tau)

    # Add a constant term to the model (intercept)
    X_const = sm.add_constant(X)

    # Fit OLS model using statsmodels
    ols_model = sm.OLS(rif_y, X_const).fit()

    # Handle standard error estimation based on the specified error type
    if error_type == 'cluster':
        if cluster_groups is None:
            raise ValueError("cluster_groups must be provided for cluster-robust standard errors")
        # Cluster-robust standard errors
        robust_model = ols_model.get_robustcov_results(cov_type='cluster', groups=cluster_groups)
        return ols_model, robust_model.bse

    elif error_type == 'bootstrap':
        # Bootstrap standard errors
        bootstrap_coefs = []
        n = len(y)

        for i in range(n_bootstraps):
            # Resample data with replacement
            bootstrap_indices = np.random.choice(range(n), size=n, replace=True)

            # Resample the outcome (y) and covariates (X)
            if isinstance(X, pd.DataFrame):
                X_bootstrap = X.iloc[bootstrap_indices, :]
            else:
                X_bootstrap = X[bootstrap_indices]

            y_bootstrap = y[bootstrap_indices]

            # Fit OLS model on the bootstrap sample
            bootstrap_model = sm.OLS(rif_quantile(y_bootstrap, tau), sm.add_constant(X_bootstrap)).fit()
            bootstrap_coefs.append(bootstrap_model.params)

        # Convert the list of bootstrapped coefficients to a NumPy array
        bootstrap_coefs = np.array(bootstrap_coefs)

        # Standard errors are the standard deviation of the bootstrap estimates
        bootstrap_se = np.std(bootstrap_coefs, axis=0)
        return ols_model, bootstrap_se

    else:
        # Default OLS model with regular standard errors
        return ols_model, ols_model.bse
