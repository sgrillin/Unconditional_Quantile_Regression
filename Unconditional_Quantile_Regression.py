import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import statsmodels.api as sm


# Step 1: Estimate quantiles of the outcome variable
def estimate_quantiles(y, tau):
    """Estimates the quantile tau of y"""
    return np.percentile(y, tau * 100)


# Step 2: Kernel density estimation for f(q_tau)
def kernel_density_estimate(y, q_tau):
    """Estimate the density at q_tau using Gaussian kernel density estimator"""
    kde = gaussian_kde(y)
    return kde.evaluate(q_tau)[0]


# Step 3: Compute RIF for the given quantile
def rif_quantile(y, tau):
    """Compute the RIF for the quantile tau"""
    q_tau = estimate_quantiles(y, tau)
    f_q_tau = kernel_density_estimate(y, q_tau)

    # Indicator function for y <= q_tau
    indicator = (y <= q_tau).astype(int)

    # Compute the RIF as per the formula
    rif = q_tau + (tau - indicator) / f_q_tau
    return rif


# Step 4: Fit the RIF regression model
def fit_rif_regression(y, X, tau):
    """Fit an OLS regression model for the RIF of a given quantile tau"""
    rif_y = rif_quantile(y, tau)
    model = LinearRegression()
    model.fit(X, rif_y)
    return model
