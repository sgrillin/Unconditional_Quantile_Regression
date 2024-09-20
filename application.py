import seaborn as sns
from Unconditional_Quantile_Regression import fit_rif_regression

# Load the tips dataset
tips_data = sns.load_dataset('tips')

# Define the outcome variable (y) and the covariates (X)
y = tips_data['tip'].values  # Outcome variable (tip amount)
X = pd.get_dummies(tips_data[['total_bill', 'day']], drop_first=True)  # Covariates: total bill and day of the week

# Fit the RIF regression for the 50th percentile (median) with cluster-robust standard errors
tau = 0.5
ols_model, cluster_robust_se = fit_rif_regression(y, X, tau, error_type='cluster', cluster_groups=tips_data['day'])

# Print the coefficients and cluster-robust standard errors
print("\nCluster-Robust Standard Errors:")
for coef, se in zip(ols_model.params, cluster_robust_se):
    print(f"Coefficient: {coef}, Standard Error: {se}")

# Fit the RIF regression for the 50th percentile (median) with bootstrapped standard errors
ols_model, bootstrap_se = fit_rif_regression(y, X, tau, error_type='bootstrap', n_bootstraps=1000)

# Print the coefficients and bootstrapped standard errors
print("\nBootstrapped Standard Errors:")
for coef, se in zip(ols_model.params, bootstrap_se):
    print(f"Coefficient: {coef}, Standard Error: {se}")
