import seaborn as sns
from Unconditional_Quantile_Regression import fit_rif_regression

# Load the tips dataset
tips_data = sns.load_dataset('tips')

# Define the outcome variable (y) and the covariates (X)
y = tips_data['tip'].values  # Outcome variable (tip amount)
X = pd.get_dummies(tips_data[['total_bill', 'day']], drop_first=True)  # Covariates: total bill and day of the week

# Fit the RIF regression for the 50th percentile (median) with cluster-robust standard errors
tau = 0.5
median_rifreg = fit_rif_regression(y, X, tau)
median_rifreg_cluster, cluster_robust_se = fit_rif_regression(y, X, tau, error_type='cluster', cluster_groups=tips_data['day'])
median_rifreg_boostrap, bootstrap_se = fit_rif_regression(y, X, tau, error_type='bootstrap', n_bootstraps=1000)

# Get the coefficients and cluster-robust results from the model
coefficients_cluster = median_rifreg_cluster.params
results_median_cluster = pd.DataFrame({
    'Coefficient': coefficients_cluster,
    'Cluster-Robust SE': cluster_robust_se})
print(results_median_cluster)

coefficients_bootstrap = median_rifreg_boostrap.params
results_median_bootstrap = pd.DataFrame({
    'Coefficient': coefficients_bootstrap,
    'Bootstrap SE': bootstrap_se})
print(results_median_bootstrap)
