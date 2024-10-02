import pandas as pd
import seaborn as sns
from Unconditional_Quantile_Regression import fit_rif_regression

# Load the tips dataset
tips_data = sns.load_dataset('tips')

# Define the outcome variable (y) and the covariates (X)
y = tips_data['tip'].values  # Outcome variable (tip amount)
x = pd.get_dummies(tips_data[['total_bill', 'day']], drop_first=True)  # Covariates: total bill and day of the week

# Fit the RIF regression for the 50th percentile (median) with cluster-robust standard errors
tau = 0.5
median_rifreg = fit_rif_regression(y, x, tau, error_type='huber-white')
median_rifreg_boostrap, bootstrap_se = fit_rif_regression(y, x, tau, error_type='bootstrap', n_bootstraps=1000)

coefficients_bootstrap = median_rifreg_boostrap.params
results_median_bootstrap = pd.DataFrame({
    'Coefficient': coefficients_bootstrap,
    'Bootstrap SE': bootstrap_se})
print(results_median_bootstrap)

# Define deciles (0.1, 0.2, ..., 0.9)
deciles = [i / 10 for i in range(1, 10)]

# Create a list to store the results as dictionaries
results_list = []

# Loop through each decile and run RIF regression with bootstrapped SE
for tau in deciles:
    print(f"\nFitting RIF regression for the {int(tau * 100)}th percentile")

    # Fit the RIF regression for bootstrapped standard errors
    rifreg_bootstrap, bootstrap_se = fit_rif_regression(y, X, tau, error_type='bootstrap', n_bootstraps=1000)
    coefficients_bootstrap = rifreg_bootstrap.params

    # Store results for bootstrapped SE in the results list
    for i, coef in enumerate(coefficients_bootstrap):
        results_list.append({
            'Quantile': f'{int(tau * 100)}th percentile',
            'Coefficient': coef,
            'Bootstrap SE': bootstrap_se[i]
        })

# Convert the list of results to a DataFrame using pd.concat
results_df = pd.concat([pd.DataFrame([row]) for row in results_list], ignore_index=True)

# Display the full results table
print("\nAll Results (Coefficients and Bootstrapped Standard Errors by Quantile):")
print(results_df)

# Add plot and predict distribution