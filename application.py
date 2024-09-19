from Unconditional_Quantile_Regression import *
import seaborn as sns
import pandas as pd

# Load the tips dataset
tips_data = sns.load_dataset('tips')

# Define the outcome variable (y) and the covariates (X)
y = tips_data['tip']  # Outcome variable (tip amount)
X = pd.get_dummies(tips_data[['total_bill', 'day']], drop_first=True)  # Covariates: total bill and day of the week

# Fit the RIF regression for the 25th percentile
tau = 0.25
rif_model = fit_rif_regression(y, X, tau)

# Print the regression coefficients
print("RIF Regression Coefficients:", rif_model.coef_)

# Define the deciles (0.1, 0.2, ..., 0.9)
deciles = [i / 10 for i in range(1, 10)]

# Loop through each decile and run RIF regression
for tau in deciles:
    print(f"\nFitting RIF regression for the {int(tau * 100)}th percentile")

    # Fit the RIF regression for the given decile (tau)
    rif_model = fit_rif_regression(y, X, tau)

    # Print the regression coefficients
    print(f"RIF Regression Coefficients for {int(tau * 100)}th percentile:", rif_model.coef_)