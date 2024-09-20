# Unconditional Quantile Regression (UQR)

## Project Overview

This project implements the methodology proposed by **Sergio Firpo, Nicole M. Fortin, and Thomas Lemieux** in their 2009 paper titled *"Unconditional Quantile Regressions"* (Econometrica, Vol. 77, No. 3). The goal of this project is to provide a Python-based implementation of **Recentered Influence Function (RIF)** regression, which allows researchers to estimate the effect of covariates on the unconditional quantiles of an outcome variable.

### Paper Summary

The paper proposes a new method to evaluate the impact of changes in explanatory variables on quantiles of the unconditional distribution of an outcome variable. The core idea is to compute the **Recentered Influence Function (RIF)** for the quantile of interest and then regress it on the covariates.

The RIF for a quantile $\tau$ is given by:

$$\text{RIF}(y, q_{\tau}) = q_\tau + \frac{\tau - \mathbb{1} \{y \leq q_{\tau}\}}{f_Y(q_\tau)}$$

Where:
- $q_\tau$ is the $\tau^{th}$ quantile of the outcome variable $Y$,
- $\mathbb{1}\{y \leq q_\tau\}$ is an indicator function for whether $y$ is less than or equal to the quantile $q_\tau$,
- $f_Y(q_\tau)$ is the density of the outcome variable $Y$ at the quantile $q_\tau$.

This approach is called **Unconditional Quantile Regression (UQR)** because it directly targets changes in the marginal (unconditional) quantiles of \( Y \), unlike traditional quantile regressions, which model conditional quantiles.

---

## Project Structure

- **`Unconditional_Quantile_Regression.py`**: The core Python script that contains the functions to implement the RIF regression method.
- **`application.py`**: A script that provides an example of how to use the UQR implementation with a real dataset.
- **`README.md`**: This file, providing an overview of the project and instructions for use.

---

## Requirements

The project relies on the following Python packages:

- `numpy`: For array and matrix operations.
- `pandas`: For data manipulation and handling datasets.
- `scipy`: For kernel density estimation.
- `statsmodels`: For advanced statistical modeling and kernel density estimation.
- `sklearn`: For running the linear regression model.

---

## Applications in the literature

If you want to see an example of UQR, check out my 2020 co-authored paper: _"Unconditional quantile regression analysis of UK inbound tourist expenditures"_ with Sharma, Abhijit and Woodward, Richard (Economics Letters)
Available at:

https://bradscholars.brad.ac.uk/bitstream/handle/10454/17530/elmanuscriptR2.pdf?sequence=2&isAllowed=n
