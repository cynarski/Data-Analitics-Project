# Data Analitics Project - Bayesian models for predicting the number of fatalities in terrorist attacks using GTD data.

## Overview

This project uses Bayesian statistical modeling techniques to predict the number of fatalities in terrorist attacks, utilizing the Global Terrorism Database (GTD). The work includes Bayesian regression, data preprocessing, and extensive visualizations for model evaluation and understanding.

### Authors

Cynarski Michał & Barszczak Bartłomiej, AGH WEAIiIB AIR-ISZ

## Directory Structure

- `data/`: Contains the GTD dataset (`globalterrorismdb_0718dist.csv`)
- `model1/` & `model2/`: Stan model files for the first and second Bayesian model approaches

## Features

- **Data Preprocessing:** Loading and cleaning GTD data
- **Modeling Techniques:** Bayesian statistical methods using Stan and CmdStanPy
- **Visualization:** Rich plotting of scatter histograms, parameter posteriors, and Bayesian network structures
- **Validation:** Posterior predictive checks, parameter density comparisons, and Bayesian network plotting

## Model Details

- **Model 1:** Models based on `country`, `weapon type`, `target type` and `nperps` predictors. Combines logarithm link function with Poisson distribution.
- **Model 2:** Models based on `country`, `weapon type`, `target type` and `nperps` predictors. Combines logarithm link function with Negative Binomial distribution.

## Getting Started

1. Install required libraries:
   ```bash
   pip install cmdstanpy pandas arviz numpy matplotlib seaborn scikit-learn networkx
   ```
2. Ensure GTD data is available in `data/globalterrorismdb_0718dist.csv` (can be downloaded from https://www.kaggle.com/datasets/START-UMD/gtd).
3. Run the notebook `Project_TerrorismAttacks_BC.ipynb` for data exploration, model fitting, and result visualization.

## Results & Insights

Results highlight how Bayesian modeling captures the statistical relationships between attack characteristics and fatalities, providing valuable insights for risk assessment and policy-making.

For more information or to collaborate, contact the authors listed above.