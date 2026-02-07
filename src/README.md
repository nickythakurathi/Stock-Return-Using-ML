# Stock Return Predictability with Machine Learning (Quarterly, 2000–2024)

Author: Nicky Thakurathi

## Summary

This project tests whether firm-level information can predict next-quarter stock returns using quarterly U.S. stock data from 2000–2024.

After finding that simple linear regression performs poorly out-of-sample, I extend the analysis using machine learning methods commonly used in empirical finance research. Models are evaluated using a time-based train/test split to avoid look-ahead bias, and performance is compared across three feature sets: accounting-only, market-only, and combined.

## Motivation

In earlier work, basic linear regression models produced negative out-of-sample R², meaning they performed worse than predicting the historical mean return. This project investigates whether regularization and nonlinear ML methods improve predictive performance, and whether market variables add incremental value beyond accounting fundamentals.

## Data

The analysis uses quarterly firm-level accounting ratios and market variables for U.S. stocks.

Raw datasets are not included due to CRSP/Compustat licensing restrictions.

### Key Variables

- permno: firm identifier  
- qdate: quarter end date  
- ret_q_next: next-quarter return (target)

## Model Design

To keep the experiment clean, accounting and market information are separated first, then combined later.

### Model A (Accounting Only)

- roe, roa, npm, gpm, GProf  
- de_ratio, debt_capital, cash_debt  
- at_turn, inv_turn, invt_act, rect_act  
- divyield  

### Model M (Market Only)

- VOL, SHROUT, ret_q, sprtrn, DLRET  

### Model B (Combined)

- Model A + Model M predictors  

The purpose of Model B is to test whether market variables subsume accounting information or whether fundamentals add incremental predictive content.

## Missing Data Handling

Financial panel data contains substantial missingness across firms and quarters. Dropping all missing observations would heavily reduce the sample and bias it toward large, stable firms.

Predictors are imputed using quarter-by-quarter cross-sectional medians. Observations with missing ret_q_next are dropped.

This produces aligned, ML-ready datasets with the same firm-quarter universe across models.

## Methods

### Models

- Ridge regression  
- Lasso regression  
- Elastic Net  
- Random Forest  

### Evaluation Setup

- Training period: 2000–2016  
- Test period: 2017–2024  
- Metrics: Train/Test R², Train/Test MSE  
- Overfitting check: Train R² minus Test R²  

## Results (High Level)

Out-of-sample performance remains weak across all models, consistent with the finance literature that short-horizon returns are difficult to predict.

Regularized linear models (especially Lasso) tend to perform best among the tested methods, while Random Forest shows clear overfitting (higher train fit but worse test performance).

Feature importance plots are generated for:

- Lasso: absolute standardized coefficients  
- Random Forest: permutation importance on the test set  

## Outputs

Saved artifacts include:

- model_comparison_table.csv  
- Feature importance plots (.png) for each dataset (A, M, B)  

## Notes for Interpretation

Negative out-of-sample R² is common in return prediction. This project emphasizes a correct research design: time-based evaluation, consistent sample alignment across models, overfitting diagnostics, and transparent handling of missing data.

## Data Restrictions

This repository does not include raw CRSP/Compustat datasets due to licensing. The code is designed to run locally once the datasets are available.
