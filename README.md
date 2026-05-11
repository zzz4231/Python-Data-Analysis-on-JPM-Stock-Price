# Python-Data-Analysis-on-JPM-Stock-Price
Download JP Morgan historical stock data from 2010 to 2025, covering descriptive statistics, return distributions, and basic risk metrics (VaR, normality tests)  

Merge it with Fama-French mulyi-factor model. Using 'statsmodels' OLS regression to build single and multifactor models  
Apply 'sklearn' + 'mlxtend' Sequential Feature Selection to compare models by adjusted R², AIC, and BIC  
Evaluate out-of-sample R² for each model, improving R² from 0.65 to 0.82  

Analyze JPM log-returns via ACF plots and Ljung-Box tests  
Tune ARMA models (selecting order by BIC)  
Extend the analysis to volatility modeling with ARCH/GARCH to capture heteroskedasticity in financial returns  

Compares prediction models: (1) ARMA, (2) FNN built in PyTorch with hyperparameter tuning (lookback, hidden size, layers), and (3) PyTorch LSTM with the same tuning pipeline. Models are evaluated by RMSE on a held-out test set  

Compares prediction models: (1) Random Forest with hyperparameter tuning (lookback, hidden size, layers), and (2) XGBoost with the same tuning pipeline. Models are evaluated by RMSE on a held-out test set  
