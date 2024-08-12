# Bayesian Vector Autoregressive Models in Stan

In this project, I demonstrate how to fit a hierarchical Bayesian autoregressive model in Stan using simulated data and successfully recovering our parameters. 
I apply these types of models to econometric time series data to analyze the relationships between GDP, investment, and consumption for Ireland. The Quarto notebook covers:
(1) Demonstrating the basic pattern on a simple VAR model using simulated data: We'll show how the model recovers the true data-generating parameters.
(2) Applying the model to macro-economic data: We'll compare the results to those achieved on the same data using statsmodels MLE fits.
(3) Estimating a hierarchical Bayesian VAR model over multiple countries.

