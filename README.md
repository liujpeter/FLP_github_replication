# Functional Local Projections Replication Codes
 
Author: Peter Liu

Date: 11/24/2024

This folder contains replication codes for implementing the functional local projections (FLP) estimator. 

The primary application of this estimator is to understand the casual effect of microeconomic heterogeneity on the macroeconomy. FLP is a two-step method. The first step uses nonparametric density estimation to summarize distributional changes in the economy with a finite set of basis coefficients. The second step is to use quasi-experimental econometric methods such as instrumental variables (IV) to understand the causal effect of distributional changes in the economy. 

For more details on the methodology, see the FLP.pdf file in this folder. Note that this project is a work in progress and I am in the process of writing more detailed documentation and cleaning up the code.  

Here is a description of the relevant files to run the FLP estimator. 

1) density_estimation.py estimates the basis coefficients that describe the economic distribution of interest. The coefficients are stored in PhatDensCoef.csv.

2) FLP_estimation.py implements the FLP estimator with the estimated coefficients from density_estimation.py. 

3) FLP_IV_estimation.py implements the FLP estimator in 2), but now using Romer-Romer monetary shocks as an instrument to control for endogeneity.
