#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:27:02 2024

@author: peterliu
"""

#change working directory
#import os
#os.chdir("/Users/peterliu/Desktop/FLP_github_replication")

#cwd = os.getcwd()

#import packages
import time
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import datetime

#import personalized functions
import load_data as my_d
import my_functions as my


#load aggregate data
unrate, unrate_period = my_d.loadaggdata()
unrate = unrate[1:]
unrate_period = unrate_period[1:]


#load density data
PhatDensCoef = my_d.loaddensdata()

#load instrument data
mp_shock = my_d.loadinstrdata()


#drop get data standardized
unrate = unrate[6:]
unrate_period = unrate_period[6:]
PhatDensCoef = PhatDensCoef[6:]

#specify matrices for 2SLS regression

X = np.hstack([np.ones([len(unrate_period),1]), PhatDensCoef ] )
y = unrate

# construct the instrument matrix Z
quant_vec = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65,0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 0.975])

# construct selector matrix
quant_sel = np.zeros([1,22])
quant_sel[0,6]  = 1 # 0.25
quant_sel[0,11] = 1 # 0.5
quant_sel[0,16] = 1 # 0.75
quant_sel[0,3] = 1 # .1 
quant_sel[0,19] = 1 # .9

# number of knots
K_vec = np.sum(quant_sel,axis=1)

# K dimension
K_vec_n   = len(K_vec)
K_vec     = K_vec + np.ones(K_vec_n)

#construct the knots
knots_all = np.quantile(mp_shock, quant_vec)
knots         = knots_all[quant_sel[0,:]==1]

#generate basis functions
Z =  my.basis_logspline_data(mp_shock, knots)

# #construct the estimator

# Z_transpose = Z.T
# Z_transpose_Z = Z_transpose @ Z

# #rank_check = np.linalg.matrix_rank(Z_transpose_Z)
# #cond_check =  np.linalg.cond(Z_transpose_Z)

# Z_transpose_Z_inv = np.linalg.inv(Z_transpose_Z)
# X_transpose = X.T

# beta = np.linalg.inv(X_transpose @ Z @ Z_transpose_Z_inv @ Z_transpose @ X ) @ X_transpose @ Z @ Z_transpose_Z_inv @ Z_transpose @ y

#specify density knots, manually written for now
density_knots = [0.591021, 0.871043, 1.22027]

# specify grid that is used to evaluate p(x)
xmin = 0
xmax = 4
#xmax = 4
xn   = 301
xgrid = np.linspace(xmin, xmax, xn)
lb = min(xgrid)
ub = max(xgrid)


IR = np.zeros(5)
horizon = np.zeros(5)

for h in range(5):
    
    horizon[h] = h
    lX = X[:len(X)-h,:]
    lZ = Z[:len(X)-h,:]
    fy = y[h:]
    

    # OLS Regression using matrix methods
    # X is the design matrix, y is the response vector    
    Z_transpose = lZ.T
    Z_transpose_Z = Z_transpose @ lZ

    #rank_check = np.linalg.matrix_rank(Z_transpose_Z)
    #cond_check =  np.linalg.cond(Z_transpose_Z)

    Z_transpose_Z_inv = np.linalg.inv(Z_transpose_Z)
    X_transpose = lX.T

    beta = np.linalg.inv(X_transpose @ lZ @ Z_transpose_Z_inv @ Z_transpose @ lX ) @ X_transpose @ lZ @ Z_transpose_Z_inv @ Z_transpose @ fy

    

    phi = beta[1:]
    intercept = beta[0]

    basis_prod_int = np.zeros([4,4])

    for i in range(4):
        for j in range(4):
            
            Inte = integrate.quad(lambda x: my.basis_logspline(x, density_knots)[i]*my.basis_logspline(x, density_knots)[j], lb,ub)
            basis_prod_int[i,j] = Inte[0]


    basis_prod_int_inv = np.linalg.inv(basis_prod_int)

    Bh =  phi.reshape(-1,1).T @ basis_prod_int_inv



    counterfac = np.zeros([4,4])


    for i in range(4):
        for j in range(4):
            
            #.5 for 75% increase
            Inte = integrate.quad(lambda x: my.basis_logspline(x + .1, density_knots)[i]*my.basis_logspline(x + .1, density_knots)[j], lb,ub)
            counterfac[i,j] = Inte[0]



    phi_star =  Bh @ counterfac


    IR[h] = (phi_star - phi).reshape(-1,1).T @  PhatDensCoef[0,:].T


plt.figure()
plt.plot(horizon, IR*100, color='red')
plt.xticks(horizon)
plt.title("FLP-IV: IR of unemployment (in pp) to approx $3000 stimulus check ")

plt.savefig('figures/flp_iv.png') 

plt.show()





