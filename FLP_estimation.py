#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:26:13 2024

@author: peterliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:18:03 2024

@author: peterliu
"""

# import packages
import time
import numpy as np
import pandas as pd
import random
from scipy import integrate
from scipy.optimize import minimize
from matplotlib import pyplot as plt

#import personalized functions
import load_data as my_d
import my_functions as my

#specify random stuff
#knots, manually written for now
knots = [0.591021, 0.871043, 1.22027]

# specify grid that is used to evaluate p(x)
xmin = 0
xmax = 4
#xmax = 4
xn   = 301
xgrid = np.linspace(xmin, xmax, xn)
lb = min(xgrid)
ub = max(xgrid)


#load aggregate data
unrate, unrate_period = my_d.loadaggdata()

#load density data
PhatDensCoef = my_d.loaddensdata()

#drop first period of unrate data
unrate = unrate[1:]
unrate_period = unrate_period[1:]

#run OLS regression

X = np.hstack([np.ones([len(unrate_period),1]), PhatDensCoef ] )
y = unrate

IR = np.zeros(5)
horizon = np.zeros(5)

for h in range(5):
    
    horizon[h] = h
    lX = X[:len(X)-h,:]
    fy = y[h:]
    

    # OLS Regression using matrix methods
    # X is the design matrix, y is the response vector
    X_transpose = lX.T
    X_transpose_X = X_transpose @ lX
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    X_transpose_y = X_transpose @ fy
    beta = X_transpose_X_inv @ X_transpose_y
    
    phi = beta[1:]
    intercept = beta[0]

    basis_prod_int = np.zeros([4,4])

    for i in range(4):
        for j in range(4):
            
            Inte = integrate.quad(lambda x: my.basis_logspline(x, knots)[i]*my.basis_logspline(x, knots)[j], lb,ub)
            basis_prod_int[i,j] = Inte[0]


    basis_prod_int_inv = np.linalg.inv(basis_prod_int)

    Bh =  phi.reshape(-1,1).T @ basis_prod_int_inv



    counterfac = np.zeros([4,4])


    for i in range(4):
        for j in range(4):
            
            #.5 for 75% 
            Inte = integrate.quad(lambda x: my.basis_logspline(x+.1, knots)[i]*my.basis_logspline(x+.1, knots)[j], lb,ub)
            counterfac[i,j] = Inte[0]



    phi_star =  Bh @ counterfac


    IR[h] = (phi_star - phi).reshape(-1,1).T @  PhatDensCoef[0,:].T


plt.figure()
plt.plot(horizon, IR*100, color='red')
plt.xticks(horizon)
plt.title("FLP-OLS: impulse response of unemployment (in pp) to 3000 dollar stimulus check ")
plt.show()

# time to run the bootstrap procedure
#bootstrap number
B = 50 
X = np.hstack([np.ones([len(unrate_period),1]), PhatDensCoef ] )
y = unrate
N = len(X)


bootstrap_trials_IR = [None] * B

for b in range(B):

    indices = np.random.choice(N, size=N, replace=True)

    bootstrap_X = X[indices]
    bootstrap_y = y[indices]
    
    bootstrap_IR = np.zeros(5)
    horizon = np.zeros(5)

    for h in range(5):
        
        horizon[h] = h
        lX = bootstrap_X[:len(bootstrap_X)-h,:]
        fy = bootstrap_y[h:]
        

        # OLS Regression using matrix methods
        # X is the design matrix, y is the response vector
        X_transpose = lX.T
        X_transpose_X = X_transpose @ lX
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_y = X_transpose @ fy
        beta = X_transpose_X_inv @ X_transpose_y
        
        phi = beta[1:]
        intercept = beta[0]

        basis_prod_int = np.zeros([4,4])

        for i in range(4):
            for j in range(4):
                
                Inte = integrate.quad(lambda x: my.basis_logspline(x, knots)[i]*my.basis_logspline(x, knots)[j], lb,ub)
                basis_prod_int[i,j] = Inte[0]


        basis_prod_int_inv = np.linalg.inv(basis_prod_int)

        Bh =  phi.reshape(-1,1).T @ basis_prod_int_inv



        counterfac = np.zeros([4,4])


        for i in range(4):
            for j in range(4):
                
                #.5 for 75% 
                Inte = integrate.quad(lambda x: my.basis_logspline(x+.1, knots)[i]*my.basis_logspline(x+.1, knots)[j], lb,ub)
                counterfac[i,j] = Inte[0]



        phi_star =  Bh @ counterfac


        bootstrap_IR[h] = (phi_star - phi).reshape(-1,1).T @  PhatDensCoef[0,:].T
    
        #enter bootstrap estimates
    bootstrap_trials_IR[b] = bootstrap_IR

upper_band = np.zeros(5)
lower_band = np.zeros(5)


IR = IR*100

for h in range(5):

    first_elements = [vector[h]*100 for vector in bootstrap_trials_IR]
    
    # plt.figure()
    # plt.hist(first_elements, bins=10, color='skyblue', edgecolor='black')
    # # Adding labels and title
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title(str(h))
    # plt.show()

    variance = np.var(first_elements, ddof=1)
    
    upper_band[h] = IR[h] + 1.96*variance 
    lower_band[h] = IR[h] - 1.96*variance 
    
# Plotting
plt.figure()

# Plot impulse response
plt.plot(horizon, IR, label='Impulse Response', color='blue')

# Plot confidence bands
plt.fill_between(horizon, lower_band, upper_band, color='blue', alpha=0.2, label='95% Confidence Interval')

# Add titles and labels
plt.xticks(horizon)
plt.title("FLP-OLS: IR of unemployment (in pp) to $3000 stimulus check ")
plt.xlabel('Horizon')
plt.legend()
plt.grid(True)

plt.savefig('figures/flp_ols.png') 

# Show plot
plt.show()

    
    
    
    
 




