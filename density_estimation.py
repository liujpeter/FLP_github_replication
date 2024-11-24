# main file

# import packages
import time
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize
from matplotlib import pyplot as plt

#import personalized functions
import my_functions as my


# construct vector
quant_vec = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65,0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 0.975])

# construct selector matrix
quant_sel = np.zeros([1,22])
quant_sel[0,6]  = 1 # 0.25
quant_sel[0,11] = 1 # 0.5
quant_sel[0,16] = 1 # 0.75



# number of knots
K_vec = np.sum(quant_sel,axis=1)

# K dimension
K_vec_n   = len(K_vec)
K_vec     = K_vec + np.ones(K_vec_n)


# specify grid that is used to evaluate p(x)
xmin = 0
xmax = 3
#xmax = 4
xn   = 301
xgrid = np.linspace(xmin, xmax, xn)


# import data
cexp_data = pd.read_csv('nondurserv_detrended.csv').to_numpy()
cexp_data = cexp_data[:,1:3]

earnings_t = cexp_data[:,0]
cons_pc = cexp_data[:,1]

Tend = len(set(earnings_t))
earnings_detrended = cons_pc[cons_pc>0] # positive consumption

# construct knots
knots_all = np.quantile(earnings_detrended, quant_vec)
MDD_GoF_sum  = np.zeros([K_vec_n, 3])


# k specification
ii = 0 
K             = int(K_vec[ii])
knots         = knots_all[quant_sel[ii,:]==1]


PhatDensValue = np.zeros([Tend, len(xgrid)])
PhatDensCoef  = np.zeros([Tend, K])


PhatDensNorm  = np.zeros([Tend, 1])
PhatlogLike   = np.zeros([Tend, 1])
Vinv_all      = np.zeros([K*Tend, K])
N_all         = np.zeros([Tend, 1])
Period_all    = np.zeros([Tend, 1])
N_details     = np.zeros([Tend, 4+K])

# time index
timeidx = 1990+1/4

# t specification

#Tend = 1
for tt in range(Tend):
    
    start = time.time()
    

    #tt = 0
    
    Period_all[tt] = timeidx
    
    # time t data
    selecteddraws_t = earnings_detrended[earnings_t==timeidx]
    #    timeidx = timeidx + 1/12 # monthly frequency
    timeidx = timeidx + 1/4 # quarterly frequency
    N_all[tt]       = len(selecteddraws_t)
    
    # count observations with knot restriction
    # recall that there are K-1 knots
    N_knots      = np.zeros([1,K])
    N_knots[0,0] = sum(selecteddraws_t<=knots[0])
    for kk in range(1,K-1):
        count = 0 
        for l in range(len(selecteddraws_t)):
            if (knots[kk-1] < selecteddraws_t[l] <= knots[kk]):
                count = count+1
        N_knots[0,kk] = count
    
    
    N_knots[0,K-1] = sum(selecteddraws_t>knots[-1])
    print("Number of obs in knot brackets: " + str(N_knots))
    print("Max knot: " + str(knots[K-2]))
    
    
    #compute MLE estimator
    
    # initial guess
    if tt == 0:
        alpha_initial = np.zeros(K)
    else:
        alpha_initial = PhatDensCoef[tt-1,:]
    
    
    results_t = minimize(lambda x: my.logspline_obj(selecteddraws_t, x, knots, xmin, xmax), alpha_initial, method =  'nelder-mead')
    
    coef_t = results_t.x
    
    PhatlogLike[tt] = - N_all[tt]*results_t.fun
    pi_hat = 0
    print("no top coding")
    
    #print(coef_t)
    #results
    PhatDensCoef[tt,:]  = coef_t
    
    #computation time
    end = time.time()
    time_length = end - start
    print("time for period " + str(tt) + " is " + str(time_length))
    

    #compute inverse Hessian
    Hess_t = my.hessian_loglh(PhatDensCoef[tt,:], knots, min(xgrid), max(xgrid))

    Vinv_t = - Hess_t
    
    test = my.is_pos_def(Vinv_t)






# convert array into dataframe 
DF_PhatDensCoef = pd.DataFrame(PhatDensCoef) 
#DF_Vinv_all = pdf.

# save the dataframe as a csv file 
DF_PhatDensCoef.to_csv("PhatDensCoef.csv")
#


    
# test code to plot densities
# x_values = np.linspace(0,3,100)
# y_values = np.zeros(100)
# for i in range(100):    
#     y_values[i] = my.pdfEval_noNorm(x_values[i], PhatDensCoef[106,:],knots)




# plt.figure()
# plt.plot(x_values, y_values, color='red')
# plt.show()

# plt.figure()
# plt.hist(selecteddraws_t)
# plt.show()





