# function file

# import packages
import numpy as np
import pandas as pd
from scipy import integrate
from matplotlib import pyplot as plt


# functions

#check positive definiteness
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# basis function
def basis_logspline(x, knots):
    #functions
    UpperBound = 3
    K      = len(knots)+1
    basis_fcn = np.zeros([K])
    
    basis_fcn[K-1] = max(UpperBound-x,0)
    for i in reversed(range(K-1)):
            basis_fcn[i] = (max((knots[i]-x) , 0))**3 

    return basis_fcn

def basis_logspline_data(x, knots):
    
    UpperBound = 3
    x_dim  = len(x)
    K      = len(knots)+1
    basis_fcn = np.zeros([x_dim,K])
    
    for j in range(x_dim):
        basis_fcn[j,K-1] = max(UpperBound-x[j],0)
        for i in reversed(range(K-1)):
            basis_fcn[j,i] = (max((knots[i]-x[j]) , 0))**3 

    return basis_fcn


def pdfEval_noNorm(x, coef,knots):
    # this procedure evaluates the unnormalized exp(the log spline density)
    # x is n*1
    # knots is (K-1)*1
    # basis is n*(K+1)
    # coef is (k+1)*1
    pdf_logspline = np.exp(basis_logspline(x,knots) @ coef)
    return pdf_logspline
        



def logspline_obj(data, coef, knots, lb, ub):
    IntPdf = integrate.quad(lambda x: pdfEval_noNorm(x, coef,knots), lb,ub)
    out = np.mean(basis_logspline_data(data,knots),axis=0) @ coef - np.log(IntPdf[0])
    return -out


#def pdfEval(x,coef,knots,lnnorm):


def hessian_loglh(coef,knots,lb,ub):
    
    K = len(knots)+1
    IntPdf = integrate.quad(lambda x: pdfEval_noNorm(x, coef,knots), lb,ub)
    hess_l_jk = np.zeros([K,K])
    
    for j in range(K):
        for k in range(j,K):
    
            Int_j = integrate.quad(lambda x: basis_logspline(x,knots)[j]*pdfEval_noNorm(x, coef,knots), lb,ub)
            Int_j_norm = Int_j[0]/IntPdf[0]
            
            Int_k = integrate.quad(lambda x: basis_logspline(x,knots)[k]*pdfEval_noNorm(x, coef,knots), lb,ub)
            Int_k_norm = Int_k[0]/IntPdf[0]
            
            Int_jk = integrate.quad(lambda x:  (basis_logspline(x,knots)[j] - Int_j_norm)*(basis_logspline(x,knots)[k] - Int_k_norm)*pdfEval_noNorm(x, coef,knots), lb,ub)
            hess_l_jk[j,k] = Int_jk[0]/IntPdf[0] 
            
    hess_sym = hess_l_jk

    return hess_sym