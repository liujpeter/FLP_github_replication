#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:55:52 2024

@author: peterliu
"""

# import packages
import time
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import os
import datetime


def loadaggdata():
    
    #info shock paper data

    unrate_load = pd.read_csv('UNRATE_fred_quarterly.csv').to_numpy()
    unrate = unrate_load[:,1]/100
    unrate_period = unrate_load[:,0]
    
    
    return unrate, unrate_period


def loaddensdata():
    
    PhatDensCoef = pd.read_csv('PhatDensCoef.csv').to_numpy()
    PhatDensCoef = PhatDensCoef[:,1:]
    
    return PhatDensCoef


def loadinstrdata():
    
    #load dataframe
    df = pd.read_excel('pre-and-post-ZLB-factors-extended.xlsx')

    # drop first column
    df = df.iloc[1:,1:3].reset_index(drop=True)

    #rename column
    df.rename(columns={'Unnamed: 1': 'date'}, inplace=True)

    #change dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    df = df[df['date'] <= datetime.datetime(2016, 9, 30)]


    df.set_index('date', inplace=True)

    #df['year'] = df['date'].dt.year
    #df['month'] = df['date'].dt.month

    quarterly_df = df.resample('Q').mean()
    

    
    mp_shock = quarterly_df.to_numpy()
    
    
    return mp_shock

    