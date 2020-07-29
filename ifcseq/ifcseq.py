#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:53:41 2020

@author: nikos
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

def normalize_markers_01(X, copy=False):
    '''
    

    ParaXeters
    ----------
    X : 2D numpy array
        Surface markers for the single-cell RNA-seq experiment
        Rows: cells, columns, marker values
    copy : bool, optional
        DESCRIPTION. The default is False.
        If True, creates a copy of X instead of normalizing in-place

    Returns
    -------
    A column-wise [0,1] normalized version of X

    '''
    if copy==True:
        X = X.copy()
    
    for c in np.arange(X.shape[1]):#iterate over columns
        X[:,c][X[:,c]==-np.inf]=0
        X[:,c] = X[:,c]-np.nanmin(X[:,c])
        X[:,c] = X[:,c]/np.nanmax(X[:,c])
        X[np.isnan(X)] = 0#pad nans with 0
    return(X)

def fit_RF(X_seq, Y_seq, savePath = './model.pkl', n_estimators = 50, criterion='mae', max_features = 'sqrt', n_jobs=-1):
    '''
    Trains a random forest regression model to predict gene expression in Y_seq
    based on the surface marker values available in X_seq
    
    Parameters
    ----------
    X_seq : 2D numpy array
        Surface markers for the single-cell RNA-seq experiment
        Rows: cells, columns, marker values
        It is recommended to normalize each column to [0,1] beforehand
    Y_seq : 2D numpy array
        Count matrix for the single-cell RNA-seq experiment
        Rows: cells, columns, marker values
    savePath : str
        Path to save the fitted model to disk
    The rest of the arguments are specific to sklearn.ensemble.RandomForestRegressor
    
    Returns
    -------
    A fitted RandomForestRegressor model

    '''
    print('setting up the model')
    model = RandomForestRegressor(n_estimators=n_estimators,
                               criterion=criterion,
                               max_features=max_features,
                               random_state=0,
                               n_jobs=n_jobs)
    
    print('training the model')
    X_tr = X_seq
    Y_tr = Y_seq
    model.fit(X_tr,Y_tr)
    
    #save the model to disk
    if savePath != None:
        print('saving the trained model to disk')
        joblib.dump(model,savePath,compress=True)
    
    return model

def fit_LR(X_seq, Y_seq, savePath = './model.pkl', n_estimators = 50, criterion='mae', max_features = 'sqrt', n_jobs=-1):
    '''
    Trains a linear regression model to predict gene expression in Y_seq
    based on the surface marker values available in X_seq
    
    Parameters
    ----------
    X_seq : 2D numpy array
        Surface markers for the single-cell RNA-seq experiment
        Rows: cells, columns, marker values
        It is recommended to normalize each column to [0,1] beforehand
    Y_seq : 2D numpy array
        Count matrix for the single-cell RNA-seq experiment
        Rows: cells, columns, marker values
    savePath : str
        Path to save the fitted model to disk

    Returns
    -------
    A fitted LinearRegression model

    '''
    print('setting up the model')
    model = LinearRegression()
    
    print('training the model')
    X_tr = X_seq
    Y_tr = Y_seq
    model.fit(X_tr,Y_tr)
    
    #save the model to disk
    if savePath != None:
        print('saving the trained model to disk')
        joblib.dump(model,savePath,compress=True)
    
    return model

def predict(X, model):
    '''

    Parameters
    ----------
    X : 2D numpy array
        Surface markers used to predict gene expression
        Rows: cells, columns, marker values
        It is recommended to normalize each column to [0,1] beforehand
    model : str, sklearn.ensemble.RandomForestRegressor or sklearn.linear_model.LinearRegression
        Trained model used for predictions. If str, it will load a model from disk
        found at the path denoted by the str. Otherwise it will use the
        provided RandomForestRegressor or LinearRegression model

    Returns
    -------
    Y: 2D Numpy array of predicted gene expression

    '''
    
    #load a model from disk
    if type(model) == str:
        print('loading model from disk, at:',model)
        model = joblib.load(model)
    
    #predict gene expression
    print('predicting gene expression')
    Y = model.predict(X)
    return(Y)
           
















           
