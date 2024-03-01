#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:12:38 2023

@author: mxguevarra
"""

import sklearn as sk
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
import matplotlib.pyplot as plt
import pandas as pd

#%%

def sigmoid(u):
    
    sig = 1 / (1 + np.exp(-u))
    
    return sig
    

def cost_function(y, u):
    
    cost = np.mean( -y * np.log(u) - (1 - y) * np.log(1 - u) )
    
    return cost


def eval_L(beta, X, y):
    
    X_hat = np.insert(X, 0, 1, axis = 1)
    
    L = cost_function(y, sigmoid(X_hat@beta))
    
    return L


def grad_L(beta, X, y):
    
    X_hat = np.insert(X, 0, 1, axis = 1)

    gL = np.mean( (sigmoid(X_hat@beta) - y) * X_hat.transpose() )
    
    return gL


def grad_L_S(beta, X, y):
    
    X_hat = np.insert(X, 0, 1, axis = 0)

    gL = (sigmoid(X_hat@beta) - y) * X_hat.transpose()
    
    return gL


def train_model_grad_desc(X, y, alpha, max_iter):
    
    N, d = X.shape
    beta = np.zeros(d + 1)
    L_vals = []
    
    for i in range(max_iter):
        beta = beta - (alpha * grad_L(beta, X, y))
        L = eval_L(beta, X, y)
        L_vals.append(L)
    
    return beta, L_vals


def stoch_grad_desc(X, y, alpha):
    
    N, d = X.shape
    beta = np.zeros(d + 1)
    L_vals = []
    N_epochs = 20
    
    
    for ep in range(N_epochs):
        L = 0
        shuff_ind = np.random.permutation(N)
        
        for i in shuff_ind:
            
            xi = X[i]
            yi = y[i]
            
            beta = beta - (alpha * grad_L_S(beta, xi, yi))
            
        
        L = eval_L(beta, X, y)
        L_vals.append(L)
        
    return beta, L_vals

#%%
#initial data load and standardization
dataset = sk.datasets.load_breast_cancer()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, train_size=0.8)

mu = np.mean(X_train, axis = 0)
s = np.std(X_train, axis = 0)
X_train = (X_train - mu) / s
X_test = (X_test - mu) / s

#%%
#Using Regular Gradient Descent
#Creating beta vectors and lists of L-values with different learning rates: 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005
#These values are the only values tested for all beta ventor creations in this project
betas_01, L_vals_01 = train_model_grad_desc(X_train, y_train, 0.01, 400)
betas_05, L_vals_05 = train_model_grad_desc(X_train, y_train, 0.05, 400)
betas_001, L_vals_001 = train_model_grad_desc(X_train, y_train, 0.001, 400)
betas_005, L_vals_005 = train_model_grad_desc(X_train, y_train, 0.005, 400)
betas_0001, L_vals_0001 = train_model_grad_desc(X_train, y_train, 0.0001, 400)
betas_0005, L_vals_0005 = train_model_grad_desc(X_train, y_train, 0.0005, 400)

#%%
#Graphing the cost function values for each learning rate
plt.figure()
plt.plot(L_vals_0005, label='alpha = 0.0005')
plt.plot(L_vals_0001, label='alpha = 0.0001')
plt.plot(L_vals_005, label='alpha = 0.005')
plt.plot(L_vals_001, label='alpha = 0.001')
plt.plot(L_vals_05, label='alpha = 0.05')
plt.plot(L_vals_01, label='alpha = 0.01')
plt.title('Cost Function Values vs Iterations (Breast Cancer)')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Values')
plt.legend()

#Based on the graph, it seems that when the learning rate = 0.05, the cost function values converged the fastest
#Additionally, those values are the lowest compared to the other learning rates
#For a = 0.05, the cost calues converged around the 75 iteration mark

#%%
#Creating predictions using the best learning rate (a = 0.05)
y_pred_bc_gd_p = sigmoid(betas_05[0] + X_test@betas_05[1:])
y_pred_bc_gd = (y_pred_bc_gd_p >= 0.5).astype('int64')

#Checking the accuracy score for these predictions
acc_count_bc_gd = (y_pred_bc_gd == y_test).astype('int64')
acc_perc_bc_gd = np.sum(acc_count_bc_gd) / len(acc_count_bc_gd)

#%%
#Using Stochastic Gradient Descent
#Creating the beta vectors and the list of L-values
betas_S_01, L_vals_S_01 = stoch_grad_desc(X_train, y_train, 0.01)
betas_S_05, L_vals_S_05 = stoch_grad_desc(X_train, y_train, 0.05)
betas_S_001, L_vals_S_001 = stoch_grad_desc(X_train, y_train, 0.001)
betas_S_005, L_vals_S_005 = stoch_grad_desc(X_train, y_train, 0.005)
betas_S_0001, L_vals_S_0001 = stoch_grad_desc(X_train, y_train, 0.0001)
betas_S_0005, L_vals_S_0005 = stoch_grad_desc(X_train, y_train, 0.0005)

#%%
#Graphing the L-values from the stochastic gd beta vectors

plt.figure()
plt.plot(L_vals_S_0005, label = 'alpha = 0.0005')
plt.plot(L_vals_S_0001, label = 'alpha = 0.0001')
plt.plot(L_vals_S_005, label = 'alpha = 0.005')
plt.plot(L_vals_S_001, label = 'alpha = 0.001')
plt.plot(L_vals_S_05, label = 'alpha = 0.05')
plt.plot(L_vals_S_01, label = 'alpha = 0.01')
plt.title('Cost Function Values vs Epochs (Breast Cancer)')
plt.xlabel('Epoch')
plt.ylabel('Cost Function Value')
plt.legend()

#Based on the graph, when the learning rate = 0.05, this also yielded the fastest convergence of L-values
#The convergence occurred at around 1-2 epochs

#%%
#Creating predictions using the best learning rate for stochastic gd (a = 0.05)
y_pred_bc_sgd_p = sigmoid(betas_S_05[0]+ X_test@betas_S_05[1:])
y_pred_bc_sgd = (y_pred_bc_sgd_p >= 0.5).astype('int64')

acc_count_bc_sgd = (y_pred_bc_sgd == y_test).astype('int64')
acc_prec_bc_sgd = np.sum(acc_count_bc_sgd) / len(acc_count_bc_sgd)

#%%
#Graphing the L-values from the best learning rate from both regular and stochastic gd methods
plt.figure()
plt.plot(L_vals_05, label='grad desc; a = 0.05')
plt.plot(L_vals_S_05, label = 'stoch grad desc; a = 0.05')
plt.title('Gradient Descent vs Stochastic Gradient Descent')
plt.xlabel('Iteration/Epoch')
plt.ylabel('Cost Function Value')
plt.legend()

#Even though 1 epoch roughly equates to 1 iteration, the stochastic gd method resulted in a faster convergence
#Both of the curves do eventually find a minimizer
#%%
#Initial data load and train/test split for the MNIST dataset
#Had to change the target data (y2) to be a boolean vector where the row is either a 5 or not a 5
dataset2 = sk.datasets.fetch_openml('mnist_784')

X2 = pd.DataFrame(dataset2.data).to_numpy()
y2_raw = dataset2.target.values
y2 = (y2_raw == '5').astype('float64')

X2_train = X2[:60000] / 255
y2_train = y2[:60000]

X2_test = X2[60000:] / 255
y2_test = y2[60000:]

#%%
#Creating beta vectors and lists of L-values using different learning rates (using stochastic gd method)
betas_S2_01, L_vals_S2_01 = stoch_grad_desc(X2_train, y2_train, 0.01)
betas_S2_05, L_vals_S2_05 = stoch_grad_desc(X2_train, y2_train, 0.05)
betas_S2_001 , L_vals_S2_001 = stoch_grad_desc(X2_train, y2_train, 0.001)
betas_S2_005, L_vals_S2_005 = stoch_grad_desc(X2_train, y2_train, 0.005)
betas_S2_0001 , L_vals_S2_0001 = stoch_grad_desc(X2_train, y2_train, 0.0001)
betas_S2_0005 , L_vals_S2_0005 = stoch_grad_desc(X2_train, y2_train, 0.0005)

#%%
#Graphing the L-values for each learning rate
plt.figure()
plt.plot(L_vals_S2_0005, label='alpha = 0.0005')
plt.plot(L_vals_S2_0001, label='alpha = 0.0001')
plt.plot(L_vals_S2_005, label='alpha = 0.005')
plt.plot(L_vals_S2_001, label='alpha = 0.001')
plt.plot(L_vals_S2_05, label='alpha = 0.05')
plt.plot(L_vals_S2_01, label='alpha = 0.01')
plt.title('Cost Function Values vs Epochs (MNIST)')
plt.xlabel('Epoch')
plt.ylabel('Cost Function Values')
plt.legend()

#Although a little bit ambiguous, a = 0.005 and a = 0.01 seem to be the learning rates that converge the fastest
#I chose a = 0.01 as the best learning rate since it seems to converge a little bit faster (past 2-3 epochs) at lower L-values

#%%
#Makig predictions with the best learning rate (a = 0.01)
y_pred_mn_sgd_p = sigmoid(betas_S2_01[0] + X2_test@betas_S2_01[1:])
y_pred_mn_sgd = (y_pred_mn_sgd_p >= 0.5).astype('int64')

#Checking the model accuracy
acc_count_mn_sgd = (y_pred_mn_sgd == y2_test).astype('int64')
acc_perc_mn_sgd = np.sum(acc_count_mn_sgd) / len(acc_count_mn_sgd)

#%%
#Finding the indicies where the model predicted incorrectly
ordered_p_indx = np.argsort(y_pred_mn_sgd_p) #orders the indicies of the prediction probability vector

indxs = []

for i in ordered_p_indx:
    if(np.any(y_pred_mn_sgd[i] != y2_test[i])):
        indxs.append(i)

#From the indicies that were predicted incorrectly, finds the indicies that had probabilites >0.95 or <0.05
indxs95 = []
for o in indxs:
    if(y_pred_mn_sgd_p[o] > 0.95):
        indxs95.append(o)

indxs05 = []
for n in indxs:
    if(y_pred_mn_sgd_p[n] < 0.05):
        indxs05.append(n)
        
#%%
#The 8 most "confusing" images
#Four of these images are where the model confidently predicted "not a 5", but was incorrect
#The other four images are where the model confidently predicted "is a 5", but was incorrect
row1 = np.reshape(X2_test[indxs[0]], (28,28))
row2 = np.reshape(X2_test[indxs[1]], (28,28))
row3 = np.reshape(X2_test[indxs[2]], (28,28))
row4 = np.reshape(X2_test[indxs[3]], (28,28))
row5 = np.reshape(X2_test[indxs[len(indxs) - 1]], (28,28))
row6 = np.reshape(X2_test[indxs[len(indxs) - 2]], (28,28))
row7 = np.reshape(X2_test[indxs[len(indxs) - 3]], (28,28))
row8 = np.reshape(X2_test[indxs[len(indxs) - 4]], (28,28))


figure, axis = plt.subplots(2, 4)
axis[0, 0].imshow(row1)
axis[0, 1].imshow(row2)
axis[0, 2].imshow(row3)
axis[0, 3].imshow(row4)
axis[1, 0].imshow(row5)
axis[1, 1].imshow(row6)
axis[1, 2].imshow(row7)
axis[1, 3].imshow(row8)
plt.show()

#The model may have made incorrect predictions on these images because they generally are either not complete, or they can look like other numbers
#Some of these elements either have some part that is either missing or added onto the image
#These additional/missing parts most likely threw off the model
