# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:59:05 2025

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# load dataset
df = pd.read_csv("winequality-red.csv")  

X = df.drop(columns=["quality"]).values  #wine data
y = df["quality"].values #wine quality

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_class = np.where(y >= 7, 2, np.where((y > 4) & (y < 7), 1, 0))


# Bayesian Linear Regression using the EM Algorithm
def em_bayesian_linear_regression(X, y, max_iter=100, tol=1e-6):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias term
    N, D = X.shape
    
    # initialize alpha and beta
    alpha = 1.0
    beta = 1.0
    
    # initialize mean and covariance
    I = np.eye(D)
    Sigma = np.linalg.inv(alpha * I + beta * np.dot(X.T, X))
    mu = beta * np.dot(Sigma, np.dot(X.T, y))
    
    for _ in range(max_iter):
        # E-step: Compute expected values
        Sigma = np.linalg.inv(alpha * I + beta * np.dot(X.T, X))
        mu = beta * np.dot(Sigma, np.dot(X.T, y))
        
        # # M-step: Update alpha and beta
        # gamma = np.trace(np.dot(Sigma, np.linalg.inv(Sigma + np.dot(mu[:, None], mu[None, :]))))
        # alpha_new = gamma / np.dot(mu, mu)
        # beta_new = (N - gamma) / np.sum((y - np.dot(X, mu)) ** 2)
        
  
        alpha_new =     D / (np.dot(mu, mu) + np.trace(Sigma))
        beta_new = N  / (np.sum((y - np.dot(X, mu)) ** 2) + np.trace(np.dot(X, np.dot(Sigma, X.T))))
        
        
        # Check for convergence
        if np.abs(alpha_new - alpha) < tol and np.abs(beta_new - beta) < tol:
            break
        
        alpha, beta = alpha_new, beta_new
    
    return mu, Sigma, alpha, beta

# predict wine quality (continuous values)
def predict(X, mu, Sigma):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias term
    predicted_values = np.dot(X, mu)
    prediction_variance = np.sum(np.dot(X, Sigma) * X, axis=1)    
    return predicted_values, prediction_variance


# Train model using EM algorithm
mu_opt_EM, Sigma_opt_EM, alpha_opt_EM, beta_opt_EM = em_bayesian_linear_regression(X_scaled, y)
print(f"Optimized Alpha: {alpha_opt_EM}, Optimized Beta: {beta_opt_EM}")


y_pred, y_var = predict(X_scaled, mu_opt_EM, Sigma_opt_EM)
y_pred_class = np.where(y_pred >= 7, 2, np.where((y_pred > 4) & (y_pred < 7), 1, 0))

# evaluate model
accuracy = accuracy_score(y_class, y_pred_class)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_class, y_pred_class))


data = pd.DataFrame({
    'Actual': y,      # actual wine quality
    'Predicted': y_pred,   # predicted wine quality
    'Uncertainty': np.sqrt(y_var)   # model uncertainty (std dev)
})

# sort the data by actual wine quality (ascending order)
data_sorted = data.sort_values(by='Actual').reset_index(drop=True)

# scatter plot with error bars
plt.figure(figsize=(12, 6))
plt.errorbar(data_sorted.index, data_sorted['Predicted'], yerr=data_sorted['Uncertainty'], 
             fmt='o', color='blue', alpha=0.5, label='Predicted (with uncertainty)')
plt.scatter(data_sorted.index, data_sorted['Actual'], color='red', marker='x', label='Actual')

plt.xlabel("Ordered Sample Index")
plt.ylabel("Wine Quality")
plt.title("Bayesian EM: Ordered Predictions vs. Actual Wine Quality")
plt.legend()
plt.show()


# compute MSE
mse = np.mean((y - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')


