# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:21:35 2025

@author: user
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report



# load dataset
df = pd.read_csv("winequality-red.csv")  

X = df.drop(columns=["quality"]).values  #wine data
y = df["quality"].values #wine quality

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_class = np.where(y >= 7, 2, np.where((y > 4) & (y < 7), 1, 0))


def bayesian_linear_regression(X, y, alpha, beta):
    """Performs Bayesian linear regression given data, prior precision (alpha), 
    and noise precision (beta)."""
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias term
    
    # Compute posterior mean and covariance
    I = np.eye(X.shape[1])
    Sigma = np.linalg.inv(alpha * I + beta * np.dot(X.T, X))
    mu = beta * np.dot(Sigma, np.dot(X.T, y))
    return mu, Sigma

# predict wine quality (continuous values)
def predict(X, mu, Sigma):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias term
    predicted_values = np.dot(X, mu)
    prediction_variance = np.sum(np.dot(X, Sigma) * X, axis=1)    
    return predicted_values, prediction_variance



# Hyperparameter tuning using k-fold cross-validation
def tune_hyperparameters(X, y, alpha_vals, beta_vals, k=5):
    best_alpha, best_beta = None, None
    best_mse = float("inf")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for alpha in alpha_vals:
        for beta in beta_vals:
            mse_scores = []
            
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                mu, Sigma = bayesian_linear_regression(X_train, y_train, alpha, beta)
                y_pred, _ = predict(X_val, mu, Sigma)
                mse_scores.append(mean_squared_error(y_val, y_pred))
            
            avg_mse = np.mean(mse_scores)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha, best_beta = alpha, beta
    
    return best_alpha, best_beta

# Define possible values for alpha and beta
alpha_values = np.logspace(-3, 3, 10)
beta_values = np.logspace(-3, 3, 10)

# Perform tuning
best_alpha, best_beta = tune_hyperparameters(X_scaled, y, alpha_values, beta_values)
print(f"Best Alpha: {best_alpha}, Best Beta: {best_beta}")

# Train the final model with the best hyperparameters
mu_opt, Sigma_opt = bayesian_linear_regression(X_scaled, y, best_alpha, best_beta)

y_pred, y_var = predict(X_scaled, mu_opt, Sigma_opt)
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
plt.title("Bayesian Linear Regression with K-Fold CV: Ordered Predictions vs. Actual Wine Quality")
plt.legend()
plt.show()


# compute MSE
mse = np.mean((y - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')


