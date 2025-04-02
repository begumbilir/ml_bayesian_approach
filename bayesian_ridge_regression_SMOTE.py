# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 21:15:17 2025

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


smote=SMOTE(sampling_strategy='minority') 
# load dataset
df = pd.read_csv("winequality-red.csv")

X = df.drop(columns=["quality"]).values  #wine data
y = df["quality"].values #wine quality

#SMOTE:
x_SMOTE,y_SMOTE=smote.fit_resample(X,y)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_SMOTE = scaler.fit_transform(x_SMOTE)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# def em_bayesian_regression(X, y, max_iter=100, tol=1e-3):
#     """ EM algorithm for Bayesian Linear Regression to estimate α and β """
    
#     model = BayesianRidge(compute_score=True, alpha_init = 1, lambda_init = 1)
#     alpha_old, lambda_old = 1, 1
#     model.fit(X, y)
#     for i in range(max_iter):
#         alpha_old, lambda_old = model.alpha_, model.lambda_
        
#         # E-Step: Fit the model with current alpha and lambda
#         model.fit(X, y)
        
#         # M-Step: Update alpha and lambda
#         alpha_new, lambda_new = model.alpha_, model.lambda_
        
#         # Check for convergence
#         if abs(alpha_new - alpha_old) < tol and abs(lambda_new - lambda_old) < tol:
#             break
    
#     return model

## train the Bayesian Linear Regression model using EM
#model = em_bayesian_regression(X_train, y_train)

"""
In the empirical Bayes approach (Bayesian Ridge Regression) , the parameters  α  and  β  are obtained by maximizing 
the evidence function. The empirical Bayes algorithm infers  α  and  β  from data, 
therefore, no tuning is required in this approach.

"""

model = BayesianRidge(compute_score=True)
model.fit(X_scaled_SMOTE, y_SMOTE)
# predictions
y_pred = model.predict(X_scaled)

y_class = np.where(y >= 7, 2, np.where((y > 4) & (y < 7), 1, 0))
y_pred_class = np.where(y_pred >= 7, 2, np.where((y_pred > 4) & (y_pred < 7), 1, 0))

# evaluate model
accuracy = accuracy_score(y_class, y_pred_class)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_class, y_pred_class))



# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Estimated α (prior precision): {model.alpha_:.6f}")
print(f"Estimated β (likelihood precision): {model.lambda_:.6f}")
# print(f"Test RMSE: {rmse:.4f}")

alpha = model.lambda_
beta = model.alpha_ #noise precision


#alpha = 1/model.lambda_
#beta = 1/model.alpha_



I = np.eye(X_scaled.shape[1])
Sigma = np.linalg.inv(alpha * I + beta * np.dot(X_scaled.T, X_scaled))
y_std = np.sqrt(np.sum(np.dot(X_scaled, Sigma) * X_scaled, axis=1))
mu_bayesian_ridge = beta * np.dot(Sigma, np.dot(X_scaled.T, y)) 



data = pd.DataFrame({
    'Actual': y,      # actual wine quality
    'Predicted': y_pred,   # predicted wine quality
    'Uncertainty': y_std   # model uncertainty (std dev)
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
plt.title("Bayesian Ridge Regression: Ordered Predictions vs. Actual Wine Quality")
plt.legend()
plt.show()

# compute MSE
mse = np.mean((y - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')




