# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:04:16 2025

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("winequality-red.csv")

X = df.drop(columns=['quality'])
y = df['quality']


# Create histogram of quality values
plt.figure(figsize=(8, 6))
plt.hist(y, bins=range(y.min(), y.max() + 2), edgecolor='black', alpha=0.7, align='left')
plt.title('Histogram of Wine Quality Values')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.xticks(range(y.min(), y.max() + 1))  # Ensure all quality values are shown
plt.grid(True)
plt.show()

# Convert wine quality into binary classification (excellent/normal/bad)
y = np.where(y >= 7, 2, np.where((y > 4) & (y < 7), 1, 0))


# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Try different values of k
# for k in [3, 5, 7]:
#     # Train KNN model
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)

#     # Evaluate model
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'K = {k} → Accuracy: {accuracy:.4f}')
#     print(classification_report(y_test, y_pred))
#     print("="*50)

# Define k values and fold values
k_values = [3, 5, 7]
kf_values = [5, 10]

# Perform k-fold cross-validation for each k in k-NN
for k in k_values:
    print(f'\nEvaluating k-NN with k = {k}\n' + "="*50)
    
    for kf in kf_values:
        kf_cv = KFold(n_splits=kf, shuffle=True, random_state=42)
        model = KNeighborsClassifier(n_neighbors=k)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf_cv, scoring='accuracy')
        
        print(f'K-Fold = {kf} → Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    
    # Train on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Set Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))
