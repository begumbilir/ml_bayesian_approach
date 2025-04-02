# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:49:54 2025

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

smote=SMOTE(sampling_strategy='minority') 

# Load dataset
df = pd.read_csv("winequality-red.csv")

X = df.drop(columns=['quality'])
y = df['quality']
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#SMOTE:
x_SMOTE,y_SMOTE=smote.fit_resample(X_train,y_train)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
x_SMOTE,y_SMOTE=smote.fit_resample(x_SMOTE,y_SMOTE)
X_train,y_train=smote.fit_resample(x_SMOTE,y_SMOTE)

# Classify:
y_train = np.where(y_train >= 7, 2, np.where((y_train > 4) & (y_train < 7), 1, 0))
y_test = np.where(y_test >= 7, 2, np.where((y_test > 4) & (y_test < 7), 1, 0))

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

#############################################################################3
# Naïve Bayes algorithm from scratch
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # adding a small value to avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[c], self.variances[c]))) #assuming features follow a Gaussian distribution
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
    
    
# train Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

# Evaluate model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Accuracy: {accuracy_nb:.4f}')
print(classification_report(y_test, y_pred_nb))
