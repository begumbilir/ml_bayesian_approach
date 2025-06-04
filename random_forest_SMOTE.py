# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:42:15 2025

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

smote=SMOTE(sampling_strategy='minority') 

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

# train Random forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", 'Class2'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()