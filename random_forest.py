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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# load dataset
df = pd.read_csv("winequality-red.csv")  

X = df.drop(columns=['quality'])
y = df['quality']

# convert wine quality into binary classification (excellent/normal//bad)
y = np.where(y >= 7, 2, np.where((y > 4) & (y < 7), 1, 0))

# split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# normalize features
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