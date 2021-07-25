#modified from this source: 
#https://github.com/yixuanzhou/Pulsar-Candidates-Calssification/blob/master/knn.py
#https://github.com/yixuanzhou/

import pandas as pd
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load data set
column_headers = ['mean','st_dev','exc_kurtosis','skewness','dm_mean','dmst_dev','dmexc_kurtosis','dmskewness','pulsar']
df = pd.read_csv("./HTRU_2.csv",names=column_headers)
X = df.drop(columns=['pulsar'])
y = df['pulsar'].values

# Normalization
normalizer = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Train model with fine-tuned parameters
knn = KNeighborsClassifier(n_neighbors=5)
clf_knn = knn.fit(X_train, y_train)

# Cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=95)
res = {}
for scoring in ('f1', 'roc_auc', 'precision', 'recall'):
    res[scoring] = cross_val_score(clf_knn, X_test, y_test, cv=cv, scoring=scoring, n_jobs=-1)
print(f"\nf1 score mean: {np.round(res['f1'].mean(),4)}")
print(f"roc_auc score mean: {np.round(res['roc_auc'].mean(),4)}")
print(f"precision mean: {np.round(res['precision'].mean(),4)}")
print(f"recall mean: {np.round(res['recall'].mean(),4)}\n")


# Final result for training set and test set
print(f"Train score: {np.round(clf_knn.score(X_train, y_train),4)}")
print(f"Test score: {np.round(clf_knn.score(X_test, y_test),4)}\n")

# Plot confusion
y_pred = clf_knn.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test,y_pred)
print(f"Confusion matrix:\n{conf_matrix}\n")
print(f"Classification report:\n{class_report}")
