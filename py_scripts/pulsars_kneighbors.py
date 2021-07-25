#UCI data source: https://archive.ics.uci.edu/ml/datasets/HTRU2

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#create column headers, pull in the dataset
column_headers = ['mean','st_dev','exc_kurtosis','skewness','dm_mean','dmst_dev','dmexc_kurtosis','dmskewness','pulsar']
pulsar_df = pd.read_csv("../dataset/HTRU_2.csv",names=column_headers)

#instantiate scaler, KNN classifier
sc = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=5)

#set up X and y, split and scale the data
X = pulsar_df.drop(columns=['pulsar'])
y = pulsar_df['pulsar']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.4)

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#fit the model and make predictions
knn.fit(X_train_sc,y_train)
y_preds = knn.predict(X_test_sc)

print(f"Train score: {np.round(knn.score(X_train_sc,y_train),4)}")
print(f"Test score: {np.round(knn.score(X_test_sc,y_test),4)}\n")

conf_matrix = metrics.confusion_matrix(y_test,y_preds)
class_report = metrics.classification_report(y_test,y_preds)

print(f"Confusion matrix:\n{conf_matrix}\n")
print(f"Classification report:\n{class_report}")