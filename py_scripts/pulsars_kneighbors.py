#UCI data source: https://archive.ics.uci.edu/ml/datasets/HTRU2

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.model_selection import GridSearchCV


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


#############################
# GridSearchCV to find the best parameters

print('*********************************************************')
print(f"\nNOW we introduce a GridSearch to find the best parameters")

parameters_grid = {'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],
                  'weights' : ['uniform', 'distance'],
                  'leaf_size' : [10, 20, 30, 40, 50, 60]}

grid_cv = GridSearchCV(estimator = KNeighborsClassifier(),
                       param_grid=parameters_grid,
                       scoring='recall',
                       cv=5)

grid_cv.fit(X_train_sc, y_train)

print(f"\nThe best parameters are:\n{grid_cv.best_params_}")

print(f"\nGrid train score: {grid_cv.score(X_train_sc,y_train)}")
print(f"Grid test score: {grid_cv.score(X_test_sc,y_test)}\n")

#make predictions
grid_preds = grid_cv.predict(X_test_sc)

print(f"Grid confusion matrix:\n{metrics.confusion_matrix(y_test, grid_preds)}\n")

print(f"Grid classification report:\n{metrics.classification_report(y_test,grid_preds)}")