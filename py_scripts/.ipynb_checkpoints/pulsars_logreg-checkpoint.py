import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics


print("Hello!\n\nThis .py script implements a Logistic Regression classifier \
to predict whether or not an observation is a pulsar\n")

print("The data is very unbalanced, 90% noise and 10% pulsars\n")
print("I'm hoping to minimize the number of false negatives\n")

#first let's pull the df and give it column names
column_headers = [	'mean','st_dev','exc_kurtosis','skewness',
					'dm_mean','dmst_dev','dmexc_kurtosis','dmskewness',
					'pulsar'	]

pulsar_df = pd.read_csv("../dataset/HTRU_2.csv",names=column_headers)


#set up X and y
X = pulsar_df.drop(columns=['pulsar'])
y = pulsar_df['pulsar']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y,test_size=0.5)

#instantiate logreg, scale and fit the data
sc = StandardScaler()
logreg = LogisticRegression()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

logreg.fit(X_train_sc,y_train)

#score model
print(f"Train score: {logreg.score(X_train_sc,y_train)}")
print(f"Test score: {logreg.score(X_test_sc,y_test)}\n")

#predictions and reports
y_preds = logreg.predict(X_test_sc)

print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test,y_preds)}\n")

print(f"Classification report:\n{metrics.classification_report(y_preds,y_test)}")
