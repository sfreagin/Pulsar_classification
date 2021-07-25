import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics


#first let's pull the df and give it column names
column_headers = [	'mean','st_dev','exc_kurtosis','skewness',
					'dm_mean','dmst_dev','dmexc_kurtosis','dmskewness',
					'pulsar'	]

pulsar_df = pd.read_csv("../dataset/HTRU_2.csv",names=column_headers)

################
#set up X and y
X = pulsar_df.drop(columns=['pulsar'])
y = pulsar_df['pulsar']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y,test_size=0.5)

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


################
#create and fit the model

model = Sequential()

model.add(Dense(8,input_dim=8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train_sc,y_train,epochs=50,batch_size=32)

###############
#make predictions and round answer to 0 or 1
y_preds = model.predict(X_test_sc)
y_preds = [ 1 if num >=0.5 else 0 for num in y_preds] 

evaluation = model.evaluate(X_test_sc,y_test)

###############
#results

print("Hello!\n\nThis .py script implements a Logistic Regression classifier \
to predict whether or not an observation is a pulsar\n")

print("The data is very unbalanced, 90% noise and 10% pulsars\n")
print("I'm hoping to minimize the number of false negatives\n")


print(f"\n\nModel evaluation:\n{evaluation}\n")
print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test,y_preds)}\n")
print(f"Classification report:\n{metrics.classification_report(y_test,y_preds)}")