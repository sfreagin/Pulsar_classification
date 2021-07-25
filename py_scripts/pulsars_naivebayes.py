import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics

print("Hello!\n This .py script implements a Naive Bayes classifiers \
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

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.5)

#StandardScaler throws errors because of negative numbers(?)
#so let's use MinMaxScaler()

#sc = StandardScaler()
sc = MinMaxScaler()

#instantiate all the Naive Bayes algorithms
gnb = GaussianNB()
mnb = MultinomialNB()
cnb = ComplementNB()
bnb = BernoulliNB()
#catnb = CategoricalNB()

#############################
# Categorical Naive Bayes throws all sorts of errors here
# so I left the code in but commented it out
#############################

#transform and fit the data
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

gnb.fit(X_train_sc,y_train)
mnb.fit(X_train_sc,y_train)
cnb.fit(X_train_sc,y_train)
bnb.fit(X_train_sc,y_train)
# catnb.fit(X_train_sc,y_train)



print(f"GaussianNB train score: {np.round(gnb.score(X_train_sc, y_train),4)}")
print(f"GaussianNB test score: {np.round(gnb.score(X_test_sc, y_test),4)}\n")

print(f"MultinomialNB train score: {np.round(mnb.score(X_train_sc, y_train),4)}")
print(f"MultinomialNB test score: {np.round(mnb.score(X_test_sc, y_test),4)}\n")

print(f"ComplementNB train score: {np.round(cnb.score(X_train_sc, y_train),4)}")
print(f"ComplementNB test score: {np.round(cnb.score(X_test_sc, y_test),4)}\n")

print(f"BernoulliNB train score: {np.round(bnb.score(X_train_sc, y_train),4)}")
print(f"BernoulliNB test score: {np.round(bnb.score(X_test_sc, y_test),4)}\n")

# print(f"CategoricalNB train score: {np.round(catnb.score(X_train_sc, y_train),4)}")
# print(f"CategoricalNB test score: {np.round(catnb.score(X_test_sc, y_test),4)}\n")

#make predictions
gnb_preds = gnb.predict(X_test_sc)
mnb_preds = mnb.predict(X_test_sc)
cnb_preds = cnb.predict(X_test_sc)
bnb_preds = bnb.predict(X_test_sc)
# catnb_preds = catnb.predict(X_test_sc)

#print confusion matrices
print(f"GaussianNB:\n{metrics.confusion_matrix(y_test,gnb_preds)}\n")
print(f"MultinomialNB:\n{metrics.confusion_matrix(y_test,mnb_preds)}\n")
print(f"ComplementNB:\n{metrics.confusion_matrix(y_test,cnb_preds)}\n")
print(f"BernoulliNB:\n{metrics.confusion_matrix(y_test,bnb_preds)}\n")
# print(f"CategoricalNB:\n{metrics.confusion_matrix(y_test,catnb_preds)}\n")

#print classification reports
print(f"GaussianNB:\n{metrics.classification_report(y_test,gnb_preds)}\n")
print(f"MultinomialNB:\n{metrics.classification_report(y_test,mnb_preds)}\n")
print(f"ComplementNB:\n{metrics.classification_report(y_test,cnb_preds)}\n")
print(f"BernoulliNB:\n{metrics.classification_report(y_test,bnb_preds)}\n")
# print(f"CategoricalNB:\n{metrics.classification_report(y_test,catnb_preds)}\n")
