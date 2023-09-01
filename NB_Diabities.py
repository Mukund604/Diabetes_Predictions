#Importing the required Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Importing the datset using Pandas
df = pd.read_csv("/Users/mukund/Desktop/MachineLearning/Udemy/Diabities_Prediciton/diabetes_prediction_dataset.csv")

#Splitting into Target and the data
y = df.diabetes
X = df.drop('diabetes', axis='columns')

#Processing the data using Dummy variables and one hot encoding
dummies = pd.get_dummies(X.gender).astype(int)
X = pd.concat([X,dummies], axis='columns')
X.drop('gender', axis='columns', inplace=True)
dummies2 = pd.get_dummies(X.smoking_history).astype(int)
X = pd.concat([X,dummies2], axis='columns')
X.drop('smoking_history', axis='columns', inplace=True)

#Splitting the dataset into Traning and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Initializing the Classifier and Training the model
clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Checking the Accuracy of the model
def Accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


print("The Accuracy of the model is:",  Accuracy(y_test, y_pred))

