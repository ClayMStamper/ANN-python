import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense


# Importing the dataset
dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode categorical data
countryEncoder = genderEncoder = LabelEncoder()
X[:, 1] = countryEncoder.fit_transform(X[:, 1])
X[:, 2] = genderEncoder.fit_transform(X[:, 2])
hotEncoder = OneHotEncoder(categorical_features=[1])
X = hotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
# method for normalizing the range of features (X variables)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)  # calculate mean and std_dev then scale
X_test = scalar.transform(X_test)  # just scale


ann = Sequential()
