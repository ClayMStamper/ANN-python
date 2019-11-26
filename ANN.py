import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


#Split data into training set and test set
from sklearn.model_selection import train_test_split
inputTrain, inputTest, outputTrain, outputTest = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature scaling
#method for normalizing the range of features (X variables)
from sklearn.preprocessing import StandardScaler
inputScalar = StandardScaler()
# inputTrain = inputScalar.fit_transform(inputTrain) #calculate mean and std_dev then scale
# inputTest = inputScalar.transform(inputTest) # just scale