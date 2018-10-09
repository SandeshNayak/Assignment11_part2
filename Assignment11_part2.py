#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#importing the dataset
boston = load_boston()

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(boston.target, columns=["MEDV"])

X = df
y = target

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33, random_state = 5)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lm.predict(X_test)

print("r2_score:",end=" ")
print(r2_score(y_test,y_pred))
print("accuracy of the model:",end=" ")
print(lm.score(X_test,y_test))
print("coefficiants of the features are: ")
print(lm.coef_)


