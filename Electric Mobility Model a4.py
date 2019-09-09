# -*- coding: utf-8 -*-
"""
Electric Mobility Model 
@author: kholo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
file="ModelSpecification_a4.xlsx"
dataset = pd.read_excel(file, sheetname='FullData_1')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Calculating MAE & MAPE
mae_sum = np.sum((abs(y_test-y_pred)))
mae = mae_sum / len(y_test)
print(mae)
mape_sum = np.sum(abs(((y_test - y_pred)/y_test)))
mape = mape_sum / len(y_test)
print(mape)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
y = np.reshape(y, (len(y), -1))
sc_y = StandardScaler()
y = sc_y.fit_transform(y)"""

# Model Optimization
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((212, 1)).astype(int), values = X, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues[1:]).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    print(regressor_OLS.summary())
        else:
            break                                        
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
X_Modeled = backwardElimination(X_opt, SL)
regressor_OLS = sm.OLS(y, X_Modeled).fit()
print(regressor_OLS.summary())

# Calculating Regression Error Metrics
mae_sum = 0
for item, x in zip(y, X_Modeled):
    
    y_pred = regressor_OLS.predict(x)
    mae_sum += abs(item - y_pred)
mae = mae_sum / len(y)
print(mae)

mape_sum = 0
for item, x in zip(y, X_Modeled):
    y_pred = regressor_OLS.predict(x)
    mape_sum += abs((item - y_pred)/item)
mape = mape_sum/len(y)
print(mape)