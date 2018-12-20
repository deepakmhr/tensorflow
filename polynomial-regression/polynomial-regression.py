# Multiple Linear Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show():
    # Importing the datasets

    datasets = pd.read_csv('Position_Salaries.csv')
    X = datasets.iloc[:, 1:2].values
    Y = datasets.iloc[:, 2].values

    # Fitting Polynomial Regression to the dataset


    poly_reg = PolynomialFeatures(degree=4)
    X_Poly = poly_reg.fit_transform(X)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_Poly, Y)

    # Visualising the Polynomial Regression results
    X_Grid = np.arange(min(X), max(X), 0.1)
    X_Grid = X_Grid.reshape((len(X_Grid), 1))
    plt.scatter(X, Y, color='red')
    plt.plot(X_Grid, lin_reg_2.predict(
        poly_reg.fit_transform(X_Grid)), color='blue')
    plt.title('Polynomial Regression results')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # Predicting a new result with the Polynomial Regression

    lin_reg_2.predict(poly_reg.fit_transform(6.5))

show()
