import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import mlflow

mlflow.autolog()

dataset = pd.read_csv('./data/original.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# print(dataset)
# print(X)
# print(Y)

mlflow.set_tracking_uri('http://52.205.59.55/')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.predict([[5]]))