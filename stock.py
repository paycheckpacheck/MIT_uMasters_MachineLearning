import openpyxl
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

stock_data_file = 'HistoricalData_1681579205818.xlsx'

# Load the stock data from an excel file
wb = openpyxl.load_workbook(stock_data_file)
ws = wb.active
data = ws.values

# Convert the data to a pandas dataframe
df = pd.DataFrame(data, columns=['Date', 'Close/Last', 'Volume','Open', 'High', 'Low'])





# Split the data into features and target
X = df[['Volume','Open', 'High', 'Low']].values
y = df['Close/Last'].values

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and fit a SGD regressor
sgd = SGDRegressor()
sgd.fit(X_train, y_train)

# Predict the test set and evaluate the performance
y_pred = sgd.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', mse)
