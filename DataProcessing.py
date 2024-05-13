import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class DataProcessing:
  def __init__(self, data):
    self.data = data

  def filtering(self, input_regex):
    return self.data.filter(regex='Date|capacity solar|Aggregated|'+ input_regex)

  def CreatingModel(self, horizon, operation, prognozing):
    self.data = self.data.drop(columns='Date')
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    if prognozing == "generation":
      Y = self.data['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar']
      X = self.data.drop(columns=['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])

    elif prognozing == "capacity":
      Y = self.data['Installed capacity solar Poland']
      X = self.data.drop(columns=['Installed capacity solar Poland'])


    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

    Y_preds = []
    Y_tests = []
    for i in range(horizon):
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled[:-i-1 if i > 0 else None], Y_scaled[:-i-1 if i > 0 else None], test_size=1, random_state=42)
        if operation == "nn":
            model = MLPRegressor(hidden_layer_sizes=(25,), activation='relu', random_state=42)
        elif operation == "gradient":                
            model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, Y_train.ravel())
        Y_pred = model.predict(X_test)
        Y_pred = scaler_Y.inverse_transform(Y_pred.reshape(-1, 1))
        Y_preds.append(Y_pred)
        Y_tests.append(Y_test)

    Y_preds_parallel = Parallel(n_jobs=-1)(delayed(model.predict)(X_test) for X_test in X_scaled)
    Y_pred_all = np.concatenate(Y_preds_parallel)
    Y_pred_all = np.concatenate(Y_preds)
    Y_test_all = np.concatenate(Y_tests)
    mse = mean_squared_error(Y_test_all, Y_pred_all)
    R2 = r2_score(Y_test_all, Y_pred_all)
    return R2, mse, Y_pred_all


