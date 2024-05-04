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

  def modelling(self, horizon, operation):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(self.data)
    data = pd.DataFrame(scaled_data, columns=self.data.columns)
    data.loc[:, :] = data.ffill()
    X = data.drop(columns=['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
    Y = data['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar']
    def train_model(model_type):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=horizon, random_state=42)
        if model_type == "nn":
            model = MLPRegressor(hidden_layer_sizes=(25,), activation='relu', random_state=42, max_iter=1000)
        elif model_type == "gradient":                
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, Y_train)
        return model, X_test, Y_test
    models = Parallel(n_jobs=-1)(delayed(train_model)(operation) for _ in range(horizon))
      
    Y_preds = []
    Y_tests = []
    for model, X_test, Y_test in models:
      Y_pred = model.predict(X_test)
      Y_preds.append(Y_pred)
      Y_tests.append(Y_test)

    Y_pred_all = np.concatenate(Y_preds)
    Y_test_all = np.concatenate(Y_tests)
    mse = mean_squared_error(Y_test_all, Y_pred_all)
    R2 = r2_score(Y_test_all, Y_pred_all)
    return R2, mse, Y_pred_all