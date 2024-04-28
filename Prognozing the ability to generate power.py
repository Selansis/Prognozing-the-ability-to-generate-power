from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import csv 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

def Filtering(data,input_regex):
  return data.filter(regex='Date|capacity solar|Aggregated|'+ input_regex)

def Neural_network(data, horizon, operation):
  data.loc[:, :] = data.ffill()
  #data.ffill(inplace = True)
  X = data.drop(columns=['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar','Date'])
  Y = data['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=horizon, random_state=42)
  models = []
  def train_model(model_type):
      if model_type == "nn":
          model = MLPRegressor(hidden_layer_sizes=(25,), activation='relu', random_state=42, solver='adam', alpha=0.05, max_iter=1000)
      elif model_type == "gradient":
          model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
      model.fit(X_train, Y_train)
      return model
  models = Parallel(n_jobs=4)(delayed(train_model)(operation) for _ in range(horizon))
  Y_preds = []
  for model, X_test, _ in models:
      Y_pred = model.predict(X_test)
      Y_preds.append(Y_pred)
  Y_pred_all = np.concatenate(Y_preds)
  mse = mean_squared_error(Y_test, Y_pred_all)
  R2 = r2_score(Y_test, Y_pred_all)
  return R2, mse, models

capacity = pd.read_csv('Energy Poland Installed capacity.csv', usecols=['Date', 'Installed capacity solar Poland', 'Installed capacity energy_storage Poland'])
mixEnergy = pd.read_csv('Mix Energy Poland.csv', usecols =['Date','Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
hourMeteo = pd.read_csv('Godzinowe dane meteo dla poszczególnych lokalizacji.csv')
before_cloud = pd.read_csv('przed_uwzg_zachmurzenia.csv', encoding='utf-8')
after_cloud = pd.read_csv('po_uwzg_zachmurzenia.csv')

capacity = pd.DataFrame(capacity)
mixEnergy = pd.DataFrame(mixEnergy)
hourMeteo = pd.DataFrame(hourMeteo)
before_cloud = pd.DataFrame(before_cloud)
after_cloud = pd.DataFrame(after_cloud)

merged_data = capacity.merge(mixEnergy, on='Date', how='outer') \
    .merge(hourMeteo, on='Date', how='outer') \
    .merge(before_cloud, on='Date', how='outer') \
    .merge(after_cloud, on='Date', how='outer')

krakow_data = Filtering(merged_data,"Krak")
warszawa_data = Filtering(merged_data,"Warszawa")
wroclaw_data = Filtering(merged_data,"Wroc")
gdansk_data = Filtering(merged_data,"Gda")
szczecin_data = Filtering(merged_data,"Szczecin")



#krakow_data.to_csv('nazwa_pliku.csv', index=False)  # index=False oznacza, że nie chcemy eksportować indeksów wierszy


krakow_R2, mse, krakow_models = Neural_network(krakow_data, 48, "nn")
print(krakow_R2)
print(mse)
print(krakow_models)

