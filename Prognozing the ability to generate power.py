from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import csv 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def Filtering(data,input_regex):
  return data.filter(regex='Date|capacity solar|Aggregated|'+ input_regex)

def Neural_network(data, horizon, operation):
  data.fillna(method='ffill', inplace=True)
  X = data.drop(columns=['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar','Date'])
  Y = data['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=horizon, random_state=42)
  models = []
  match operation:
    case "nn":
      for i in range(horizon):
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
        model.fit(X_train, Y_train)
        models.append(model)

    case "gradient":
      for i in range(horizon):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, Y_train)
        models.append(model)

  Y_preds = []
  for model in models:
    Y_pred = model.predict(X_test)
    Y_preds.append(Y_pred)
  
  Y_pred_all = np.concatenate(Y_preds)
  mse = mean_squared_error(Y_test, Y_pred_all)
  R2 = r2_score(Y_test, Y_pred_all)
  return R2, mse

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


krakow_R2, mse = Neural_network(krakow_data, 48, "nn")
print(krakow_R2)
print(mse)
krakow_R2, mse = Neural_network(krakow_data, 48, "gradient")
print(krakow_R2)
print(mse)


'''

capacity_train, capacity_test = train_test_split(capacity, test_size=0.2, random_state=42)
mixEnergy_train, mixEnergy_test = train_test_split(mixEnergy, test_size=0.2, random_state=42)
hourMeteo_train, hourMeteo_test = train_test_split(hourMeteo, test_size=0.2, random_state=42)
before_cloud_train, before_cloud_test = train_test_split(before_cloud, test_size=0.2, random_state=42)
after_cloud_train, after_cloud_test = train_test_split(after_cloud, test_size=0.2, random_state=42)


from sklearn.neural_network import MLPRegressor

# Definicja modelu sieci neuronowej
model_nn_capacity = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model_nn_mixEnergy = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model_nn_hourMeteo = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model_nn_before_cloud = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model_nn_after_cloud = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)

# Trenowanie modeli na danych treningowyc
model_nn_capacity.fit(capacity_train.drop(columns=['Date']), capacity_train['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
model_nn_mixEnergy.fit(mixEnergy_train.drop(columns=['Date']), mixEnergy_train['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
model_nn_hourMeteo.fit(hourMeteo_train.drop(columns=['Date']), hourMeteo_train['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
model_nn_before_cloud.fit(before_cloud_train.drop(columns=['Date']), before_cloud_train['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
model_nn_after_cloud.fit(after_cloud_train.drop(columns=['Date']), after_cloud_train['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
'''