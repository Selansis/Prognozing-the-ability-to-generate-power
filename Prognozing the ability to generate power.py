def filtering(data,input_regex):
  return data.filter(regex='Date|capacity solar|Aggregated|'+ input_regex)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import csv 
import matplotlib.pyplot
import pandas as pd 
import matplotlib.pyplot as plt
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

krakow_data = filtering(merged_data,"Krak")
warszawa_data = filtering(merged_data,"Warszawa")
wroclaw_data = filtering(merged_data,"Wroc")
gdansk_data = filtering(merged_data,"Gda")
szczecin_data = filtering(merged_data,"Szczecin")

print(krakow_data)
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