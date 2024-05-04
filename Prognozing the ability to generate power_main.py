import pandas as pd 
import matplotlib.pyplot as plt
from DataProcessing import DataProcessing
from PreparingData import PreparingData

capacity = pd.read_csv('data/Energy Poland Installed capacity.csv', usecols=['Date', 'Installed capacity solar Poland', 'Installed capacity energy_storage Poland'])
mixEnergy = pd.read_csv('data/Mix Energy Poland.csv', usecols =['Date','Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
hourMeteo = pd.read_csv('data/Godzinowe dane meteo dla poszczególnych lokalizacji.csv')
before_cloud = pd.read_csv('data/przed_uwzg_zachmurzenia.csv', encoding='utf-8')
after_cloud = pd.read_csv('data/po_uwzg_zachmurzenia.csv')


data_frames = [capacity, mixEnergy, hourMeteo, before_cloud, after_cloud]
prepering_data = PreparingData()
merged_data = prepering_data.Merging(data_frames)
merged_data = merged_data.drop(columns='Date')
model = DataProcessing(merged_data)
r2,mse,ypred = model.modelling(48,'nn')
print(ypred)

