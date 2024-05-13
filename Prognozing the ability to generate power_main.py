import pandas as pd 
import matplotlib.pyplot as plt
from DataProcessing import DataProcessing
from PercentCounting import PercentCounting
from PreparingData import PreparingData

capacity = pd.read_csv('data/Energy Poland Installed capacity.csv')#, usecols=['Date', 'Installed capacity solar Poland'])
mixEnergy = pd.read_csv('data/Mix Energy Poland.csv', usecols =['Date','Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
hourMeteo = pd.read_csv('data/Godzinowe dane meteo dla poszczególnych lokalizacji.csv')
before_cloud = pd.read_csv('data/przed_uwzg_zachmurzenia.csv', encoding='utf-8')
after_cloud = pd.read_csv('data/po_uwzg_zachmurzenia.csv')
hourMeteo = hourMeteo.filter(regex='Date|cloud')

# generation for 48hours nn 
data_frames = [mixEnergy, hourMeteo, before_cloud, after_cloud]
preparing_gen_data = PercentCounting()
r2_48nn,mse_48nn,Prog48nn = preparing_gen_data.Counting(data_frames, 48,  'gradient')
print(Prog48nn)
#r2_48gradient,mse_48gradient,Prog48gradient = preparing_gen_data.Counting(data_frames, 48,  'nn')
#r2_48nn,mse_48nn,Prog48nn = preparing_gen_data.Counting(data_frames, 72,  'gradient')
#r2_72gradient,mse_72gradient,Prog72gradient = preparing_gen_data.Counting(data_frames, 72,  'gradient')

"""Performs data modeling. 
Parameters: 
data (list): Dataframes we want to use.
horizon (int): Number of forecasting periods. 
operation (str): Type of modeling operation. It can be 'nn' for neural network or 'gradient' for gradient regression. 
prognozing (str): Type of prognozing data. It can be 'generation' for aggreagated generation or 'capacity' for predicting capacity. 
Returns: the coefficient of determination (R^2 score), mean squared error (MSE), and predictions."""

  