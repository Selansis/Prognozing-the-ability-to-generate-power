import pandas as pd
from DataProcessing import DataProcessing
from PreparingData import PreparingData

class PercentCounting:
  def Counting(self, data, horizon, operation):
    
    prepering_data = PreparingData()
    prognozing_data_merged = prepering_data.Merging(data)
    model = DataProcessing(prognozing_data_merged)
    r2_gen,mse_gen,Progpred_gen = model.CreatingModel(horizon, operation, 'generation')
    
    capacity = pd.read_csv('data/Energy Poland Installed capacity.csv')
    model = DataProcessing(capacity)
    r2_cap,mse_cap,Progpreg_cap = model.CreatingModel(horizon,operation,'capacity')
    resultPercente = []

    for num in range(horizon):
      resultPercente.append(Progpred_gen[num] / Progpreg_cap[num] * 100)
    return r2_cap, mse_cap, resultPercente[0]