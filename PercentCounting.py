from DataProcessing import DataProcessing
from PreparingData import PreparingData

class PercentCounting:
  def Counting(self, data_frames, horizon, operation):
    prepering_gen_data = PreparingData()
    prognozing_data_merged = prepering_gen_data.Merging(data_frames)
    model = DataProcessing(prognozing_data_merged)
    r2_gen,mse_gen,Progpred_gen = model.CreatingModel(horizon, operation, 'generation')
    prepering_cap_data = PreparingData()
    prognozing_data_merged = prepering_cap_data.Merging(data_frames)
    model = DataProcessing(prognozing_data_merged)
    r2_cap,mse_cap,Progpreg_cap = model.CreatingModel(horizon,operation,'capacity')
    resultPercente = []
    for num in Progpred_gen:
      resultPercente[num] = Progpred_gen[num]/Progpreg_cap[num]*100
    return r2_gen, mse_gen, resultPercente