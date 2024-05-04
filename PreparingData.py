import pandas as pd 

class PreparingData:
  def __init__(self):
    self.data = None

  def Merging(self, data_frames):
    merged_data = data_frames[0]
    for df in data_frames[1:]:
      merged_data = merged_data.merge(df, on='Date', how='outer')
    return merged_data
