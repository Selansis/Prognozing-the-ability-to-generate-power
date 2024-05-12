import pandas as pd 

class PreparingData:
  def __init__(self):
    self.data = None

  def Merging(self, data_frames):
    for df in data_frames:
      df['Date'] = pd.to_datetime(df['Date'])
    merged_data = data_frames[0]
    for df in data_frames[1:]:
      merged_data = merged_data.merge(df, on='Date', how='outer')
    merged_data.dropna(inplace=True)
    return merged_data
