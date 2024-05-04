import pandas as pd 
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from DataProcessing import DataProcessing

#def Filtering(data,input_regex):
#  return data.filter(regex='Date|capacity solar|Aggregated|'+ input_regex)
'''
def Learning(data, horizon, operation):
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)
  data = pd.DataFrame(scaled_data, columns=data.columns)
  data.loc[:, :] = data.ffill()
  #data.ffill(inplace = True)
  X = data.drop(columns=['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
  Y = data['Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar']
  
  models = []
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
'''





capacity = pd.read_csv('data/Energy Poland Installed capacity.csv', usecols=['Date', 'Installed capacity solar Poland', 'Installed capacity energy_storage Poland'])
mixEnergy = pd.read_csv('data/Mix Energy Poland.csv', usecols =['Date','Aggregated Generation Per Type, PSE SA CA, Actual Generation Output, Solar'])
hourMeteo = pd.read_csv('data/Godzinowe dane meteo dla poszczególnych lokalizacji.csv')
before_cloud = pd.read_csv('data/przed_uwzg_zachmurzenia.csv', encoding='utf-8')
after_cloud = pd.read_csv('data/po_uwzg_zachmurzenia.csv')

capacity = pd.DataFrame(capacity)
mixEnergy = pd.DataFrame(mixEnergy)
hourMeteo = pd.DataFrame(hourMeteo)
before_cloud = pd.DataFrame(before_cloud)
after_cloud = pd.DataFrame(after_cloud)

merged_data = capacity.merge(mixEnergy, on='Date', how='outer') \
    .merge(hourMeteo, on='Date', how='outer') \
    .merge(before_cloud, on='Date', how='outer') \
    .merge(after_cloud, on='Date', how='outer')
merged_data = merged_data.drop(columns='Date')


model = DataProcessing(merged_data)
result = model.modelling(48,'nn')


