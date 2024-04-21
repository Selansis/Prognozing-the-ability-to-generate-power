import csv 
import pandas as pd 
capacity = pd.read_csv('Energy Poland Installed capacity.csv')
mixEnergy = pd.read_csv('Mix Energy Poland.csv')
hourMeteo = pd.read_csv('Godzinowe dane meteo dla poszczególnych lokalizacji.csv')
before_cloud = pd.read_csv('przed_uwzg_zachmurzenia.csv')
after_cloud = pd.read_csv('po_uwzg_zachmurzenia.csv')

print(capacity)
