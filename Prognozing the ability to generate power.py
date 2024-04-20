import csv 
import pandas as pd
capacity = []
mixEnergy = []
hourMeteo = []

with open('Energy Poland Installed capacity.csv', 'r') as file:
  reader = csv.reader(file)
  for row in reader:
    capacity.append(row)


with open('Mix Energy Poland.csv', 'r') as file:
  reader = csv.reader(file)
  for row in reader:
    mixEnergy.append(row)

with open('Godzinowe dane meteo dla poszczególnych lokalizacji.csv') as file:
  reader = csv.reader(file)
  for row in reader:
    hourMeteo.append(row)

