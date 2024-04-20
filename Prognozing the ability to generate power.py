import csv 
with open('Energy Poland Installed capacity.csv', 'r') as file:
  capacity = csv.reader(file)

with open('Mix Energy Poland.csv', 'r') as file:
  mixEnergy = csv.reader(file)

with open('Godzinowe dane meteo dla poszczególnych lokalizacji.csv') as file:
  hourMeteo = csv.reader(file)


