# Prognozing-the-ability-to-generate-power
#### This project aims to develop a predictive model for the capacity to generate power from photovoltaic installations in Poland. By "capacity to generate power," we refer to the percentage share of actual power generation in the total installed capacity.

## Assumptions

#### 
* The territory of Poland will be represented by the locations: Gdańsk, Szczecin, Warsaw, Wrocław, and Kraków, with equal shares.
* For the project, it is assumed that the installed capacity will change at a rate determined by the trend from the last 5 years.
* The model will be developed for two-time horizons of forecast granularity: 48 and 72 hours.
* The final will be developed using the following methods:
    * Simulated annealing
    * Neural network
* The quality of the model will be measured by the following metrics:
    * Mean squared error
    * R-squared
* Comparison of the quality and utility of the developed models will be performed for both methods and both time horizons of forecasts.
* Models will be developed in the following environments:
   * Python
