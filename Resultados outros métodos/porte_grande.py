# -*- coding: utf-8 -*-
"""porte_grande.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MhIHO8zRCibnxWnCQ7txFU_o-7YY15mE
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

dataset = pd.read_csv('/content/drive/My Drive/Projeto Internações/Preparação dos dados/dataset_internacoes_completo.csv')

dataset

dataset['populacao'].dtype

def porte(populacao):
  if populacao <= 20000:
    return 'Pequeno Porte I'
  elif populacao >= 20001 and populacao <= 50000 :
    return 'Pequeno Porte II'
  elif populacao >= 50001 and populacao <= 100000:
    return 'Médio Porte'
  elif populacao >= 100001 and populacao <= 900000:
    return 'Grande Porte'
  elif populacao >= 900001:
    return 'Metrópole'

def taxa_internacao (row):
  return row['Qtd. internacoes']*1000/row['populacao']

dataset['Taxa Internacoes'] = dataset.apply(taxa_internacao, axis =1)

dataset['Porte'] = dataset['populacao'].apply(porte)

dataset

"""Filtro"""

dataset = dataset[dataset['Porte'] == 'Grande Porte']

dataset

time_series = dataset[['Data completa', 'Taxa Internacoes']]
time_series['Data completa'] = pd.to_datetime(time_series['Data completa'])

time_series = time_series.set_index('Data completa').resample('M').mean()

time_series = time_series[:'2019-12-31']

time_series.describe()

time_series.hist()

plt.plot(time_series)
plt.title('Taxa de Internações', fontsize=24)
plt.ylabel('Taxa Internações')
plt.xlabel('Data')
plt.show()

rcParams['figure.figsize'] = 15, 6

"""#Teste de Estacionariedade"""

from statsmodels.tsa.stattools import adfuller

X = time_series['Taxa Internacoes']
result = adfuller(X)
print('ADF Estatíticas: %f' % result[0])
print('Valor de P: %f' % result[1])
print('Valores Críticos:')
for key, value in result[4].items():
   print('\t%s: %.3f' % (key, value))

"""#Tornando a série estacionária (Diferenciação)"""

xdiff = X.diff()
xdiff = xdiff.dropna()

xlabel='Data'
xdiff.plot()

result = adfuller(xdiff)
print('ADF Estatíticas: %f' % result[0])
print('Valor de P: %f' % result[1])
print('Valores Críticos:')
for key, value in result[4].items():
   print('\t%s: %.3f' % (key, value))

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(X)
plt.show()

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

"""#Suavização Exponencial Simples

"""

X_base = X[:'2018-12-31']

X_base

fit1 = SimpleExpSmoothing(X_base).fit(smoothing_level=0.2,optimized=False) #alpha = 0.2: 20% de peso para as observações mais recentes, optimazed: acha um valor otimizado para o smoothing_level
fcast1 = fit1.forecast(12)

fcast1

fcast1.plot(marker='.', color='blue', legend=True, label = 'Previsto')
X.plot(marker='.',  color='red', label = 'Real', legend = True)
plt.show()

real = X['2019-01-01':]

real

mean_absolute_error(real, fcast1)

np.sqrt(mean_squared_error(real, fcast1))

mean_squared_error(real, fcast1)

mean_absolute_percentage_error(real, fcast1)

"""#Holt"""

fit1 = Holt(X_base).fit(smoothing_level=0.2, smoothing_trend=0.8, optimized=False)
fcast1 = fit1.forecast(12)

mean_absolute_error(real, fcast1)

np.sqrt(mean_squared_error(real, fcast1))

mean_squared_error(real, fcast1)

mean_absolute_percentage_error(real, fcast1)

fcast1.plot(marker='.', color='blue', legend=True, label = 'Previsto')
X.plot(marker='.',  color='red', label = 'Real', legend = True)
plt.show()

"""#Holt tendencia amortecida"""

fit3 = Holt(X_base, damped_trend=True).fit(smoothing_level=0.8, smoothing_trend=0.2)
fcast3 = fit3.forecast(12)

mean_absolute_error(real, fcast3)

mean_squared_error(real, fcast3)

np.sqrt(mean_squared_error(real, fcast3))

mean_absolute_percentage_error(real, fcast3)

fcast3.plot(marker='.', color='blue', legend=True, label = 'Previsto')
X.plot(marker='.',  color='red', label = 'Real', legend = True)
plt.show()

"""#Holt Winters"""

fit1 = ExponentialSmoothing(X_base, seasonal_periods=12, trend='additive', seasonal='add', use_boxcox=True).fit()

forecast = fit1.forecast(12)

forecast

mean_absolute_error(real, forecast)

np.sqrt(mean_squared_error(real, forecast))

mean_squared_error(real, forecast)

mean_absolute_percentage_error(real, forecast)

fit1.forecast(12).plot(style='--', marker='o', color='black', legend=True, label = 'Previsto')
X.plot(marker='.',  color='red', label = 'Real', legend = True)
plt.show()