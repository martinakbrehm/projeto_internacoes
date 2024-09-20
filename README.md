# Previsão de Internações Hospitalares com Modelos de Séries Temporais

Este projeto implementa e compara diversos modelos de séries temporais para prever internações hospitalares. Entre os modelos utilizados estão: ARIMA, SARIMA, Holt-Winters, Holt com Tendência Amortecida, Holt, Suavização Exponencial Simples e LSTM (Long Short-Term Memory). Para otimizar o desempenho dos modelos, foi implementado um algoritmo de Grid Search para ajuste fino dos hiperparâmetros. A previsão para o ano de 2019 foi realizada com todos os modelos, e o LSTM foi utilizado especificamente para previsões no período do programa Previne Brasil.

## Introdução

A previsão de demanda hospitalar é uma questão crítica para o planejamento de recursos no setor da saúde. Este projeto visa comparar diferentes modelos de séries temporais para prever o número de internações hospitalares. Foram utilizadas diversas abordagens tradicionais e de aprendizado profundo, com foco em prever o comportamento futuro e ajudar no planejamento para programas como o Previne Brasil.

Além disso, foi implementada uma técnica de Grid Search para realizar o ajuste fino dos hiperparâmetros dos modelos, garantindo a escolha das melhores configurações para cada abordagem.

## Modelos utilizados

Os seguintes modelos foram implementados para a previsão de internações hospitalares:

ARIMA (Autoregressive Integrated Moving Average)

SARIMA (Seasonal ARIMA)

Holt-Winters (Suavização Exponencial com sazonalidade)

Holt com Tendência Amortecida

Holt

Suavização Exponencial Simples

LSTM (Long Short-Term Memory) usando Keras e TensorFlow

Cada um desses modelos foi ajustado aos dados históricos e utilizado para prever as internações hospitalares para o ano de 2019. Em seguida, o LSTM foi treinado para realizar previsões em períodos críticos, como os exigidos pelo programa Previne Brasil

## Requisitos

Para rodar o projeto, você precisará dos seguintes pacotes:

Python 3.x
Keras
TensorFlow
NumPy
Scikit-learn (para análise exploratória, Grid Search e pré-processamento dos dados)
Statsmodels (para modelos ARIMA, SARIMA, etc.)
Matplotlib/Seaborn (para visualizações)


Para instalar as dependências, utilize o seguinte comando:

pip install -r requirements.txt





