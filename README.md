# Predição de Internações por Condições Sensíveis à Atenção Primária no Brasil utilizando ARIMA e SARIMA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UFSC](https://img.shields.io/badge/UFSC-Universidade%20Federal%20de%20Santa%20Catarina-003366.svg)](https://ufsc.br/)
[![CBIS24](https://img.shields.io/badge/CBIS'24-XX%20Congresso%20Brasileiro%20de%20Inform%C3%A1tica%20em%20Sa%C3%BAde-orange.svg)](https://cbis.org.br/)

## Publicação

Este repositório contém o código-fonte do estudo científico:

> **Predição de Internações por Condições Sensíveis à Atenção Primária no Brasil utilizando ARIMA e SARIMA**
>
> *Martina Klippel Brehm, Luís Antonio Lourenço, Vinicius Faria Culmant Ramos e Júlia Gualdi Schlichting*
>
> Universidade Federal de Santa Catarina (UFSC)
>
> Publicado no **XX Congresso Brasileiro de Informática em Saúde — CBIS'24**, na modalidade Resumos Expandidos — Trabalhos de Iniciação Científica. Belo Horizonte, 08–11 de outubro de 2024.
>
> [Certificado de publicação](https://www.even3.com.br/documentos/imprimir?i=36651834.307944.124522.6.88254969173665062008&cc=50D0A581-ED27-4D79-90CC-8EB8A1D12D29)

## Resumo

Este projeto implementa um pipeline completo de **previsão de demanda hospitalar** utilizando modelos de séries temporais, com foco na projeção da taxa de internação hospitalar para municípios brasileiros classificados por porte populacional (IBGE). O estudo compara abordagens estatísticas clássicas e métodos de aprendizado profundo, incluindo previsões para o período do programa **Previne Brasil**.

## Modelos Implementados

| Família | Modelo | Descrição |
|---------|--------|-----------|
| **Suavização Exponencial** | SES | Suavização Exponencial Simples (α = 0,2) |
| | Holt | Suavização Exponencial Dupla (nível + tendência) |
| | Holt Amortecido | Tendência amortecida para horizontes longos |
| | Holt-Winters | Tripla suavização (nível + tendência + sazonalidade) |
| **Box-Jenkins** | ARIMA | Modelo autorregressivo integrado de média móvel |
| | SARIMA | ARIMA com componente sazonal (m=12) |
| **Deep Learning** | LSTM | Rede Long Short-Term Memory (4 camadas empilhadas) |

## Estrutura do Projeto

```
projeto_internacoes/
├── README.md                           # Documentação do projeto
├── requirements.txt                    # Dependências Python
├── setup.py                            # Configuração do pacote
├── .gitignore                          # Arquivos ignorados pelo Git
│
├── src/                                # Pacote Python principal
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                # Configurações centralizadas
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Carregamento de dados
│   │   └── preprocessing.py           # Pré-processamento (porte, taxa)
│   ├── features/
│   │   ├── __init__.py
│   │   └── time_series.py             # Estacionariedade, ADF, LSTM prep
│   ├── models/
│   │   ├── __init__.py
│   │   ├── statistical.py             # SES, Holt, Holt-Winters
│   │   ├── arima.py                   # ARIMA, SARIMA
│   │   └── lstm.py                    # Rede LSTM
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                 # MAE, RMSE, MAPE, Theil U₂
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                   # Gráficos padronizados
│
├── notebooks/                          # Notebooks organizados por etapa
│   ├── 01_preparacao_dados/
│   │   ├── 01_dados_novos.ipynb
│   │   └── 02_join_dados.ipynb
│   ├── 02_analise_exploratoria/
│   │   ├── 01_eda_grande_porte.ipynb
│   │   └── 02_eda_medio_porte.ipynb
│   ├── 03_modelos_parametricos/
│   │   ├── 01_arima_sarima_grande_porte.ipynb
│   │   └── 02_arima_sarima_medio_porte.ipynb
│   ├── 04_modelos_exponenciais/
│   │   ├── 01_exponenciais_grande_porte.ipynb
│   │   └── 02_exponenciais_medio_porte.ipynb
│   └── 05_lstm/
│       ├── 01_lstm_grande_porte_2019.ipynb
│       ├── 02_lstm_medio_porte_2019.ipynb
│       └── 03_previne_brasil/
│           └── 01_grande_porte_2010.ipynb
│
├── data/                               # Dados (não versionados)
│   ├── raw/                            # Dados brutos
│   ├── processed/                      # Dados processados
│   └── results/                        # Resultados e predições
│
└── legacy/                             # Notebooks originais (referência)
    ├── Análise exploratória/
    ├── Preparação dos dados/
    ├── Resultados Arima e Sarima/
    ├── Resultados LSTM/
    └── Resultados outros métodos/
```

## Metodologia

### 1. Coleta e Preparação dos Dados

- **Fonte**: DATASUS — Sistema de Informações Hospitalares (SIH/SUS)
- **Variável-alvo**: Taxa de internação por 1.000 habitantes
- **Granularidade**: Mensal, por município
- **Segmentação**: Porte municipal segundo classificação IBGE

### 2. Análise Exploratória

- Estatísticas descritivas da série temporal
- Decomposição sazonal aditiva (tendência + sazonalidade + resíduo)
- Teste ADF (Augmented Dickey-Fuller) para estacionariedade
- Diferenciação de primeira ordem
- Análise de autocorrelação (ACF/PACF)

### 3. Modelagem

#### Modelos Estatísticos
- Ajuste paramétrico com hiperparâmetros definidos para cada método
- Intervalos de confiança de 80% e 95%
- Seleção automática ARIMA/SARIMA via `auto_arima` (critério AIC)

#### Deep Learning (LSTM)
- Arquitetura empilhada: 800 → 300 → 100 → 200 unidades
- Dropout de 20% entre camadas
- Normalização MinMaxScaler [0, 1]
- Janela deslizante de 12 meses (lookback)

### 4. Avaliação

| Métrica | Descrição |
|---------|-----------|
| MAE | Erro Absoluto Médio |
| MSE | Erro Quadrático Médio |
| RMSE | Raiz do Erro Quadrático Médio |
| MAPE | Erro Percentual Absoluto Médio |
| Theil U₂ | Coeficiente U₂ de Theil (vs. naïve) |

## Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd projeto_internacoes

# Crie um ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt

# (Opcional) Instale o pacote em modo editável
pip install -e .
```

## Uso

### Execução Local

1. Coloque o dataset `dataset_internacoes_completo.csv` em `data/processed/`
2. Execute os notebooks na ordem numérica dentro de `notebooks/`

### Google Colab

Nos notebooks, substitua o carregamento por:
```python
from google.colab import drive
drive.mount('/content/drive')
dataset = loader.load_from_colab()
```

### Uso Programático

```python
from src.config.settings import ProjectConfig
from src.data import DataLoader, DataPreprocessor
from src.models import StatisticalModels
from src.evaluation import ModelEvaluator

config = ProjectConfig()
loader = DataLoader(config)
preprocessor = DataPreprocessor(config)

# Pipeline completo
dataset = loader.load_dataset()
series = preprocessor.pipeline_completo(dataset, porte_filtro="Grande Porte")

# Modelagem
models = StatisticalModels(config)
forecast = models.holt_winters(series['taxa_internacao'][:'2018-12-31'])

# Avaliação
evaluator = ModelEvaluator()
report = evaluator.evaluate(real, forecast, model_name="Holt-Winters")
print(report)
```

## Referências

- **Brehm, M.K.; Lourenço, L.A.; Ramos, V.F.C.; Schlichting, J.G.** (2024). Predição de Internações por Condições Sensíveis à Atenção Primária no Brasil utilizando ARIMA e SARIMA. In: *XX Congresso Brasileiro de Informática em Saúde — CBIS'24*. Belo Horizonte.
- **Box, G.E.P. & Jenkins, G.M.** (1976). *Time Series Analysis: Forecasting and Control*.
- **Holt, C.C.** (1957). Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages.
- **Hochreiter, S. & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- **Hyndman, R.J. & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice*. 3rd ed. OTexts.
- **Theil, H.** (1966). *Applied Economic Forecasting*. North-Holland.





