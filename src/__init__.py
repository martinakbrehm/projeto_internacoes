"""
Previsão de Internações Hospitalares — Pacote Principal
========================================================

Módulo central do projeto de previsão de internações hospitalares utilizando
modelos de séries temporais (ARIMA, SARIMA, Holt-Winters, LSTM).

Estrutura do pacote:
    - config: Configurações centralizadas do projeto
    - data: Carregamento e pré-processamento de dados
    - features: Engenharia de features para séries temporais
    - models: Implementações dos modelos preditivos
    - evaluation: Métricas de avaliação de desempenho
    - visualization: Funções padronizadas de visualização

Autora: Martina
"""

__version__ = "1.0.0"
__author__ = "Martina"

from src.config.settings import ProjectConfig
