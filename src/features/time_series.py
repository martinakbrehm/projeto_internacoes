"""
Análise de Estacionariedade e Engenharia de Features
=====================================================

Módulo que implementa testes estatísticos de estacionariedade e
transformações de séries temporais, componentes essenciais da
etapa de análise exploratória e preparação para modelagem.

Testes implementados:
    - ADF (Augmented Dickey-Fuller)
    - Ljung-Box (autocorrelação dos resíduos)

Transformações:
    - Diferenciação de primeira ordem
    - Normalização MinMaxScaler para LSTM

Referências:
    - Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators
      for Autoregressive Time Series with a Unit Root.
    - Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

from src.config.settings import LSTMConfig, ProjectConfig

logger = logging.getLogger(__name__)


@dataclass
class ADFResult:
    """Resultado estruturado do teste Augmented Dickey-Fuller.

    Attributes
    ----------
    statistic : float
        Estatística ADF calculada.
    p_value : float
        Valor-p do teste.
    critical_values : dict
        Valores críticos para cada nível de significância.
    is_stationary : bool
        True se a série é estacionária ao nível de 5%.
    """
    statistic: float
    p_value: float
    critical_values: dict
    is_stationary: bool

    def __str__(self) -> str:
        lines = [
            f"Teste ADF — Augmented Dickey-Fuller",
            f"{'─'*40}",
            f"  ADF Estatística : {self.statistic:.6f}",
            f"  Valor de P      : {self.p_value:.6f}",
            f"  Estacionária    : {'Sim' if self.is_stationary else 'Não'} (α=0.05)",
            f"  Valores Críticos:",
        ]
        for key, value in self.critical_values.items():
            lines.append(f"    {key}: {value:.3f}")
        return "\n".join(lines)


class StationarityAnalyzer:
    """Analisador de estacionariedade para séries temporais.

    Encapsula os testes estatísticos e transformações necessárias
    para preparar a série para ajuste de modelos ARIMA/SARIMA.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações do projeto.
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()

    # ------------------------------------------------------------------
    # Teste ADF
    # ------------------------------------------------------------------
    def adf_test(
        self,
        series: pd.Series,
        significance_level: float = 0.05,
    ) -> ADFResult:
        """Executa o teste Augmented Dickey-Fuller.

        Hipóteses:
            H₀: A série possui raiz unitária (não estacionária).
            H₁: A série não possui raiz unitária (estacionária).

        Parameters
        ----------
        series : pd.Series
            Série temporal para teste.
        significance_level : float
            Nível de significância para conclusão.

        Returns
        -------
        ADFResult
            Resultado estruturado do teste.
        """
        result = adfuller(series)

        adf_result = ADFResult(
            statistic=result[0],
            p_value=result[1],
            critical_values=result[4],
            is_stationary=result[1] < significance_level,
        )

        logger.info("Teste ADF: estatística=%.4f, p-valor=%.4f", result[0], result[1])
        return adf_result

    # ------------------------------------------------------------------
    # Diferenciação
    # ------------------------------------------------------------------
    @staticmethod
    def diferenciar(series: pd.Series, order: int = 1) -> pd.Series:
        """Aplica diferenciação à série temporal.

        Parameters
        ----------
        series : pd.Series
            Série original.
        order : int
            Ordem de diferenciação.

        Returns
        -------
        pd.Series
            Série diferenciada (sem NaNs).
        """
        diff = series.diff(periods=order).dropna()
        logger.info(
            "Diferenciação de ordem %d aplicada: %d → %d observações.",
            order,
            len(series),
            len(diff),
        )
        return diff

    # ------------------------------------------------------------------
    # Teste Ljung-Box
    # ------------------------------------------------------------------
    @staticmethod
    def ljung_box_test(
        series: pd.Series,
        lags: int = 24,
    ) -> pd.DataFrame:
        """Executa o teste de Ljung-Box para autocorrelação.

        Hipóteses:
            H₀: Os resíduos são independentes (sem autocorrelação).
            H₁: Os resíduos apresentam autocorrelação significativa.

        Parameters
        ----------
        series : pd.Series
            Série ou resíduos para teste.
        lags : int
            Número de lags para avaliação.

        Returns
        -------
        pd.DataFrame
            Resultados do teste (estatística e p-valor).
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(series, lags=[lags])
        logger.info("Teste Ljung-Box executado com %d lags.", lags)
        return result

    # ------------------------------------------------------------------
    # Preparação de Dados para LSTM
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_lstm_data(
        series: pd.DataFrame,
        lookback: int = 12,
        train_size: Optional[int] = None,
        scaler_range: Tuple[float, float] = (0, 1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepara os dados para entrada no modelo LSTM.

        Etapas:
            1. Normalização via MinMaxScaler.
            2. Criação de janelas deslizantes (lookback).
            3. Divisão treino/teste.

        Parameters
        ----------
        series : pd.DataFrame
            Série temporal (única coluna).
        lookback : int
            Tamanho da janela de observação.
        train_size : int, optional
            Número de amostras para treino. Se None, utiliza tudo.
        scaler_range : tuple
            Intervalo de normalização.

        Returns
        -------
        tuple
            (x_train, y_train, x_test, y_test, scaler)
        """
        # Normalização
        scaler = MinMaxScaler(feature_range=scaler_range)
        data = scaler.fit_transform(series.values)

        # Janelas deslizantes
        x, y = [], []
        for i in range(lookback, len(data)):
            x.append(data[i - lookback:i, 0])
            y.append(data[i, 0])

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Divisão treino/teste
        if train_size is not None:
            x_train, y_train = x[:train_size], y[:train_size]
            x_test, y_test = x[train_size:], y[train_size:]
        else:
            x_train, y_train = x, y
            x_test, y_test = np.array([]), np.array([])

        logger.info(
            "Dados LSTM preparados: treino=%d, teste=%d, lookback=%d",
            len(x_train),
            len(x_test),
            lookback,
        )
        return x_train, y_train, x_test, y_test, scaler
