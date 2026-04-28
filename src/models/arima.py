"""
Modelos ARIMA e SARIMA
=======================

Implementação dos modelos autorregressivos integrados de média móvel,
com e sem componente sazonal.

    - ARIMA: Captura dependências autorregressivas e de média móvel.
    - SARIMA: Extensão sazonal do ARIMA para séries com padrões periódicos.

A seleção automática de hiperparâmetros utiliza ``auto_arima`` do pacote
``pmdarima``, que implementa busca stepwise baseada no critério AIC.

Referências:
    - Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis:
      Forecasting and Control.
    - Hyndman, R.J. & Khandakar, Y. (2008). Automatic Time Series
      Forecasting: The forecast Package for R.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson

from src.config.settings import ARIMAConfig, ProjectConfig, SARIMAConfig

logger = logging.getLogger(__name__)


class ARIMAModels:
    """Interface para ajuste e previsão com modelos ARIMA e SARIMA.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações com hiperparâmetros de busca.

    Examples
    --------
    >>> arima = ARIMAModels()
    >>> forecast, model = arima.fit_sarima(train_series)
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()
        self._sarima_model = None
        self._arima_model = None

    # ------------------------------------------------------------------
    # SARIMA
    # ------------------------------------------------------------------
    def fit_sarima(
        self,
        series: pd.Series,
        train: Optional[pd.Series] = None,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[pd.Series, object]:
        """Ajusta modelo SARIMA via ``auto_arima`` e gera previsões.

        Utiliza busca stepwise para selecionar a melhor combinação
        (p, d, q)(P, D, Q, m) com base no critério AIC.

        Parameters
        ----------
        series : pd.Series
            Série completa para seleção do modelo.
        train : pd.Series, optional
            Subconjunto de treino para ajuste final. Se None, usa ``series``.
        forecast_horizon : int, optional
            Horizonte de previsão.

        Returns
        -------
        tuple
            (previsões, modelo_ajustado)
        """
        from pmdarima.arima import auto_arima

        cfg: SARIMAConfig = self.config.sarima
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Selecionando modelo SARIMA via auto_arima "
            "(p=[%d,%d], q=[%d,%d], m=%d, D=%d)...",
            cfg.START_P,
            cfg.MAX_P,
            cfg.START_Q,
            cfg.MAX_Q,
            cfg.M,
            cfg.SEASONAL_D,
        )

        self._sarima_model = auto_arima(
            series,
            start_p=cfg.START_P,
            start_q=cfg.START_Q,
            max_p=cfg.MAX_P,
            max_q=cfg.MAX_Q,
            m=cfg.M,
            start_P=cfg.START_SEASONAL_P,
            seasonal=True,
            d=cfg.D,
            D=cfg.SEASONAL_D,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        logger.info("SARIMA selecionado — AIC: %.2f", self._sarima_model.aic())

        # Ajuste final com dados de treino
        fit_data = train if train is not None else series
        self._sarima_model.fit(fit_data)

        forecast = self._sarima_model.predict(n_periods=horizon)
        logger.info("Previsão SARIMA gerada para %d períodos.", horizon)

        return forecast, self._sarima_model

    # ------------------------------------------------------------------
    # ARIMA (não sazonal)
    # ------------------------------------------------------------------
    def fit_arima(
        self,
        series: pd.Series,
        train: Optional[pd.Series] = None,
        forecast_horizon: Optional[int] = None,
    ) -> Tuple[pd.Series, object]:
        """Ajusta modelo ARIMA (sem sazonalidade) via ``auto_arima``.

        Parameters
        ----------
        series : pd.Series
            Série completa para seleção do modelo.
        train : pd.Series, optional
            Subconjunto de treino. Se None, usa ``series``.
        forecast_horizon : int, optional
            Horizonte de previsão.

        Returns
        -------
        tuple
            (previsões, modelo_ajustado)
        """
        from pmdarima.arima import auto_arima

        cfg: ARIMAConfig = self.config.arima
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Selecionando modelo ARIMA via auto_arima "
            "(p=[%d,%d], q=[%d,%d])...",
            cfg.START_P,
            cfg.MAX_P,
            cfg.START_Q,
            cfg.MAX_Q,
        )

        self._arima_model = auto_arima(
            series,
            start_p=cfg.START_P,
            start_q=cfg.START_Q,
            max_p=cfg.MAX_P,
            max_q=cfg.MAX_Q,
            seasonal=False,
            d=cfg.D,
            D=cfg.D,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        # Ajuste final
        fit_data = train if train is not None else series
        self._arima_model.fit(fit_data)

        forecast = self._arima_model.predict(n_periods=horizon)
        logger.info("Previsão ARIMA gerada para %d períodos.", horizon)

        return forecast, self._arima_model

    # ------------------------------------------------------------------
    # Diagnóstico de Resíduos
    # ------------------------------------------------------------------
    @staticmethod
    def durbin_watson_test(model) -> float:
        """Executa o teste de Durbin-Watson nos resíduos do modelo.

        O teste verifica a presença de autocorrelação nos resíduos.
        Valores próximos de 2 indicam ausência de autocorrelação.

        Parameters
        ----------
        model : fitted ARIMA model
            Modelo ARIMA/SARIMA ajustado.

        Returns
        -------
        float
            Estatística de Durbin-Watson.
        """
        model_fit = model.fit(model.data)
        dw = durbin_watson(model_fit.resid())
        logger.info("Durbin-Watson: %.4f", dw)
        return dw
