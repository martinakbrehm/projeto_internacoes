"""
Modelos de Suavização Exponencial
==================================

Implementação dos modelos de suavização exponencial utilizados para
previsão de séries temporais de internações hospitalares:

    - SES (Suavização Exponencial Simples)
    - Holt (Suavização Exponencial Dupla)
    - Holt com Tendência Amortecida
    - Holt-Winters (Suavização Exponencial Tripla)

Todos os modelos preservam **exatamente** os mesmos hiperparâmetros
dos notebooks originais para garantir reprodutibilidade dos resultados.

Referências:
    - Holt, C.C. (1957). Forecasting Seasonals and Trends by
      Exponentially Weighted Moving Averages.
    - Winters, P.R. (1960). Forecasting Sales by Exponentially
      Weighted Moving Averages.
    - Hyndman, R.J. et al. (2008). Forecasting with Exponential
      Smoothing: The State Space Approach. Springer.
"""

import logging
from typing import Optional, Tuple

import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing

from src.config.settings import (
    HoltConfig,
    HoltDampedConfig,
    HoltWintersConfig,
    ProjectConfig,
    SESConfig,
)

logger = logging.getLogger(__name__)


class StatisticalModels:
    """Interface unificada para modelos de suavização exponencial.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações dos hiperparâmetros.

    Examples
    --------
    >>> models = StatisticalModels()
    >>> forecast = models.simple_exponential_smoothing(train_data)
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()

    # ------------------------------------------------------------------
    # Suavização Exponencial Simples (SES)
    # ------------------------------------------------------------------
    def simple_exponential_smoothing(
        self,
        train: pd.Series,
        forecast_horizon: Optional[int] = None,
    ) -> pd.Series:
        """Ajusta e prevê com Suavização Exponencial Simples.

        Modelo adequado para séries sem tendência nem sazonalidade.
        Parâmetro α = 0.2 (20% de peso para observações recentes).

        Parameters
        ----------
        train : pd.Series
            Série de treino.
        forecast_horizon : int, optional
            Horizonte de previsão em meses.

        Returns
        -------
        pd.Series
            Previsões para o horizonte especificado.
        """
        cfg: SESConfig = self.config.ses
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Ajustando SES (α=%.2f, optimized=%s, horizon=%d)...",
            cfg.SMOOTHING_LEVEL,
            cfg.OPTIMIZED,
            horizon,
        )

        fit = SimpleExpSmoothing(train).fit(
            smoothing_level=cfg.SMOOTHING_LEVEL,
            optimized=cfg.OPTIMIZED,
        )
        forecast = fit.forecast(horizon)

        logger.info("SES ajustado com sucesso.")
        return forecast

    # ------------------------------------------------------------------
    # Holt (Suavização Exponencial Dupla)
    # ------------------------------------------------------------------
    def holt(
        self,
        train: pd.Series,
        forecast_horizon: Optional[int] = None,
    ) -> pd.Series:
        """Ajusta e prevê com o método de Holt.

        Extensão da SES que adiciona modelagem de tendência linear.

        Parameters
        ----------
        train : pd.Series
            Série de treino.
        forecast_horizon : int, optional
            Horizonte de previsão.

        Returns
        -------
        pd.Series
            Previsões.
        """
        cfg: HoltConfig = self.config.holt
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Ajustando Holt (α=%.2f, β=%.2f, optimized=%s)...",
            cfg.SMOOTHING_LEVEL,
            cfg.SMOOTHING_TREND,
            cfg.OPTIMIZED,
        )

        fit = Holt(train).fit(
            smoothing_level=cfg.SMOOTHING_LEVEL,
            smoothing_trend=cfg.SMOOTHING_TREND,
            optimized=cfg.OPTIMIZED,
        )
        forecast = fit.forecast(horizon)

        logger.info("Holt ajustado com sucesso.")
        return forecast

    # ------------------------------------------------------------------
    # Holt com Tendência Amortecida
    # ------------------------------------------------------------------
    def holt_damped(
        self,
        train: pd.Series,
        forecast_horizon: Optional[int] = None,
    ) -> pd.Series:
        """Ajusta e prevê com Holt com Tendência Amortecida.

        A tendência amortecida evita projeções lineares irreais para
        horizontes mais longos, "suavizando" a tendência ao longo do tempo.

        Parameters
        ----------
        train : pd.Series
            Série de treino.
        forecast_horizon : int, optional
            Horizonte de previsão.

        Returns
        -------
        pd.Series
            Previsões.
        """
        cfg: HoltDampedConfig = self.config.holt_damped
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Ajustando Holt Amortecido (α=%.2f, β=%.2f, damped=%s)...",
            cfg.SMOOTHING_LEVEL,
            cfg.SMOOTHING_TREND,
            cfg.DAMPED_TREND,
        )

        fit = Holt(train, damped_trend=cfg.DAMPED_TREND).fit(
            smoothing_level=cfg.SMOOTHING_LEVEL,
            smoothing_trend=cfg.SMOOTHING_TREND,
        )
        forecast = fit.forecast(horizon)

        logger.info("Holt Amortecido ajustado com sucesso.")
        return forecast

    # ------------------------------------------------------------------
    # Holt-Winters (Suavização Exponencial Tripla)
    # ------------------------------------------------------------------
    def holt_winters(
        self,
        train: pd.Series,
        forecast_horizon: Optional[int] = None,
    ) -> pd.Series:
        """Ajusta e prevê com Holt-Winters (Exponential Smoothing tripla).

        Extensão que modela nível, tendência e sazonalidade. Utiliza
        transformação Box-Cox conforme configuração original.

        Parameters
        ----------
        train : pd.Series
            Série de treino.
        forecast_horizon : int, optional
            Horizonte de previsão.

        Returns
        -------
        pd.Series
            Previsões.
        """
        cfg: HoltWintersConfig = self.config.holt_winters
        horizon = forecast_horizon or cfg.FORECAST_HORIZON

        logger.info(
            "Ajustando Holt-Winters (periodo=%d, tendência='%s', "
            "sazonalidade='%s', box-cox=%s)...",
            cfg.SEASONAL_PERIODS,
            cfg.TREND,
            cfg.SEASONAL,
            cfg.USE_BOXCOX,
        )

        fit = ExponentialSmoothing(
            train,
            seasonal_periods=cfg.SEASONAL_PERIODS,
            trend=cfg.TREND,
            seasonal=cfg.SEASONAL,
            use_boxcox=cfg.USE_BOXCOX,
        ).fit()
        forecast = fit.forecast(horizon)

        logger.info("Holt-Winters ajustado com sucesso.")
        return forecast
