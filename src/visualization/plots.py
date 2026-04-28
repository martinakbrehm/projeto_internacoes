"""
Visualização de Séries Temporais
=================================

Módulo que centraliza todas as funções de plotagem utilizadas nos notebooks,
garantindo padronização visual e reprodutibilidade dos gráficos.

As funções preservam **exatamente** os mesmos parâmetros visuais dos notebooks
originais (cores, marcadores, intervalos de confiança) para manter a
consistência dos resultados publicados.
"""

import logging
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.pylab import rcParams

from src.config.settings import ConfidenceIntervalConfig, PlotConfig, ProjectConfig

logger = logging.getLogger(__name__)


class TimeSeriesPlotter:
    """Classe para geração padronizada de gráficos de séries temporais.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações do projeto (utiliza PlotConfig e ConfidenceIntervalConfig).

    Examples
    --------
    >>> plotter = TimeSeriesPlotter()
    >>> plotter.plot_series(time_series, title="Taxa de Internações")
    >>> plotter.plot_forecast_vs_real(real, previsto, title="SARIMA")
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()
        self.plot_cfg: PlotConfig = self.config.plot
        self.ci_cfg: ConfidenceIntervalConfig = self.config.ci
        self._apply_global_style()

    def _apply_global_style(self) -> None:
        """Aplica configurações globais de estilo do matplotlib."""
        rcParams["figure.figsize"] = self.plot_cfg.FIGSIZE

    # ------------------------------------------------------------------
    # Gráficos Básicos de Série Temporal
    # ------------------------------------------------------------------
    def plot_series(
        self,
        series: pd.DataFrame,
        title: str = "Taxa de Internações",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plota série temporal simples.

        Parameters
        ----------
        series : pd.DataFrame or pd.Series
            Série temporal com índice DatetimeIndex.
        title : str
            Título do gráfico.
        xlabel : str, optional
            Rótulo do eixo X.
        ylabel : str, optional
            Rótulo do eixo Y.
        show : bool
            Se True, exibe o gráfico imediatamente.

        Returns
        -------
        matplotlib.figure.Figure
            Figura gerada.
        """
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title(title)
        ax.set_xlabel(xlabel or self.plot_cfg.XLABEL)
        ax.set_ylabel(ylabel or self.plot_cfg.YLABEL)
        if show:
            plt.show()
        return fig

    def plot_histogram(
        self,
        series: pd.DataFrame,
        title: str = "Taxa de Internações",
        show: bool = True,
    ) -> plt.Figure:
        """Plota histograma da distribuição da série.

        Parameters
        ----------
        series : pd.DataFrame or pd.Series
            Dados para o histograma.
        title : str
            Título do gráfico.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        series.hist()
        plt.title(title)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Gráficos de Previsão
    # ------------------------------------------------------------------
    def plot_forecast_vs_real(
        self,
        real: Union[pd.Series, np.ndarray],
        forecast: Union[pd.Series, np.ndarray],
        title: Optional[str] = None,
        marker: str = "",
        show: bool = True,
    ) -> plt.Figure:
        """Plota série real versus previsão do modelo.

        Preserva exatamente as mesmas cores (vermelho/azul) e estilo
        dos notebooks originais.

        Parameters
        ----------
        real : pd.Series or np.ndarray
            Valores reais observados.
        forecast : pd.Series or np.ndarray
            Valores previstos.
        title : str, optional
            Título do gráfico.
        marker : str
            Marcador dos pontos.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()

        if isinstance(forecast, pd.Series):
            forecast.plot(
                marker=marker,
                color=self.plot_cfg.FORECAST_COLOR,
                legend=True,
                label=self.plot_cfg.FORECAST_LABEL,
            )
        else:
            plt.plot(forecast, color=self.plot_cfg.FORECAST_COLOR, label=self.plot_cfg.FORECAST_LABEL)

        if isinstance(real, pd.Series):
            real.plot(
                marker=marker,
                color=self.plot_cfg.REAL_COLOR,
                label=self.plot_cfg.REAL_LABEL,
                legend=True,
            )
        else:
            plt.plot(real, color=self.plot_cfg.REAL_COLOR, label=self.plot_cfg.REAL_LABEL)

        if title:
            plt.title(title)
        plt.legend()
        if show:
            plt.show()
        return fig

    def plot_forecast_with_ci(
        self,
        real: pd.Series,
        forecast: pd.Series,
        historical_std: float,
        title: str = "Previsão com Intervalo de Confiança",
        show: bool = True,
    ) -> plt.Figure:
        """Plota previsão com intervalos de confiança de 80% e 95%.

        Reproduz exatamente a lógica de intervalos de confiança dos
        notebooks originais, utilizando a distribuição normal padrão
        e o desvio padrão da série histórica.

        Parameters
        ----------
        real : pd.Series
            Série completa (treino + teste).
        forecast : pd.Series
            Previsões com DatetimeIndex.
        historical_std : float
            Desvio padrão da série de treino.
        title : str
            Título do gráfico.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Valores críticos (z-score)
        z_95 = stats.norm.ppf(1 - self.ci_cfg.ALPHA_95 / 2)
        z_80 = stats.norm.ppf(1 - self.ci_cfg.ALPHA_80 / 2)

        # Limites dos intervalos
        lower_95 = forecast - z_95 * historical_std
        upper_95 = forecast + z_95 * historical_std
        lower_80 = forecast - z_80 * historical_std
        upper_80 = forecast + z_80 * historical_std

        fig = plt.figure()

        forecast.plot(
            marker="",
            color=self.plot_cfg.FORECAST_COLOR,
            legend=True,
            label=self.plot_cfg.FORECAST_LABEL,
        )
        real.plot(
            marker="",
            color=self.plot_cfg.REAL_COLOR,
            label=self.plot_cfg.REAL_LABEL,
            legend=True,
        )

        plt.fill_between(
            forecast.index,
            lower_95,
            upper_95,
            color=self.plot_cfg.CI_95_COLOR,
            alpha=self.plot_cfg.CI_95_ALPHA,
            label=self.plot_cfg.CI_95_LABEL,
        )
        plt.fill_between(
            forecast.index,
            lower_80,
            upper_80,
            color=self.plot_cfg.CI_80_COLOR,
            alpha=self.plot_cfg.CI_80_ALPHA,
            label=self.plot_cfg.CI_80_LABEL,
        )

        plt.title(title)
        plt.ylabel(self.plot_cfg.YLABEL)
        plt.xlabel(self.plot_cfg.XLABEL)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Gráficos de Decomposição e Diagnóstico
    # ------------------------------------------------------------------
    def plot_decomposition(
        self,
        series: pd.DataFrame,
        model: str = "additive",
        show: bool = True,
    ) -> plt.Figure:
        """Plota decomposição sazonal da série temporal.

        Parameters
        ----------
        series : pd.DataFrame
            Série temporal para decomposição.
        model : str
            Tipo de decomposição ('additive' ou 'multiplicative').
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(series, model=model)
        fig = result.plot()
        if show:
            plt.show()
        return fig

    def plot_acf_pacf(
        self,
        series: pd.Series,
        lags: Optional[int] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Figure]:
        """Plota funções de autocorrelação (ACF) e autocorrelação parcial (PACF).

        Parameters
        ----------
        series : pd.Series
            Série temporal.
        lags : int, optional
            Número de lags a exibir.
        show : bool
            Se True, exibe os gráficos.

        Returns
        -------
        tuple of (Figure, Figure)
            Figuras ACF e PACF.
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig_acf = plt.figure()
        plot_acf(series, ax=fig_acf.gca(), lags=lags)
        if show:
            plt.show()

        fig_pacf = plt.figure()
        plot_pacf(series, ax=fig_pacf.gca(), method="ywm", lags=lags)
        if show:
            plt.show()

        return fig_acf, fig_pacf

    def plot_stationarity_diff(
        self,
        series_diff: pd.Series,
        show: bool = True,
    ) -> plt.Figure:
        """Plota série após diferenciação.

        Parameters
        ----------
        series_diff : pd.Series
            Série diferenciada.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        series_diff.plot()
        plt.xlabel(self.plot_cfg.XLABEL)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Gráficos LSTM
    # ------------------------------------------------------------------
    def plot_lstm_predictions(
        self,
        real: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray],
        index: Optional[pd.DatetimeIndex] = None,
        xlabel: str = "Mês",
        title: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plota previsões do modelo LSTM.

        Preserva o mesmo estilo visual dos notebooks LSTM originais.

        Parameters
        ----------
        real : array-like
            Valores reais.
        predictions : array-like
            Valores previstos pela LSTM.
        index : DatetimeIndex, optional
            Índice temporal para o eixo X.
        xlabel : str
            Rótulo do eixo X.
        title : str, optional
            Título do gráfico.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        plt.plot(real, color=self.plot_cfg.REAL_COLOR, label=self.plot_cfg.REAL_LABEL)
        plt.plot(predictions, color=self.plot_cfg.FORECAST_COLOR, label=self.plot_cfg.FORECAST_LABEL)

        if index is not None:
            plt.xticks(ticks=range(len(index)), labels=index.strftime("%b"))

        plt.xlabel(xlabel)
        if title:
            plt.title(title)
        plt.legend()
        if show:
            plt.show()
        return fig

    def plot_lstm_full_series(
        self,
        time_series: pd.DataFrame,
        predictions: pd.Series,
        xlabel: str = "Ano",
        title: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plota série completa com sobreposição das previsões LSTM.

        Parameters
        ----------
        time_series : pd.DataFrame
            Série temporal completa.
        predictions : pd.Series
            Previsões com DatetimeIndex.
        xlabel : str
            Rótulo do eixo X.
        title : str, optional
            Título do gráfico.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        plt.plot(
            predictions,
            color=self.plot_cfg.FORECAST_COLOR,
            label=self.plot_cfg.FORECAST_LABEL,
        )
        plt.plot(
            time_series,
            color=self.plot_cfg.REAL_COLOR,
            label=self.plot_cfg.REAL_LABEL,
        )
        plt.xlabel(xlabel)
        if title:
            plt.title(title)
        plt.legend()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Gráfico de Holt-Winters (estilo diferente)
    # ------------------------------------------------------------------
    def plot_holtwinters_forecast(
        self,
        real: pd.Series,
        forecast: pd.Series,
        show: bool = True,
    ) -> plt.Figure:
        """Plota previsão Holt-Winters com estilo original (marcadores e tracejado).

        Parameters
        ----------
        real : pd.Series
            Série real.
        forecast : pd.Series
            Previsão.
        show : bool
            Se True, exibe o gráfico.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure()
        forecast.plot(
            style="--",
            marker="o",
            color="black",
            legend=True,
            label=self.plot_cfg.FORECAST_LABEL,
        )
        real.plot(
            marker=".",
            color=self.plot_cfg.REAL_COLOR,
            label=self.plot_cfg.REAL_LABEL,
            legend=True,
        )
        if show:
            plt.show()
        return fig
