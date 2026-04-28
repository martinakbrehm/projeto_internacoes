"""
Pré-processamento de Dados
============================

Módulo que encapsula todas as transformações aplicadas ao dataset bruto
de internações hospitalares, incluindo:

- Classificação de municípios por porte populacional (IBGE)
- Cálculo da taxa de internação por 1.000 habitantes
- Filtragem por porte municipal
- Construção de séries temporais mensais

A lógica é extraída dos notebooks originais e parametrizada via
``ProjectConfig`` para garantir reprodutibilidade.
"""

import logging
from typing import Optional

import pandas as pd

from src.config.settings import PorteConfig, ProjectConfig, TimeSeriesConfig

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Pipeline de pré-processamento dos dados de internações.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações do projeto.

    Attributes
    ----------
    porte_config : PorteConfig
        Faixas de classificação por porte.
    ts_config : TimeSeriesConfig
        Parâmetros de séries temporais.
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()
        self.porte_config: PorteConfig = self.config.porte
        self.ts_config: TimeSeriesConfig = self.config.time_series

    # ------------------------------------------------------------------
    # Classificação de Porte
    # ------------------------------------------------------------------
    def classificar_porte(self, populacao: int) -> str:
        """Classifica um município segundo a faixa populacional (IBGE).

        Parameters
        ----------
        populacao : int
            População residente do município.

        Returns
        -------
        str
            Rótulo do porte municipal.

        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> preprocessor.classificar_porte(150_000)
        'Grande Porte'
        """
        cfg = self.porte_config
        if populacao <= cfg.PEQUENO_PORTE_I:
            return cfg.LABELS["pequeno_i"]
        elif populacao <= cfg.PEQUENO_PORTE_II:
            return cfg.LABELS["pequeno_ii"]
        elif populacao <= cfg.MEDIO_PORTE:
            return cfg.LABELS["medio"]
        elif populacao <= cfg.GRANDE_PORTE:
            return cfg.LABELS["grande"]
        else:
            return cfg.LABELS["metropole"]

    # ------------------------------------------------------------------
    # Taxa de Internação
    # ------------------------------------------------------------------
    def calcular_taxa_internacao(self, row: pd.Series) -> float:
        """Calcula a taxa de internação por 1.000 habitantes.

        Fórmula:
            taxa = (Qtd. internações × 1.000) / população

        Parameters
        ----------
        row : pd.Series
            Linha do DataFrame contendo as colunas de internações e população.

        Returns
        -------
        float
            Taxa de internação calculada.
        """
        ts = self.ts_config
        return (
            row[ts.INTERNACOES_COLUMN] * ts.TAXA_MULTIPLIER / row[ts.POPULACAO_COLUMN]
        )

    # ------------------------------------------------------------------
    # Pipeline Completo
    # ------------------------------------------------------------------
    def processar(
        self,
        dataset: pd.DataFrame,
        porte_filtro: Optional[str] = None,
        target_col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Executa o pipeline completo de pré-processamento.

        Etapas:
            1. Calcula a taxa de internação.
            2. Classifica cada município por porte.
            3. (Opcional) Filtra pelo porte desejado.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame bruto com as colunas originais.
        porte_filtro : str, optional
            Porte para filtragem (e.g., 'Grande Porte', 'Médio Porte').
        target_col_name : str, optional
            Nome da coluna-alvo. Se None, utiliza o valor do config.

        Returns
        -------
        pd.DataFrame
            DataFrame pré-processado.
        """
        ts = self.ts_config
        col_name = target_col_name or ts.TARGET_COLUMN

        logger.info("Iniciando pipeline de pré-processamento...")

        # 1. Taxa de internação
        dataset = dataset.copy()
        dataset[col_name] = dataset.apply(self.calcular_taxa_internacao, axis=1)
        logger.info("Taxa de internação calculada (coluna: '%s').", col_name)

        # 2. Classificação de porte
        dataset["Porte"] = dataset[ts.POPULACAO_COLUMN].apply(self.classificar_porte)
        logger.info("Porte municipal classificado.")

        # 3. Filtragem (se solicitado)
        if porte_filtro is not None:
            n_antes = len(dataset)
            dataset = dataset[dataset["Porte"] == porte_filtro].copy()
            logger.info(
                "Filtro por '%s': %d → %d registros.",
                porte_filtro,
                n_antes,
                len(dataset),
            )

        return dataset

    # ------------------------------------------------------------------
    # Construção de Série Temporal
    # ------------------------------------------------------------------
    def construir_serie_temporal(
        self,
        dataset: pd.DataFrame,
        data_inicio: Optional[str] = None,
        data_fim: Optional[str] = None,
        target_col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Constrói série temporal mensal a partir do dataset processado.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame já pré-processado (com taxa de internação calculada).
        data_inicio : str, optional
            Data de início para recorte da série (formato 'YYYY-MM-DD').
        data_fim : str, optional
            Data final para recorte da série.
        target_col_name : str, optional
            Nome da coluna-alvo. Se None, utiliza o valor do config.

        Returns
        -------
        pd.DataFrame
            Série temporal com índice DatetimeIndex e frequência mensal.
        """
        ts = self.ts_config
        col_name = target_col_name or ts.TARGET_COLUMN

        time_series = dataset[[ts.DATE_COLUMN, col_name]].copy()
        time_series[ts.DATE_COLUMN] = pd.to_datetime(time_series[ts.DATE_COLUMN])
        time_series = time_series.set_index(ts.DATE_COLUMN).resample(ts.FREQ).mean()

        if data_inicio or data_fim:
            time_series = time_series[data_inicio:data_fim]
            logger.info(
                "Série temporal recortada: [%s, %s] → %d observações.",
                data_inicio or "início",
                data_fim or "fim",
                len(time_series),
            )

        logger.info(
            "Série temporal construída: %d observações mensais.",
            len(time_series),
        )
        return time_series

    # ------------------------------------------------------------------
    # Pipeline Integrado (processar + série)
    # ------------------------------------------------------------------
    def pipeline_completo(
        self,
        dataset: pd.DataFrame,
        porte_filtro: str,
        data_inicio: Optional[str] = None,
        data_fim: Optional[str] = None,
        target_col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Executa pré-processamento + construção de série temporal.

        Conveniência que combina ``processar()`` e ``construir_serie_temporal()``.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame bruto original.
        porte_filtro : str
            Porte para filtragem.
        data_inicio : str, optional
            Início do recorte temporal.
        data_fim : str, optional
            Fim do recorte temporal.
        target_col_name : str, optional
            Nome da coluna-alvo.

        Returns
        -------
        pd.DataFrame
            Série temporal filtrada e reamostrada.
        """
        df = self.processar(dataset, porte_filtro=porte_filtro, target_col_name=target_col_name)
        return self.construir_serie_temporal(
            df, data_inicio=data_inicio, data_fim=data_fim, target_col_name=target_col_name
        )
