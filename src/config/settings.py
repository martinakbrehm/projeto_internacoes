"""
Configurações Centralizadas do Projeto
=======================================

Módulo responsável por concentrar todos os parâmetros, caminhos e constantes
utilizados ao longo do pipeline de análise e modelagem. A centralização de
configurações segue boas práticas de engenharia de software, garantindo
reprodutibilidade e facilitando ajustes experimentais.

Referências:
    - IBGE — Classificação de municípios por porte populacional
    - Programa Previne Brasil — Ministério da Saúde
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classificação de Porte Municipal (IBGE)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PorteConfig:
    """Faixas populacionais para classificação de porte municipal.

    Baseado na metodologia do IBGE para agrupamento de municípios
    brasileiros segundo intervalo de população residente.

    Attributes:
        PEQUENO_PORTE_I: Limite superior para Pequeno Porte I (≤ 20.000 hab.)
        PEQUENO_PORTE_II: Limite superior para Pequeno Porte II (20.001–50.000 hab.)
        MEDIO_PORTE: Limite superior para Médio Porte (50.001–100.000 hab.)
        GRANDE_PORTE: Limite superior para Grande Porte (100.001–900.000 hab.)
        METROPOLE: Acima de 900.000 habitantes
    """
    PEQUENO_PORTE_I: int = 20_000
    PEQUENO_PORTE_II: int = 50_000
    MEDIO_PORTE: int = 100_000
    GRANDE_PORTE: int = 900_000

    LABELS: Dict[str, str] = field(default_factory=lambda: {
        "pequeno_i": "Pequeno Porte I",
        "pequeno_ii": "Pequeno Porte II",
        "medio": "Médio Porte",
        "grande": "Grande Porte",
        "metropole": "Metrópole",
    })


# ---------------------------------------------------------------------------
# Configuração de Séries Temporais
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSeriesConfig:
    """Parâmetros para construção e segmentação das séries temporais.

    Attributes:
        FREQ: Frequência de reamostragem (mensal).
        DATE_COLUMN: Nome da coluna de data no dataset bruto.
        TARGET_COLUMN: Nome da variável-alvo (taxa de internação).
        INTERNACOES_COLUMN: Coluna com contagem de internações.
        POPULACAO_COLUMN: Coluna com população do município.
        TAXA_MULTIPLIER: Fator multiplicador para cálculo da taxa (por 1.000 hab.).
        TRAIN_END: Data de corte para conjunto de treino (previsão 2019).
        TEST_START: Data de início do conjunto de teste.
        TEST_END: Data final do conjunto de teste.
        SEASONAL_PERIOD: Periodicidade sazonal (12 meses).
    """
    FREQ: str = "M"
    DATE_COLUMN: str = "Data completa"
    TARGET_COLUMN: str = "taxa_internacao"
    INTERNACOES_COLUMN: str = "Qtd. internacoes"
    POPULACAO_COLUMN: str = "populacao"
    TAXA_MULTIPLIER: int = 1_000
    TRAIN_END: str = "2018-12-31"
    TEST_START: str = "2019-01-01"
    TEST_END: str = "2019-12-31"
    SERIES_END: str = "2019-12-31"
    SEASONAL_PERIOD: int = 12


# ---------------------------------------------------------------------------
# Parâmetros dos Modelos Estatísticos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SESConfig:
    """Suavização Exponencial Simples (SES).

    Attributes:
        SMOOTHING_LEVEL: Parâmetro α (peso para observações recentes).
        OPTIMIZED: Se True, otimiza α automaticamente via MLE.
        FORECAST_HORIZON: Horizonte de previsão (meses).
    """
    SMOOTHING_LEVEL: float = 0.2
    OPTIMIZED: bool = False
    FORECAST_HORIZON: int = 12


@dataclass(frozen=True)
class HoltConfig:
    """Método de Holt (Suavização Exponencial Dupla).

    Attributes:
        SMOOTHING_LEVEL: Parâmetro α.
        SMOOTHING_TREND: Parâmetro β (tendência).
        OPTIMIZED: Se True, otimiza parâmetros automaticamente.
        FORECAST_HORIZON: Horizonte de previsão (meses).
    """
    SMOOTHING_LEVEL: float = 0.2
    SMOOTHING_TREND: float = 0.8
    OPTIMIZED: bool = False
    FORECAST_HORIZON: int = 12


@dataclass(frozen=True)
class HoltDampedConfig:
    """Holt com Tendência Amortecida.

    Attributes:
        SMOOTHING_LEVEL: Parâmetro α.
        SMOOTHING_TREND: Parâmetro β.
        DAMPED_TREND: Ativar amortecimento da tendência.
        FORECAST_HORIZON: Horizonte de previsão (meses).
    """
    SMOOTHING_LEVEL: float = 0.8
    SMOOTHING_TREND: float = 0.2
    DAMPED_TREND: bool = True
    FORECAST_HORIZON: int = 12


@dataclass(frozen=True)
class HoltWintersConfig:
    """Holt-Winters (Suavização Exponencial Tripla).

    Attributes:
        SEASONAL_PERIODS: Número de períodos sazonais.
        TREND: Tipo de tendência ('additive' ou 'multiplicative').
        SEASONAL: Tipo de sazonalidade ('add' ou 'mul').
        USE_BOXCOX: Aplicar transformação Box-Cox.
        FORECAST_HORIZON: Horizonte de previsão (meses).
    """
    SEASONAL_PERIODS: int = 12
    TREND: str = "additive"
    SEASONAL: str = "add"
    USE_BOXCOX: bool = True
    FORECAST_HORIZON: int = 12


# ---------------------------------------------------------------------------
# Parâmetros ARIMA / SARIMA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ARIMAConfig:
    """Configuração do modelo ARIMA.

    Attributes:
        START_P: Valor inicial de p para busca.
        START_Q: Valor inicial de q para busca.
        MAX_P: Valor máximo de p.
        MAX_Q: Valor máximo de q.
        D: Ordem de diferenciação.
        SEASONAL: Se True, ajusta modelo sazonal (SARIMA).
        FORECAST_HORIZON: Horizonte de previsão (meses).
    """
    START_P: int = 1
    START_Q: int = 1
    MAX_P: int = 6
    MAX_Q: int = 6
    D: int = 1
    SEASONAL: bool = False
    FORECAST_HORIZON: int = 12


@dataclass(frozen=True)
class SARIMAConfig:
    """Configuração do modelo SARIMA.

    Attributes:
        START_P: Valor inicial de p.
        START_Q: Valor inicial de q.
        MAX_P: Valor máximo de p.
        MAX_Q: Valor máximo de q.
        M: Periodicidade sazonal.
        START_SEASONAL_P: Valor inicial de P sazonal.
        D: Ordem de diferenciação.
        SEASONAL_D: Ordem de diferenciação sazonal.
        FORECAST_HORIZON: Horizonte de previsão.
    """
    START_P: int = 1
    START_Q: int = 1
    MAX_P: int = 6
    MAX_Q: int = 6
    M: int = 12
    START_SEASONAL_P: int = 0
    D: int = 1
    SEASONAL_D: int = 1
    FORECAST_HORIZON: int = 12


# ---------------------------------------------------------------------------
# Parâmetros LSTM
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LSTMConfig:
    """Configuração da rede LSTM (Long Short-Term Memory).

    Arquitetura: 4 camadas LSTM empilhadas com Dropout seguidas
    de uma camada Dense para regressão.

    Attributes:
        LAYER_UNITS: Tupla com número de neurônios por camada LSTM.
        DROPOUT_RATE: Taxa de dropout entre camadas.
        ACTIVATION: Função de ativação da camada de saída.
        OPTIMIZER: Algoritmo de otimização.
        LOSS: Função de perda.
        EPOCHS: Número de épocas de treinamento.
        BATCH_SIZE: Tamanho do mini-batch.
        LOOKBACK: Janela de observação (lag) para entrada do modelo.
        SCALER_RANGE: Intervalo de normalização MinMaxScaler.
    """
    LAYER_UNITS: Tuple[int, ...] = (800, 300, 100, 200)
    DROPOUT_RATE: float = 0.2
    ACTIVATION: str = "linear"
    OPTIMIZER: str = "adam"
    LOSS: str = "mean_squared_error"
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    LOOKBACK: int = 12
    SCALER_RANGE: Tuple[float, float] = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Configuração de Intervalos de Confiança
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfidenceIntervalConfig:
    """Parâmetros para cálculo de intervalos de confiança.

    Attributes:
        ALPHA_95: Nível de significância para IC de 95%.
        ALPHA_80: Nível de significância para IC de 80%.
    """
    ALPHA_95: float = 0.05
    ALPHA_80: float = 0.20


# ---------------------------------------------------------------------------
# Configuração de Visualização
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlotConfig:
    """Parâmetros globais de visualização.

    Attributes:
        FIGSIZE: Dimensões padrão das figuras (largura, altura).
        REAL_COLOR: Cor da série real.
        FORECAST_COLOR: Cor da série prevista.
        CI_95_COLOR: Cor do intervalo de confiança de 95%.
        CI_80_COLOR: Cor do intervalo de confiança de 80%.
        CI_95_ALPHA: Transparência do IC 95%.
        CI_80_ALPHA: Transparência do IC 80%.
        REAL_LABEL: Rótulo da série real.
        FORECAST_LABEL: Rótulo da série prevista.
        CI_95_LABEL: Rótulo do IC 95%.
        CI_80_LABEL: Rótulo do IC 80%.
        XLABEL: Rótulo do eixo X.
        YLABEL: Rótulo do eixo Y.
    """
    FIGSIZE: Tuple[int, int] = (15, 6)
    REAL_COLOR: str = "red"
    FORECAST_COLOR: str = "blue"
    CI_95_COLOR: str = "gray"
    CI_80_COLOR: str = "blue"
    CI_95_ALPHA: float = 0.3
    CI_80_ALPHA: float = 0.3
    REAL_LABEL: str = "Real"
    FORECAST_LABEL: str = "Previsto"
    CI_95_LABEL: str = "Intervalo de Confiança (95%)"
    CI_80_LABEL: str = "Intervalo de Confiança (80%)"
    XLABEL: str = "Data"
    YLABEL: str = "Taxa Internações"


# ---------------------------------------------------------------------------
# Configurações de Caminhos por Ambiente
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    """Caminhos de dados configuráveis por ambiente de execução.

    Suporta execução local e Google Colab, permitindo ajuste
    transparente dos caminhos de I/O.

    Attributes:
        BASE_DIR: Diretório raiz do projeto.
        DATA_RAW: Caminho para dados brutos.
        DATA_PROCESSED: Caminho para dados processados.
        DATA_RESULTS: Caminho para resultados.
        DATASET_FILENAME: Nome do arquivo principal de dados.
    """
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DATA_RAW: Optional[Path] = None
    DATA_PROCESSED: Optional[Path] = None
    DATA_RESULTS: Optional[Path] = None
    DATASET_FILENAME: str = "dataset_internacoes_completo.csv"

    def __post_init__(self) -> None:
        if self.DATA_RAW is None:
            self.DATA_RAW = self.BASE_DIR / "data" / "raw"
        if self.DATA_PROCESSED is None:
            self.DATA_PROCESSED = self.BASE_DIR / "data" / "processed"
        if self.DATA_RESULTS is None:
            self.DATA_RESULTS = self.BASE_DIR / "data" / "results"

    @property
    def dataset_path(self) -> Path:
        """Caminho completo do dataset principal."""
        return self.DATA_PROCESSED / self.DATASET_FILENAME

    def ensure_dirs(self) -> None:
        """Cria diretórios de dados caso não existam."""
        for d in (self.DATA_RAW, self.DATA_PROCESSED, self.DATA_RESULTS):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuração Agregada do Projeto
# ---------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    """Configuração agregada que centraliza todos os parâmetros do projeto.

    Uso típico:
        >>> from src.config.settings import ProjectConfig
        >>> cfg = ProjectConfig()
        >>> print(cfg.lstm.EPOCHS)
        100
    """
    porte: PorteConfig = field(default_factory=PorteConfig)
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    ses: SESConfig = field(default_factory=SESConfig)
    holt: HoltConfig = field(default_factory=HoltConfig)
    holt_damped: HoltDampedConfig = field(default_factory=HoltDampedConfig)
    holt_winters: HoltWintersConfig = field(default_factory=HoltWintersConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    sarima: SARIMAConfig = field(default_factory=SARIMAConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    ci: ConfidenceIntervalConfig = field(default_factory=ConfidenceIntervalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def summary(self) -> str:
        """Retorna resumo textual da configuração ativa."""
        return (
            f"ProjectConfig v{__import__('src').__version__}\n"
            f"  Séries temporais: freq={self.time_series.FREQ}, "
            f"sazonalidade={self.time_series.SEASONAL_PERIOD}\n"
            f"  LSTM: layers={self.lstm.LAYER_UNITS}, "
            f"epochs={self.lstm.EPOCHS}, batch={self.lstm.BATCH_SIZE}\n"
            f"  Dataset: {self.paths.dataset_path}\n"
        )
