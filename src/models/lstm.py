"""
Modelo LSTM (Long Short-Term Memory)
======================================

Implementação da rede neural recorrente LSTM para previsão de séries
temporais de internações hospitalares.

Arquitetura:
    4 camadas LSTM empilhadas (800 → 300 → 100 → 200 unidades) com
    Dropout (0.2) entre camadas, seguidas de uma camada Dense linear
    para regressão univariada.

O modelo é treinado com otimizador Adam e função de perda MSE,
utilizando normalização MinMaxScaler no intervalo [0, 1].

Referências:
    - Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation, 9(8), 1735-1780.
    - Graves, A. (2012). Supervised Sequence Labelling with Recurrent
      Neural Networks. Springer.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config.settings import LSTMConfig, ProjectConfig

logger = logging.getLogger(__name__)


class LSTMModel:
    """Wrapper para construção, treinamento e previsão com LSTM.

    Encapsula a arquitetura definida nos notebooks originais com
    interface parametrizada via ``LSTMConfig``.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações do projeto.

    Attributes
    ----------
    model : keras.Model or None
        Modelo Keras compilado.
    scaler : MinMaxScaler or None
        Normalizer ajustado aos dados de treino.
    history : keras.callbacks.History or None
        Histórico de treinamento.

    Examples
    --------
    >>> lstm = LSTMModel()
    >>> lstm.build()
    >>> lstm.train(x_train, y_train, epochs=100)
    >>> predictions = lstm.predict(x_test)
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()
        self.lstm_cfg: LSTMConfig = self.config.lstm
        self.model = None
        self.scaler: Optional[MinMaxScaler] = None
        self.history = None

    # ------------------------------------------------------------------
    # Construção do Modelo
    # ------------------------------------------------------------------
    def build(self, input_shape: Optional[Tuple[int, int]] = None) -> None:
        """Constrói a arquitetura LSTM empilhada.

        Cria 4 camadas LSTM com as unidades definidas em ``LSTMConfig``,
        intercaladas com Dropout, e uma camada Dense de saída.

        Parameters
        ----------
        input_shape : tuple, optional
            Shape de entrada (timesteps, features). Se None, usa
            (LOOKBACK, 1).
        """
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Sequential

        cfg = self.lstm_cfg
        shape = input_shape or (cfg.LOOKBACK, 1)

        logger.info(
            "Construindo LSTM: camadas=%s, dropout=%.2f, input_shape=%s",
            cfg.LAYER_UNITS,
            cfg.DROPOUT_RATE,
            shape,
        )

        model = Sequential()

        # Camada 1 (com input_shape)
        model.add(LSTM(units=cfg.LAYER_UNITS[0], return_sequences=True, input_shape=shape))
        model.add(Dropout(cfg.DROPOUT_RATE))

        # Camadas intermediárias
        for units in cfg.LAYER_UNITS[1:-1]:
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(cfg.DROPOUT_RATE))

        # Última camada LSTM (sem return_sequences)
        model.add(LSTM(units=cfg.LAYER_UNITS[-1]))
        model.add(Dropout(cfg.DROPOUT_RATE))

        # Camada de saída
        model.add(Dense(units=1, activation=cfg.ACTIVATION))

        model.compile(
            optimizer=cfg.OPTIMIZER,
            loss=cfg.LOSS,
            metrics=[cfg.LOSS],
        )

        self.model = model
        logger.info("Modelo LSTM compilado com sucesso.")
        logger.info(model.summary())

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Treina o modelo LSTM.

        Parameters
        ----------
        x_train : np.ndarray
            Dados de entrada com shape (samples, timesteps, features).
        y_train : np.ndarray
            Valores-alvo.
        epochs : int, optional
            Número de épocas. Se None, usa ``LSTMConfig.EPOCHS``.
        batch_size : int, optional
            Tamanho do mini-batch. Se None, usa ``LSTMConfig.BATCH_SIZE``.
        """
        if self.model is None:
            self.build(input_shape=(x_train.shape[1], 1))

        cfg = self.lstm_cfg
        n_epochs = epochs or cfg.EPOCHS
        n_batch = batch_size or cfg.BATCH_SIZE

        logger.info(
            "Iniciando treinamento: epochs=%d, batch_size=%d, "
            "amostras=%d",
            n_epochs,
            n_batch,
            len(x_train),
        )

        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=n_batch,
            epochs=n_epochs,
        )

        logger.info("Treinamento concluído.")

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    def predict(
        self,
        x_test: np.ndarray,
        scaler: Optional[MinMaxScaler] = None,
    ) -> np.ndarray:
        """Gera previsões e aplica transformação inversa.

        Parameters
        ----------
        x_test : np.ndarray
            Dados de teste.
        scaler : MinMaxScaler, optional
            Scaler para inversão da normalização.

        Returns
        -------
        np.ndarray
            Previsões na escala original.
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi construído/treinado. Chame build() e train() primeiro.")

        predictions = self.model.predict(x_test)

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)

        logger.info("Previsões geradas: %d amostras.", len(predictions))
        return predictions

    # ------------------------------------------------------------------
    # Utilitário: Montar DataFrame de Resultados
    # ------------------------------------------------------------------
    @staticmethod
    def build_results_dataframe(
        predictions: np.ndarray,
        y_test: np.ndarray,
        start_date: str,
        freq: str = "M",
    ) -> pd.DataFrame:
        """Constrói DataFrame estruturado com previsões e valores reais.

        Parameters
        ----------
        predictions : np.ndarray
            Previsões do modelo.
        y_test : np.ndarray
            Valores reais.
        start_date : str
            Data inicial para o índice.
        freq : str
            Frequência do índice temporal.

        Returns
        -------
        pd.DataFrame
            DataFrame com colunas 'previsao', 'valor_real' e índice temporal.
        """
        results = pd.DataFrame(
            zip(predictions, y_test),
            columns=["previsao", "valor_real"],
        )
        results["previsao"] = results["previsao"].apply(
            lambda x: x[0] if hasattr(x, "__len__") else x
        )
        results["valor_real"] = results["valor_real"].apply(
            lambda x: x[0] if hasattr(x, "__len__") else x
        )

        results["data"] = pd.date_range(
            start=start_date,
            periods=len(results),
            freq=freq,
        )
        results.set_index("data", inplace=True)

        return results
