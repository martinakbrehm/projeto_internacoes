"""
Métricas de Avaliação de Modelos
=================================

Módulo que implementa as métricas de avaliação utilizadas para comparar
o desempenho dos modelos de previsão de séries temporais.

Métricas implementadas:
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Theil's U₂ Coefficient

Referências:
    - Theil, H. (1966). Applied Economic Forecasting. North-Holland.
    - Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures
      of forecast accuracy. International Journal of Forecasting.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Relatório estruturado de avaliação de um modelo.

    Attributes
    ----------
    model_name : str
        Nome identificador do modelo.
    porte : str
        Porte municipal avaliado.
    mae : float
        Mean Absolute Error.
    mse : float
        Mean Squared Error.
    rmse : float
        Root Mean Squared Error.
    mape : float
        Mean Absolute Percentage Error.
    theil_u2 : float, optional
        Theil's U₂ Coefficient.
    """
    model_name: str
    porte: str
    mae: float
    mse: float
    rmse: float
    mape: float
    theil_u2: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Converte o relatório em dicionário."""
        return {
            "Modelo": self.model_name,
            "Porte": self.porte,
            "MAE": self.mae,
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "Theil U₂": self.theil_u2,
        }

    def __str__(self) -> str:
        lines = [
            f"{'='*55}",
            f"  Relatório de Avaliação — {self.model_name}",
            f"  Porte: {self.porte}",
            f"{'='*55}",
            f"  MAE  : {self.mae:.6f}",
            f"  MSE  : {self.mse:.6f}",
            f"  RMSE : {self.rmse:.6f}",
            f"  MAPE : {self.mape:.6f}",
        ]
        if self.theil_u2 is not None:
            lines.append(f"  Theil U₂: {self.theil_u2:.6f}")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


class ModelEvaluator:
    """Avaliador centralizado de modelos de previsão.

    Implementa todas as métricas utilizadas no projeto e oferece
    interfaces tanto para avaliação individual quanto para comparação
    entre múltiplos modelos.

    Examples
    --------
    >>> evaluator = ModelEvaluator()
    >>> report = evaluator.evaluate(y_real, y_pred, model_name="SARIMA", porte="Grande Porte")
    >>> print(report)
    """

    # ------------------------------------------------------------------
    # Theil's U₂ Coefficient
    # ------------------------------------------------------------------
    @staticmethod
    def theil_u2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o Coeficiente U₂ de Theil.

        O coeficiente U₂ de Theil compara a acurácia da previsão com
        a previsão naïve (último valor observado). Valores U₂ < 1
        indicam que o modelo supera a previsão naïve.

        Fórmula:
            U₂ = √(1/N Σ((Yₜ₊₁ᵖ - Yₜ₊₁) / Yₜ)²) /
                  √(1/N Σ((Yₜ₊₁ - Yₜ) / Yₜ)²)

        Parameters
        ----------
        y_true : array-like
            Valores reais observados.
        y_pred : array-like
            Valores previstos pelo modelo.

        Returns
        -------
        float
            Coeficiente U₂ de Theil.

        References
        ----------
        Theil, H. (1966). Applied Economic Forecasting. North-Holland.
        """
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        N = len(y_true)

        # Numerador
        numerator_terms = (y_pred[1:] - y_true[1:]) / y_true[:-1]
        numerator = np.sqrt((1 / N) * np.sum(np.power(numerator_terms, 2)))

        # Denominador
        denominator_terms = (y_true[1:] - y_true[:-1]) / y_true[:-1]
        denominator = np.sqrt((1 / N) * np.sum(np.power(denominator_terms, 2)))

        return numerator / denominator

    # ------------------------------------------------------------------
    # Avaliação Completa
    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        model_name: str = "Modelo",
        porte: str = "N/A",
        compute_theil: bool = True,
    ) -> EvaluationReport:
        """Calcula todas as métricas de avaliação.

        Parameters
        ----------
        y_true : array-like
            Valores reais.
        y_pred : array-like
            Valores previstos.
        model_name : str
            Nome do modelo avaliado.
        porte : str
            Porte municipal.
        compute_theil : bool
            Se True, calcula o coeficiente U₂ de Theil.

        Returns
        -------
        EvaluationReport
            Relatório com todas as métricas calculadas.
        """
        y_true_arr = np.ravel(y_true)
        y_pred_arr = np.ravel(y_pred)

        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr)

        theil = None
        if compute_theil and len(y_true_arr) > 1:
            theil = self.theil_u2(y_true_arr, y_pred_arr)

        report = EvaluationReport(
            model_name=model_name,
            porte=porte,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            theil_u2=theil,
        )

        logger.info("Avaliação concluída para '%s' [%s].", model_name, porte)
        logger.info("  MAE=%.6f | RMSE=%.6f | MAPE=%.6f", mae, rmse, mape)

        return report

    # ------------------------------------------------------------------
    # Tabela Comparativa
    # ------------------------------------------------------------------
    @staticmethod
    def comparar_modelos(reports: list) -> pd.DataFrame:
        """Gera tabela comparativa entre múltiplos modelos.

        Parameters
        ----------
        reports : list of EvaluationReport
            Lista de relatórios de avaliação.

        Returns
        -------
        pd.DataFrame
            Tabela comparativa ordenada por RMSE (ascendente).
        """
        df = pd.DataFrame([r.to_dict() for r in reports])
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        return df
