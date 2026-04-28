"""
Carregamento de Dados
======================

Módulo responsável pelo carregamento unificado dos dados de internações
hospitalares, com suporte a múltiplos ambientes (local, Google Colab).

O padrão de projeto utilizado (classe ``DataLoader``) encapsula a lógica
de I/O e oferece uma interface consistente independentemente da origem
dos dados.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config.settings import ProjectConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Carregador unificado de datasets para o projeto de internações.

    Parameters
    ----------
    config : ProjectConfig, optional
        Configurações do projeto. Se não informado, utiliza valores padrão.

    Examples
    --------
    >>> loader = DataLoader()
    >>> df = loader.load_dataset()
    >>> df.shape
    (XXXXX, N)
    """

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        self.config = config or ProjectConfig()
        self._dataset: Optional[pd.DataFrame] = None

    def load_dataset(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Carrega o dataset principal de internações hospitalares.

        Parameters
        ----------
        filepath : str or Path, optional
            Caminho explícito para o arquivo CSV. Se omitido, utiliza
            o caminho definido na configuração do projeto.

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados brutos carregados.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não for encontrado no caminho especificado.
        """
        if filepath is None:
            filepath = self.config.paths.dataset_path

        filepath = Path(filepath)
        logger.info("Carregando dataset de: %s", filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em '{filepath}'. "
                f"Verifique o caminho ou execute o pipeline de preparação de dados."
            )

        self._dataset = pd.read_csv(filepath)
        logger.info(
            "Dataset carregado com sucesso: %d linhas × %d colunas",
            self._dataset.shape[0],
            self._dataset.shape[1],
        )
        return self._dataset

    def load_from_colab(
        self,
        drive_path: str = "/content/drive/My Drive/Projeto Internações/Preparação dos dados/dataset_internacoes_completo.csv",
    ) -> pd.DataFrame:
        """Carrega dados diretamente do Google Drive (ambiente Colab).

        Parameters
        ----------
        drive_path : str
            Caminho completo no Google Drive.

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados carregados.
        """
        logger.info("Carregando dataset do Google Drive: %s", drive_path)
        self._dataset = pd.read_csv(drive_path)
        logger.info(
            "Dataset carregado: %d linhas × %d colunas",
            self._dataset.shape[0],
            self._dataset.shape[1],
        )
        return self._dataset

    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        """Retorna o dataset em cache, se já carregado."""
        return self._dataset
