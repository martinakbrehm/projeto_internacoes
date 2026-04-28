"""
Setup do pacote — Previsão de Internações Hospitalares
"""

from setuptools import find_packages, setup

setup(
    name="projeto_internacoes",
    version="1.0.0",
    author="Martina",
    description=(
        "Pipeline de previsão de internações hospitalares utilizando "
        "modelos de séries temporais (ARIMA, SARIMA, Holt-Winters, LSTM)"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "statsmodels>=0.13.0",
        "pmdarima>=2.0.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
