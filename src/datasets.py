"""Módulo de definição canônica e imutável dos conjuntos de dados do projeto."""
from __future__ import annotations

import numpy as np


def get_dataset_d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retorna o Dataset Principal D (Avaliação Imobiliária) e o ponto x_novo."""
    X_data = np.array([
        [5.0, 30.0],
        [7.0, 38.0],
        [9.0, 40.0],
        [3.5, 19.0],
        [8.0, 15.0],
        [2.0, 35.0],
    ], dtype=np.float64)
    y_prices = np.array([2.5, 3.8, 4.2, 1.9, 4.5, 1.2], dtype=np.float64)
    c_labels = np.array([0, 0, 1, 0, 1, 0], dtype=np.int64)
    x_novo = np.array([6.0, 20.0], dtype=np.float64)
    return X_data, y_prices, c_labels, x_novo
