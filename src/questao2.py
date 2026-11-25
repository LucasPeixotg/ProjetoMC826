import numpy as np

# 1. Dataset D: Amostras imobiliárias (N=6, d=2)
X_data = np.array([
    [5.0, 30.0],
    [7.0, 38.0],
    [9.0, 40.0],
    [3.5, 19.0],
    [8.0, 15.0],
    [2.0, 35.0]
], dtype=np.float64)

y_prices = np.array([2.5, 3.8, 4.2, 1.9, 4.5, 1.2], dtype=np.float64)

# Parte A: posto e compatibilidade
N = len(X_data)
X_design = np.column_stack([np.ones(N), X_data])
X_aumentada = np.column_stack([X_design, y_prices])
posto = np.linalg.matrix_rank(X_design)
posto_aumentado = np.linalg.matrix_rank(X_aumentada)
print("Postos de X e [X | y]:", posto, posto_aumentado)
assert (posto, posto_aumentado) == (3, 4)

X_colinear = np.column_stack([X_design, 2.5 * X_data[:, 0]])
posto_colinear = np.linalg.matrix_rank(X_colinear)
posto_gram_colinear = np.linalg.matrix_rank(X_colinear.T @ X_colinear)
print("Postos de X e da matriz de Gram com a coluna adicional:", posto_colinear, posto_gram_colinear)
assert (posto_colinear, posto_gram_colinear) == (3, 3)
