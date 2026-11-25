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
x_novo = np.array([6.0, 20.0], dtype=np.float64)

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

# Parte B: equações normais e previsão de preço
Gram = X_design.T @ X_design
Xty = X_design.T @ y_prices
print("Matriz de Gram:", Gram, sep="\n")
print("Vetor X^T y:", Xty)
w_hat = np.linalg.solve(Gram, Xty)
w_lstsq = np.linalg.lstsq(X_design, y_prices, rcond=None)[0]
np.testing.assert_allclose(w_hat, w_lstsq, atol=1e-12, rtol=0)
print("Pesos para intercepto, área e idade:", w_hat)

residuos = y_prices - X_design @ w_hat
erro_ortogonalidade = np.linalg.norm(X_design.T @ residuos, ord=np.inf)
assert erro_ortogonalidade < 1e-10
print(f"Norma infinita de X^T e: {erro_ortogonalidade:.3e}")

preco_predito = np.r_[1.0, x_novo] @ w_hat
vizinhos = np.argsort(np.linalg.norm(X_data - x_novo, axis=1))[:3]
preco_knn = y_prices[vizinhos].mean()
print("Amostras vizinhas:", vizinhos + 1)
print(f"Preços em centenas de milhares de reais: OLS = {preco_predito:.6f}, k-NN = {preco_knn:.6f}")
