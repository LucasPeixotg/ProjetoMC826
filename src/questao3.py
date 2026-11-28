import numpy as np

A = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0]
], dtype=np.float64)

y_A = np.array([4.0, 1.0, 1.0], dtype=np.float64)

# print("shape A:  ", A.shape)
# print("shape y_A:", y_A.shape)

####################################################################
# PARTE A

# Resolução via equação normal (AˆT A w = A^T ya)
w_solve = np.linalg.solve(A.T @ A, A.T @ y_A)

# Resolução via Mínimos Quadrados direto (np.linalg.lstsq)
w_lstsq = np.linalg.lstsq(A, y_A, rcond=None)[0]

# Projeção ortogonal y_hat = A w_hat
y_hat = A @ w_solve

# vetor de resíduos e_hat = y_A - y_hat
e_hat = y_A - y_hat

# Cheque de ‖A𝑇⁢ˆ𝐞‖∞ <10−14
min_ATe = np.max(np.abs(A.T @ e_hat)) < 1e-14

print("Questão 3 (Parte A): Projeção Ortogonal e Equações Normais")
print(f"w_hat (via solve) : {w_solve}")
print(f"w_hat (via lstsq) : {w_lstsq}")
print(f"Projeção y_hat    : {y_hat}")
print(f"Resíduo  e_hat    : {e_hat}")
print(f"‖A𝑇⁢ˆ𝐞‖ < 1e-14   : {min_ATe}")
