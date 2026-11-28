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

print("Questão 3 (Parte A)")
print(f"w_hat (via solve) : {w_solve}")
print(f"w_hat (via lstsq) : {w_lstsq}")
print(f"Projeção y_hat    : {y_hat}")
print(f"Resíduo  e_hat    : {e_hat}")
print(f"‖A𝑇⁢ˆ𝐞‖ < 1e-14   : {min_ATe}")



####################################################################
# PARTE C

U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

eigen_ATA = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]


print("\nQuestão 3 (Parte C)")
print(f"Valores singulares (sigma) : {sigma}")
print(f"sigma^2                    : {sigma**2}")
print(f"λ(A^T A)                   : {eigen_ATA}")


####################################################################
# PARTE D

# Â = 𝜎1 ⁢𝐮1 ⁢𝐯𝑇1
u_1 = U[:, [0]]
v_1 = Vt[[0], :].T
A_hat_1 = sigma[0] * (u_1 @ v_1.T)

error_espectral = np.linalg.norm(A - A_hat_1, ord=2)
check_ahat = np.isclose(error_espectral, sigma[1])

print("\nQuestão 3 (Parte D)")
print(f"||A - Â(1)||_2        : {error_espectral:.2f}")
print(f" 𝜎2                   : {sigma[1]:.2f}")
print(f"||A - Â(1)||_2 == 𝜎2? : {check_ahat}")