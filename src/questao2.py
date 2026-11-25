"""Questão 2: Regressão Linear, Posto de Matrizes e Mínimos Quadrados Clássicos (OLS)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.datasets import get_dataset_d


def gaussian_elimination_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resolve o sistema linear A x = b via Eliminação Gaussiana com pivoteamento parcial em NumPy puro."""
    n = len(b)
    M = np.column_stack([A.astype(np.float64), b.astype(np.float64)])
    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(M[k:, k])))
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1 : n], x[i + 1 : n])) / M[i, i]
    return x


def run_questao2(save_figures: bool = True) -> dict:
    """Executa todas as validações da Questão 2 e gera figuras."""
    X_data, y_prices, _, x_novo = get_dataset_d()
    N = len(X_data)
    X_design = np.column_stack([np.ones(N), X_data])
    X_aug = np.column_stack([X_design, y_prices])

    rk_X = int(np.linalg.matrix_rank(X_design))
    rk_X_aug = int(np.linalg.matrix_rank(X_aug))
    assert rk_X == 3 and rk_X_aug == 4, "Postos incorretos para X e [X|y]!"

    # Parte A.3: Coluna artificial colinear x3 = 2.5 * x1
    X_collinear = np.column_stack([X_design, 2.5 * X_data[:, 0]])
    rk_collinear = int(np.linalg.matrix_rank(X_collinear))
    rk_gram_collinear = int(np.linalg.matrix_rank(X_collinear.T @ X_collinear))
    assert rk_collinear == 3 and rk_gram_collinear == 3, "Posto incorreto sob multicolinearidade!"

    # Parte B: Equações Normais
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y_prices
    w_gauss = gaussian_elimination_solve(XtX, Xty)
    w_solve = np.linalg.solve(XtX, Xty)
    w_lstsq, _, _, _ = np.linalg.lstsq(X_design, y_prices, rcond=None)
    assert np.allclose(w_gauss, w_solve, rtol=0, atol=1e-12) and np.allclose(w_solve, w_lstsq, rtol=0, atol=1e-12)

    y_hat = X_design @ w_solve
    e_hat = y_prices - y_hat
    ortho_inf_norm = float(np.max(np.abs(X_design.T @ e_hat)))
    assert ortho_inf_norm < 1e-10, f"Falha na ortogonalidade: {ortho_inf_norm}"

    x_novo_design = np.array([1.0, x_novo[0], x_novo[1]])
    y_lin_novo = float(x_novo_design @ w_solve)

    distancias = np.linalg.norm(X_data - x_novo, axis=1)
    vizinhos = np.argsort(distancias)[:3]
    y_knn_novo = float(np.mean(y_prices[vizinhos]))

    print("=== QUESTÃO 2: REGRESSÃO LINEAR, POSTO E OLS ===")
    print(f"Posto rk(X) = {rk_X} | Posto rk([X|y]) = {rk_X_aug} -> Sistema Incompatível (Rouché-Capelli)")
    print(f"Com x3 = 2.5*x1: rk(X_tilde) = {rk_collinear}, rk(X_tilde^T X_tilde) = {rk_gram_collinear} (< 4, singular)")
    print(f"Matriz de Gram X^T X:\n{XtX}")
    print(f"Vetor X^T y: {Xty}")
    print(f"Pesos ótimos w_hat = {w_solve}")
    print(f"Norma máxima de ortogonalidade ||X^T (y - X w_hat)||_inf = {ortho_inf_norm:.3e} (< 1e-10)")
    print(f"Predição linear para x_novo [6.0, 20.0]: y_lin = {y_lin_novo:.6f} (vs k-NN k=3: {y_knn_novo:.6f})\n")

    if save_figures:
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x1_grid = np.linspace(1.5, 9.5, 100)
        y_line_age20 = w_solve[0] + w_solve[1] * x1_grid + w_solve[2] * 20.0
        y_line_age30 = w_solve[0] + w_solve[1] * x1_grid + w_solve[2] * 30.0
        ax.scatter(X_data[:, 0], y_prices, c="#1f77b4", s=80, label="Amostras Reais (Dataset D)", zorder=3)
        ax.plot(x1_grid, y_line_age20, color="#2ca02c", lw=2, label="Plano OLS fixando Idade = 20 anos")
        ax.plot(x1_grid, y_line_age30, color="#ff7f0e", lw=2, ls="--", label="Plano OLS fixando Idade = 30 anos")
        ax.scatter([x_novo[0]], [y_lin_novo], c="#2ca02c", marker="D", s=110, label=f"OLS x_novo ({y_lin_novo:.2f})", zorder=4)
        ax.scatter([x_novo[0]], [y_knn_novo], c="#d62728", marker="^", s=110, label=f"k-NN (k=3) x_novo ({y_knn_novo:.2f})", zorder=4)
        ax.set_title("Comparação Global Paramétrica (OLS) vs. Local Não-Paramétrica (k-NN)")
        ax.set_xlabel("Área x1 (dezenas de m²)")
        ax.set_ylabel("Preço y (centenas de milhares de R$)")
        ax.legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig("figures/q2_ols_vs_knn_superficie.png", dpi=300)
        plt.close(fig)

    return {"w_hat": w_solve, "y_lin_novo": y_lin_novo, "y_knn_novo": y_knn_novo, "ortho_inf_norm": ortho_inf_norm}


if __name__ == "__main__":
    run_questao2()
