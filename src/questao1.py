import numpy as np
import matplotlib.pyplot as plt

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
c_labels = np.array([0, 0, 1, 0, 1, 0], dtype=np.int64)
x_novo = np.array([6.0, 20.0], dtype=np.float64)

# print("Shape X_data:", X_data.shape)
# print("Shape y_prices:", y_prices.shape)
# print("Shape c_labels:", c_labels.shape)

diff = X_data - x_novo

# Cálculo vetorizado das três normas ao longo do eixo das características (axis=1)
d_l1 = np.sum(np.abs(diff), axis=1)
d_l2 = np.sqrt(np.sum(diff ** 2, axis=1))
d_linf = np.max(np.abs(diff), axis=1)


# Ranking dos vizinhos (indices de 1 a 6 ordenados da menor para a maior distância)
rank_l1 = np.argsort(d_l1) + 1
rank_l2 = np.argsort(d_l2) + 1
rank_linf = np.argsort(d_linf) + 1

print("Item 1 (Parte A):")
for i in range(len(X_data)):
    print(f"Amostra {i+1}: L1 = {d_l1[i]:5.1f} | L2 = {d_l2[i]:7.4f} | Linf = {d_linf[i]:5.1f}")

print("\nOrdem de proximidade (1º ao 6º vizinho):")
print("  Ranking L1  :", rank_l1)
print("  Ranking L2  :", rank_l2)
print("  Ranking Linf:", rank_linf)


## MONTAGEM DA TABELA

pos_l1 = {}
pos_l2 = {}
pos_linf = {}
for i in range(len(rank_l1)):
    pos_l1[rank_l1[i]] = i + 1
    pos_l2[rank_l2[i]] = i + 1
    pos_linf[rank_linf[i]] = i + 1

# A montagem da tabela a seguir foi feita com auxílio de IA 
# devido minha incompetência com matplotlib

colunas = ["Amostra (i)", "x_i^T", "d_L1", "Rank L1", "d_L2", "Rank L2", "d_L∞", "Rank L∞"]
linhas = []
for i in range(len(X_data)):
    linhas.append([
        f"{i+1}",
        f"[{X_data[i,0]:.1f}, {X_data[i,1]:.1f}]",
        f"{d_l1[i]:.1f}",
        f"{pos_l1[i+1]}º" + (" (empate)" if i+1 in [2, 6] else ""),
        f"{d_l2[i]:.4f}",
        f"{pos_l2[i+1]}º",
        f"{d_linf[i]:.1f}",
        f"{pos_linf[i+1]}º"
    ])

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")

tabela = ax.table(
    cellText=linhas,
    colLabels=colunas,
    loc="center",
    cellLoc="center"
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.1, 1.5)

# Estilização do cabeçalho
for j in range(len(colunas)):
    celula = tabela[(0, j)]
    celula.set_facecolor("#2c3e50")
    celula.set_text_props(color="white", weight="bold")

plt.title("Distâncias até x_novo = [6.0, 20.0]^T e Ranking dos Vizinhos", pad=12, weight="bold")
plt.tight_layout()
plt.savefig("tabela_distancias_q1.png", dpi=300, bbox_inches="tight")
plt.show()