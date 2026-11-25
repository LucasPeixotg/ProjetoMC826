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

####################################################################
# PARTE A

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

# A montagem da tabela a seguir foi feita com auxílio de IA, visto que  a estética de plotagem em 
# matplotlib é apenas um meio visual de apresentação, não o objeto de avaliação teórica

colunas = [
    r"Amostra ($i$)",
    r"$\mathbf{x}_i^T$",
    r"$d_{L_1}$",
    r"Rank $L_1$",
    r"$d_{L_2}$",
    r"Rank $L_2$",
    r"$d_{L_\infty}$",
    r"Rank $L_\infty$"
]
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

plt.title("Distâncias até $x_{novo} = [6.0, 20.0]^T$ e Ranking dos Vizinhos", pad=12, weight="bold")
plt.tight_layout()
plt.savefig("tabela_distancias_q1.png", dpi=300, bbox_inches="tight")
plt.show()



## MONTAGEM DA TABELA com os dados normalizados

mu_data = np.mean(X_data, axis=0)
sigma_data = np.std(X_data, axis=0, ddof=1)

X_pad = (X_data - mu_data) / sigma_data
x_novo_pad = (x_novo - mu_data) / sigma_data

diff_pad = X_pad - x_novo_pad

d_l1_pad = np.sum(np.abs(diff_pad), axis=1)
d_l2_pad = np.sqrt(np.sum(diff_pad ** 2, axis=1))
d_linf_pad = np.max(np.abs(diff_pad), axis=1)

rank_l1_pad = np.argsort(d_l1_pad) + 1
rank_l2_pad = np.argsort(d_l2_pad) + 1
rank_linf_pad = np.argsort(d_linf_pad) + 1

pos_l1_pad = {amostra: pos + 1 for pos, amostra in enumerate(rank_l1_pad)}
pos_l2_pad = {amostra: pos + 1 for pos, amostra in enumerate(rank_l2_pad)}
pos_linf_pad = {amostra: pos + 1 for pos, amostra in enumerate(rank_linf_pad)}

colunas = [
    r"Amostra ($i$)",
    r"$d_{L_1}$ (Pad.)",
    r"Rank $L_1$",
    r"Rank $L_1$ (Pad.)",
    r"$d_{L_2}$ (Pad.)",
    r"Rank $L_2$",
    r"Rank $L_2$ (Pad.)",
    r"$d_{L_\infty}$ (Pad.)",
    r"Rank $L_\infty$",
    r"Rank $L_\infty$ (Pad.)"
]

linhas = []
for i in range(len(X_data)):
    linhas.append([
        f"{i+1}",
        f"{d_l1_pad[i]:.4f}",
        f"{pos_l1[i+1]}º" + ("*" if i + 1 in [2, 6] else ""),
        f"{pos_l1_pad[i+1]}º",
        f"{d_l2_pad[i]:.4f}",
        f"{pos_l2[i+1]}º",
        f"{pos_l2_pad[i+1]}º",
        f"{d_linf_pad[i]:.4f}",
        f"{pos_linf[i+1]}º",
        f"{pos_linf_pad[i+1]}º"
    ])

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")

tabela = ax.table(
    cellText=linhas,
    colLabels=colunas,
    loc="center",
    cellLoc="center"
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(9.5)
tabela.scale(1.05, 1.55)

for j in range(len(colunas)):
    celula = tabela[(0, j)]
    celula.set_facecolor("#2c3e50")
    celula.set_text_props(color="white", weight="bold")

plt.title(
    "Comparação dos Rankings de Proximidade: Dados Brutos vs. Padronizados (Z-score)",
    pad=12,
    weight="bold"
)
plt.tight_layout()
plt.savefig("tabela_comparacao_ranks_q1.png", dpi=300, bbox_inches="tight")
plt.show()


####################################################################
# PARTE B: Classificação e Regressão KNN
idx_ordenados_l2 = np.argsort(d_l2)

# classificação com k = 1
vizinho_k1 = idx_ordenados_l2[0]
c_pred_k1 = c_labels[vizinho_k1]

# classificação / regressão com k = 3
vizinhos_k3 = idx_ordenados_l2[:3]
c_pred_k3 = np.argmax(np.bincount(c_labels[vizinhos_k3]))
y_pred_k3 = np.mean(y_prices[vizinhos_k3])

# classificação / regressão com k = 6
c_pred_k6 = np.argmax(np.bincount(c_labels))
y_pred_k6 = np.mean(y_prices)

print("\nQuestão 1 (Parte B): \n\nPredições com KNN (Distância Euclidiana L2)")
print(f"3 vizinhos mais próximos (amostras): {vizinhos_k3 + 1}")
print(f"Classificação (k=1) : c_novo = {c_pred_k1}")
print(f"Classificação (k=3) : c_novo = {c_pred_k3}")
print(f"Regressão     (k=3) : y_novo = {y_pred_k3}")
print(f"Classificação (k=6) : c_novo = {c_pred_k6}")
print(f"Regressão     (k=6) : y_novo = {y_pred_k6}")
