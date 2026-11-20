import matplotlib
matplotlib.use("TkAgg")   # чтобы окна открывались в PyCharm;
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def hungarian_min(cost_matrix):
    """Венгерский алгоритм для МИНИМИЗАЦИИ суммарного веса."""
    if not cost_matrix or not cost_matrix[0]:
        return 0, []

    n = len(cost_matrix)        # число вершин в левой доле
    m = len(cost_matrix[0])     # число вершин в правой доле

    # Алгоритм требует n <= m (строк не больше, чем столбцов).
    # Если это не так — транспонируем матрицу, запомнив это.
    transposed = False
    if n > m:
        cost_matrix = [list(row) for row in zip(*cost_matrix)]
        n, m = m, n
        transposed = True

    INF = float('inf')

    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Восстанавливаем дополняющий путь
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Собираем паросочетание
    matching = []
    for j in range(1, m + 1):
        if p[j] != 0 and p[j] <= n:
            matching.append((p[j] - 1, j - 1))

    # Если транспонировали — возвращаем пары в исходной ориентации
    if transposed:
        matching = [(j, i) for (i, j) in matching]

    # Считаем вес по исходным весам
    total = 0
    for i, j in matching:
        total += cost_matrix[i][j] if not transposed else cost_matrix[j][i]

    return total, matching


def hungarian_max(cost_matrix):
    """Венгерский алгоритм для МАКСИМИЗАЦИИ суммарного веса."""
    inverted = [[-x for x in row] for row in cost_matrix]
    _, matching = hungarian_min(inverted)
    total = sum(cost_matrix[i][j] for i, j in matching)
    return total, matching


def print_result(title, cost_matrix, total, matching):
    """Красивый вывод результата."""
    print(f"\n=== {title} ===")
    print("Матрица весов:")
    for row in cost_matrix:
        print("  ", row)
    print("Паросочетание (i, j):", matching)
    print("Суммарный вес:", total)
    print("Проверка (ручной подсчёт):",
          sum(cost_matrix[i][j] for i, j in matching))


def visualize_heatmap(cost_matrix, matching, title="Паросочетание",
                      filename=None):
    """Тепловая карта матрицы весов с выделением выбранных ячеек."""
    arr = np.array(cost_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="YlOrRd")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                    color="black", fontsize=11)

    for (i, j) in matching:
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor="blue", linewidth=3)
        ax.add_patch(rect)

    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xlabel("Правая доля V (столбцы)")
    ax.set_ylabel("Левая доля U (строки)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Вес")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        print(f"Тепловая карта сохранена в файл: {filename}")

    plt.show()


def visualize_matching(cost_matrix, matching, title="Паросочетание",
                       filename=None):
    """Двудольный граф: слева U, справа V, выбранные рёбра — красные."""
    n = len(cost_matrix)
    m = len(cost_matrix[0])

    G = nx.Graph()

    left_nodes = [f"U{i}" for i in range(n)]
    right_nodes = [f"V{j}" for j in range(m)]

    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    for i in range(n):
        for j in range(m):
            G.add_edge(f"U{i}", f"V{j}", weight=cost_matrix[i][j])

    pos = {}
    for i, node in enumerate(left_nodes):
        pos[node] = (0, -i)
    for j, node in enumerate(right_nodes):
        pos[node] = (1, -j)

    chosen_edges = [(f"U{i}", f"V{j}") for (i, j) in matching]

    plt.figure(figsize=(7, max(3, 0.6 * max(n, m))))

    nx.draw_networkx_edges(G, pos, edge_color="lightgray",
                           width=1, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=chosen_edges,
                           edge_color="red", width=3)
    nx.draw_networkx_nodes(G, pos, nodelist=left_nodes,
                           node_color="lightblue", node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=right_nodes,
                           node_color="lightgreen", node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edge_labels = {(u, v): G[u][v]["weight"] for (u, v) in chosen_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="red", font_size=10)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        print(f"Визуализация сохранена в файл: {filename}")

    plt.show()

# Пример использования
if __name__ == "__main__":
    # Пример 1: квадратная матрица 4x4
    cost = [
        [4, 1, 3, 2],
        [2, 0, 5, 3],
        [3, 2, 2, 4],
        [1, 3, 2, 1],
    ]

    print("Исходная матрица весов (задача о назначениях):")
    for row in cost:
        print("  ", row)

    # МИНИМИЗАЦИЯ
    min_cost, matching_min = hungarian_min(cost)
    print_result("МИНИМИЗАЦИЯ", cost, min_cost, matching_min)
    visualize_matching(cost, matching_min,
                       title=f"Минимальное паросочетание (вес = {min_cost})",
                       filename="matching_min.png")
    visualize_heatmap(cost, matching_min,
                      title=f"Минимальное паросочетание (вес = {min_cost})",
                      filename="matching_min_heatmap.png")

    # МАКСИМИЗАЦИЯ
    max_cost, matching_max = hungarian_max(cost)
    print_result("МАКСИМИЗАЦИЯ", cost, max_cost, matching_max)
    visualize_matching(cost, matching_max,
                       title=f"Максимальное паросочетание (вес = {max_cost})",
                       filename="matching_max.png")
    visualize_heatmap(cost, matching_max,
                      title=f"Максимальное паросочетание (вес = {max_cost})",
                      filename="matching_max_heatmap.png")

    # ПРЯМОУГОЛЬНАЯ 3x4
    cost_rect = [
        [5, 9, 1, 7],
        [3, 6, 8, 2],
        [4, 4, 6, 5],
    ]
    min_cost_r, matching_r = hungarian_min(cost_rect)
    print_result("ПРЯМОУГОЛЬНАЯ 3x4 (минимизация)",
                 cost_rect, min_cost_r, matching_r)
    visualize_matching(cost_rect, matching_r,
                       title=f"Прямоугольная 3x4, мин. (вес = {min_cost_r})",
                       filename="matching_rect.png")
    visualize_heatmap(cost_rect, matching_r,
                      title=f"Прямоугольная 3x4, мин. (вес = {min_cost_r})",
                      filename="matching_rect_heatmap.png")