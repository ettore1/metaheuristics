# Gera uma figura ilustrativa de uma instância do TSPTW usando networkx:
# cidades, depósito, a rota de referência (vizinho mais próximo) e exemplos de
# janelas de tempo. Serve de figura de exemplo na formulação do problema.
#
# Uso: python -m src.plot_example_instance

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.instance_loader import _nearest_neighbor_route, load_instance

INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEPOT_COLOR = "#c0392b"
CITY_COLOR = "#2a78d6"
ROUTE_COLOR = "#8a8a8a"
WINDOW_COLOR = "#1baf7a"
TEXT_COLOR = "#0b0b0b"

# quantas janelas de tempo anotar (espaçadas ao longo da rota, para não poluir)
NUM_WINDOW_ANNOTATIONS = 5


def plot_example_instance(instance_file: str, output_path: Path) -> None:
    instance = load_instance(INSTANCES_DIR / instance_file)

    reference_route = _nearest_neighbor_route(instance.distance_matrix)
    closed_route = [*reference_route, 0]  # retorna ao depósito

    positions = {node: tuple(instance.coordinates[node]) for node in range(len(instance.coordinates))}

    graph = nx.DiGraph()
    graph.add_nodes_from(positions)
    route_edges = list(zip(closed_route[:-1], closed_route[1:]))
    graph.add_edges_from(route_edges)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # arestas da rota de referência
    nx.draw_networkx_edges(
        graph, positions, edgelist=route_edges, ax=ax,
        edge_color=ROUTE_COLOR, width=1.6, arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.02",
    )

    # cidades e depósito
    city_nodes = [n for n in graph.nodes if n != 0]
    nx.draw_networkx_nodes(graph, positions, nodelist=city_nodes, ax=ax,
                           node_color=CITY_COLOR, node_size=320, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_nodes(graph, positions, nodelist=[0], ax=ax,
                           node_color=DEPOT_COLOR, node_shape="s", node_size=520,
                           edgecolors="white", linewidths=1.4)

    labels = {n: ("D" if n == 0 else str(n)) for n in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax,
                            font_size=8, font_color="white", font_weight="bold")

    # anota janelas de tempo [e, l] de algumas cidades ao longo da rota
    cities_in_order = [n for n in closed_route if n != 0]
    step = max(1, len(cities_in_order) // NUM_WINDOW_ANNOTATIONS)
    for city in cities_in_order[::step][:NUM_WINDOW_ANNOTATIONS]:
        x, y = positions[city]
        e = instance.ready_time[city]
        l = instance.due_time[city]
        ax.annotate(
            f"[{e:.0f}, {l:.0f}]",
            xy=(x, y), xytext=(x + 2.5, y + 2.5),
            fontsize=8.5, color=WINDOW_COLOR, fontweight="bold",
        )

    ax.set_title(
        f"Instância de exemplo (n = {instance.num_cities}): rota de referência e janelas de tempo",
        color=TEXT_COLOR, fontsize=13,
    )
    ax.set_xlabel("coordenada x", color=TEXT_COLOR)
    ax.set_ylabel("coordenada y", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # legenda manual
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="s", color="white", markerfacecolor=DEPOT_COLOR,
               markersize=11, label="Depósito (D)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=CITY_COLOR,
               markersize=10, label="Cidade"),
        Line2D([0], [0], color=ROUTE_COLOR, lw=1.8, label="Rota de referência"),
        Line2D([0], [0], marker="", color=WINDOW_COLOR, lw=0,
               label="[e, l]: janela de tempo"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figura de exemplo salva em {output_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_example_instance("tsptw_small_n25.json", RESULTS_DIR / "fig_example_instance.pdf")


if __name__ == "__main__":
    main()
