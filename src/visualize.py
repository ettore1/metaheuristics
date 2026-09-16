# Geração dos gráficos de escalabilidade e convergência a partir dos resultados
# salvos por 'experiment.py'
#
# Uso: python -m src.visualize

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# paleta categórica (3 primeiros slots: validados para distinção em pares,
# inclusive para daltonismo) + marcador/traço distintos por algoritmo, para
# que os gráficos continuem legíveis em impressão em preto e branco
ALGORITHM_STYLE = {
    "TS": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "ACO": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "PSO": {"color": "#1baf7a", "marker": "^", "linestyle": ":"},
}

GRIDLINE_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_COLOR = "#0b0b0b"


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRIDLINE_COLOR, linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


def plot_scalability(summary: pd.DataFrame, output_path: Path) -> plt.Figure:
    """Gera a figura de escalabilidade: custo, distância e tempo vs. tamanho da instância."""
    grouped = summary.groupby(["algorithm", "num_cities"]).agg(
        cost_mean=("best_cost", "mean"),
        cost_std=("best_cost", "std"),
        distance_mean=("total_distance", "mean"),
        distance_std=("total_distance", "std"),
        time_mean=("elapsed_seconds", "mean"),
        time_std=("elapsed_seconds", "std"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("cost_mean", "cost_std", "Custo (com penalidade)", axes[0], True),
        ("distance_mean", "distance_std", "Distância percorrida", axes[1], False),
        ("time_mean", "time_std", "Tempo de execução (s)", axes[2], True),
    ]

    for mean_col, std_col, title, ax, use_log_scale in metrics:
        for algorithm, style in ALGORITHM_STYLE.items():
            algorithm_data = grouped[grouped["algorithm"] == algorithm].sort_values("num_cities")
            ax.errorbar(
                algorithm_data["num_cities"],
                algorithm_data[mean_col],
                yerr=algorithm_data[std_col],
                label=algorithm,
                capsize=3,
                **style,
            )

        if use_log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("Número de cidades")
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_xticks(sorted(summary["num_cities"].unique()))
        _style_axes(ax)

    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def plot_convergence(convergence: pd.DataFrame, output_path: Path) -> plt.Figure:
    """Gera a figura de convergência: custo médio (entre execuções) por iteração, uma coluna por tamanho de instância."""
    instance_sizes = sorted(convergence["num_cities"].unique())
    fig, axes = plt.subplots(1, len(instance_sizes), figsize=(5 * len(instance_sizes), 4.5))

    if len(instance_sizes) == 1:
        axes = [axes]

    for ax, num_cities in zip(axes, instance_sizes):
        instance_data = convergence[convergence["num_cities"] == num_cities]

        for algorithm, style in ALGORITHM_STYLE.items():
            algorithm_data = instance_data[instance_data["algorithm"] == algorithm]
            mean_by_iteration = algorithm_data.groupby("iteration")["best_cost_so_far"].mean()

            ax.plot(
                mean_by_iteration.index,
                mean_by_iteration.to_numpy(),
                label=algorithm,
                marker=None,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Iteração")
        ax.set_title(f"n = {num_cities} cidades", color=TEXT_COLOR)
        _style_axes(ax)

    axes[0].set_ylabel("Custo médio (10 execuções, log)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def plot_convergence_per_instance(convergence: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Gera uma figura de convergência separada por tamanho de instância.

    Cada instância vira um arquivo próprio (fig_convergence_n{n}.pdf), com fontes
    maiores, o que torna o conteúdo mais legível do que o painel 1x3 combinado.
    Retorna a lista de caminhos salvos.
    """
    instance_sizes = sorted(convergence["num_cities"].unique())
    saved_paths: list[Path] = []

    for num_cities in instance_sizes:
        instance_data = convergence[convergence["num_cities"] == num_cities]
        fig, ax = plt.subplots(figsize=(6.5, 4.8))

        for algorithm, style in ALGORITHM_STYLE.items():
            algorithm_data = instance_data[instance_data["algorithm"] == algorithm]
            if algorithm_data.empty:
                continue
            mean_by_iteration = algorithm_data.groupby("iteration")["best_cost_so_far"].mean()
            ax.plot(
                mean_by_iteration.index,
                mean_by_iteration.to_numpy(),
                label=algorithm,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.4,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Iteração", fontsize=13)
        ax.set_ylabel("Custo médio (10 execuções, log)", fontsize=13)
        ax.set_title(f"n = {num_cities} cidades", color=TEXT_COLOR, fontsize=14)
        ax.tick_params(labelsize=11)
        ax.legend(frameon=False, fontsize=12)
        _style_axes(ax)

        output_path = output_dir / f"fig_convergence_n{num_cities}.pdf"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def plot_distributions(summary: pd.DataFrame, output_path: Path) -> plt.Figure:
    """Box plots da distribuição de custo por algoritmo e tamanho de instância (10 execuções)."""
    instance_sizes = sorted(summary["num_cities"].unique())
    fig, axes = plt.subplots(1, len(instance_sizes), figsize=(5 * len(instance_sizes), 5))

    if len(instance_sizes) == 1:
        axes = [axes]

    algorithms = list(ALGORITHM_STYLE.keys())

    for ax, n in zip(axes, instance_sizes):
        data_n = summary[summary["num_cities"] == n]

        # inclui apenas algoritmos com dados nesta instância, para que a figura
        # continue válida quando a CLI roda um subconjunto de algoritmos
        present = [alg for alg in algorithms if not data_n[data_n["algorithm"] == alg].empty]
        box_data = [data_n[data_n["algorithm"] == alg]["best_cost"].values for alg in present]
        positions = list(range(1, len(present) + 1))

        bp = ax.boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        )

        for patch, alg in zip(bp["boxes"], present):
            patch.set_facecolor(ALGORITHM_STYLE[alg]["color"])
            patch.set_alpha(0.75)

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(present)
        ax.set_title(f"n = {n} cidades", color=TEXT_COLOR)
        _style_axes(ax)

        if n == instance_sizes[0]:
            ax.set_ylabel("Custo (escala logarítmica)", color=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def main() -> None:
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    convergence = pd.read_csv(RESULTS_DIR / "convergence.csv")

    plot_scalability(summary, RESULTS_DIR / "fig_scalability.pdf")
    plot_convergence(convergence, RESULTS_DIR / "fig_convergence.pdf")
    plot_convergence_per_instance(convergence, RESULTS_DIR)
    plot_distributions(summary, RESULTS_DIR / "fig_distributions.pdf")
    plt.close("all")
    print("Gráficos salvos em results/")


if __name__ == "__main__":
    main()
