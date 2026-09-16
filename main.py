# main.py — CLI para rodar as metaheurísticas (TS, ACO, PSO) no TSPTW
#
# Exemplos:
#   python main.py                      roda tudo (3 algoritmos x 3 instâncias, 10 execuções)
#   python main.py -a ts -n 25,50       Busca Tabu nas instâncias n=25 e n=50
#   python main.py -a ts,aco -n 25,50   TS e ACO nas instâncias n=25 e n=50
#   python main.py -a pso -n 100 -r 5   PSO na instância n=100, 5 execuções
#   python main.py --no-plots           roda tudo sem exibir os gráficos ao final

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from src.experiment import ALGORITHM_RUNNERS, RESULTS_DIR, SIZE_TO_FILE, run_experiments
from src.visualize import plot_convergence, plot_distributions, plot_scalability

VALID_ALGORITHMS = list(ALGORITHM_RUNNERS)  # ["TS", "ACO", "PSO"]
VALID_SIZES = sorted(SIZE_TO_FILE)          # [25, 50, 100]


def parse_algorithms(raw: str | None) -> list[str]:
    """Converte 'ts,aco' na lista canônica ['TS', 'ACO']; None significa todos."""
    if raw is None:
        return list(VALID_ALGORITHMS)

    chosen = {token.strip().upper() for token in raw.split(",") if token.strip()}
    invalid = sorted(a for a in chosen if a not in ALGORITHM_RUNNERS)
    if invalid:
        raise ValueError(
            f"algoritmo(s) inválido(s): {', '.join(invalid)}. "
            f"Opções: {', '.join(VALID_ALGORITHMS)}"
        )
    # preserva a ordem canônica TS, ACO, PSO independentemente da ordem digitada
    return [a for a in VALID_ALGORITHMS if a in chosen]


def parse_sizes(raw: str | None) -> list[int]:
    """Converte '25,50' na lista [25, 50]; None significa todos."""
    if raw is None:
        return list(VALID_SIZES)

    chosen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            chosen.add(int(token))
        except ValueError:
            raise ValueError(f"tamanho inválido: {token!r} (use inteiros como 25, 50, 100)")

    invalid = sorted(s for s in chosen if s not in SIZE_TO_FILE)
    if invalid:
        raise ValueError(f"tamanho(s) inválido(s): {invalid}. Opções: {VALID_SIZES}")
    return sorted(chosen)


def show_plots(summary, convergence) -> None:
    """Gera as figuras a partir dos resultados desta execução, salva em results/ e as exibe.

    Cada figura é protegida individualmente: uma eventual falha (por exemplo,
    dados insuficientes em um subconjunto) apenas emite um aviso e não interrompe
    as demais.
    """
    figures = [
        ("escalabilidade", lambda: plot_scalability(summary, RESULTS_DIR / "fig_scalability.pdf")),
        ("convergência", lambda: plot_convergence(convergence, RESULTS_DIR / "fig_convergence.pdf")),
        ("distribuição", lambda: plot_distributions(summary, RESULTS_DIR / "fig_distributions.pdf")),
    ]

    generated = 0
    for name, make in figures:
        try:
            make()
            generated += 1
        except Exception as exc:  # noqa: BLE001 - não abortar as demais figuras
            print(f"[aviso] não foi possível gerar a figura de {name}: {exc}", file=sys.stderr)

    if generated:
        print(f"\n{generated} figura(s) salva(s) em {RESULTS_DIR}. Exibindo (feche as janelas para encerrar)...")
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Roda as metaheurísticas (TS, ACO, PSO) no TSPTW e gera os gráficos comparativos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  python main.py                     roda tudo (3 algoritmos x 3 instâncias)\n"
            "  python main.py -a ts -n 25,50      Busca Tabu em n=25 e n=50\n"
            "  python main.py -a ts,aco -n 25,50  TS e ACO em n=25 e n=50\n"
            "  python main.py -a pso -n 100 -r 5  PSO em n=100, 5 execuções\n"
            "  python main.py --no-plots          roda tudo sem exibir os gráficos"
        ),
    )
    parser.add_argument(
        "-a", "--algorithms", metavar="LISTA", default=None,
        help=f"algoritmos separados por vírgula ({', '.join(VALID_ALGORITHMS)}); padrão: todos",
    )
    parser.add_argument(
        "-n", "--sizes", metavar="LISTA", default=None,
        help=f"tamanhos de instância separados por vírgula ({', '.join(map(str, VALID_SIZES))}); padrão: todos",
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=10,
        help="número de execuções independentes por combinação (padrão: 10)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="não gerar nem exibir os gráficos ao final",
    )
    args = parser.parse_args()

    try:
        algorithms = parse_algorithms(args.algorithms)
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.runs < 1:
        parser.error("--runs deve ser >= 1")

    print(
        f"Rodando {', '.join(algorithms)} em n={', '.join(map(str, sizes))} "
        f"({args.runs} execução(ões) por combinação).\n"
    )
    summary, convergence = run_experiments(algorithms=algorithms, sizes=sizes, num_runs=args.runs)

    if not args.no_plots:
        show_plots(summary, convergence)


if __name__ == "__main__":
    main()
