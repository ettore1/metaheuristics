# Framework de execução dos experimentos: roda cada metaheurística N vezes por
# instância, mede tempo de CPU e salva os resultados para a análise comparativa

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.aco import ACOParams, ant_colony_optimization
from src.instance_loader import load_instance
from src.problem import TSPTWInstance, evaluate_route
from src.pso import PSOParams, particle_swarm_optimization
from src.tabu_search import TabuSearchParams, tabu_search

INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

INSTANCE_FILES = [
    "tsptw_small_n25.json",
    "tsptw_medium_n50.json",
    "tsptw_large_n100.json",
]

# mapeia o número de cidades ao arquivo da instância correspondente, usado pela
# CLI (main.py) para selecionar instâncias por tamanho
SIZE_TO_FILE: dict[int, str] = {
    25: "tsptw_small_n25.json",
    50: "tsptw_medium_n50.json",
    100: "tsptw_large_n100.json",
}

# cada algoritmo é uma função (instância, rng) -> resultado com
# .best_route, .best_cost e .cost_history (mesma interface nos três módulos)
AlgorithmRunner = Callable[[TSPTWInstance, np.random.Generator], object]

ALGORITHM_RUNNERS: dict[str, AlgorithmRunner] = {
    "TS": lambda instance, rng: tabu_search(instance, TabuSearchParams(), rng),
    "ACO": lambda instance, rng: ant_colony_optimization(instance, ACOParams(), rng),
    "PSO": lambda instance, rng: particle_swarm_optimization(instance, PSOParams(), rng),
}


@dataclass
class RunRecord:
    """Métricas de uma única execução de um algoritmo em uma instância."""

    algorithm: str
    instance_name: str
    num_cities: int
    run_index: int
    best_cost: float
    total_distance: float
    is_feasible: bool
    elapsed_seconds: float


def run_single(
    algorithm_name: str,
    instance: TSPTWInstance,
    run_index: int,
    penalty_weight: float = 1000.0,
) -> tuple[RunRecord, list[float]]:
    """Executa uma metaheurística uma vez; retorna o registro de métricas e o histórico de convergência."""
    rng = np.random.default_rng(run_index)
    runner = ALGORITHM_RUNNERS[algorithm_name]

    start_time = time.perf_counter()
    result = runner(instance, rng)
    elapsed_seconds = time.perf_counter() - start_time

    evaluation = evaluate_route(instance, result.best_route, penalty_weight)
    record = RunRecord(
        algorithm=algorithm_name,
        instance_name=instance.name,
        num_cities=instance.num_cities,
        run_index=run_index,
        best_cost=result.best_cost,
        total_distance=evaluation.total_distance,
        is_feasible=evaluation.is_feasible,
        elapsed_seconds=elapsed_seconds,
    )
    return record, result.cost_history


def run_experiments(
    algorithms: list[str] | None = None,
    sizes: list[int] | None = None,
    num_runs: int = 10,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda os algoritmos selecionados nas instâncias selecionadas, 'num_runs' vezes cada.

    'algorithms' é uma lista de nomes em ALGORITHM_RUNNERS (padrão: todos);
    'sizes' é uma lista de números de cidades em SIZE_TO_FILE (padrão: todos).
    Quando 'save' é verdadeiro, os resultados são salvos incrementalmente em
    'results/summary.csv' e 'results/convergence.csv' (sobrescritos ao final de
    cada algoritmo), para que o progresso não se perca caso a execução, que pode
    levar dezenas de minutos para a Busca Tabu na instância grande, seja
    interrompida no meio.
    """
    algorithms = algorithms or list(ALGORITHM_RUNNERS)
    sizes = sizes or sorted(SIZE_TO_FILE)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    convergence_rows: list[dict] = []

    for size in sizes:
        instance = load_instance(INSTANCES_DIR / SIZE_TO_FILE[size])

        for algorithm_name in algorithms:
            for run_index in range(num_runs):
                record, cost_history = run_single(algorithm_name, instance, run_index)
                summary_rows.append(vars(record))

                for iteration, cost in enumerate(cost_history):
                    convergence_rows.append(
                        {
                            "algorithm": algorithm_name,
                            "instance_name": instance.name,
                            "num_cities": instance.num_cities,
                            "run_index": run_index,
                            "iteration": iteration,
                            "best_cost_so_far": cost,
                        }
                    )

                print(
                    f"{instance.name} | {algorithm_name} | run {run_index}: "
                    f"custo={record.best_cost:.2f}, viável={record.is_feasible}, "
                    f"tempo={record.elapsed_seconds:.2f}s"
                )

            if save:
                # salva incrementalmente após cada algoritmo terminar suas execuções
                pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary.csv", index=False)
                pd.DataFrame(convergence_rows).to_csv(RESULTS_DIR / "convergence.csv", index=False)

    return pd.DataFrame(summary_rows), pd.DataFrame(convergence_rows)


def run_all_experiments(num_runs: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda todos os algoritmos em todas as instâncias 'num_runs' vezes cada."""
    return run_experiments(algorithms=None, sizes=None, num_runs=num_runs, save=True)


if __name__ == "__main__":
    run_all_experiments(num_runs=10)
