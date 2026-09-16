# Estudo do efeito de parâmetros-chave de cada metaheurística no desempenho
# (seção "Parâmetros" da análise experimental exigida pelo trabalho)
#
# Uso: python -m src.parameter_study

from __future__ import annotations

import time
from collections.abc import Callable
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

NUM_RUNS_PER_CONFIG = 5
# tamanho intermediário: rápido o suficiente para testar várias configurações,
# ainda grande o bastante para as diferenças entre parâmetros aparecerem
STUDY_INSTANCE_FILE = "tsptw_medium_n50.json"

AlgorithmRunner = Callable[[TSPTWInstance, np.random.Generator], object]


def _evaluate_config(
    runner: AlgorithmRunner,
    instance: TSPTWInstance,
    num_runs: int,
) -> tuple[float, float, float]:
    # retorna (custo_medio, custo_desvio_padrao, tempo_medio) de uma configuração
    costs = []
    elapsed_times = []

    for run_index in range(num_runs):
        rng = np.random.default_rng(run_index)
        start_time = time.perf_counter()
        result = runner(instance, rng)
        elapsed_times.append(time.perf_counter() - start_time)
        costs.append(evaluate_route(instance, result.best_route).total_cost)

    return float(np.mean(costs)), float(np.std(costs)), float(np.mean(elapsed_times))


def _study_parameter(
    algorithm_name: str,
    parameter_name: str,
    values: list,
    make_runner: Callable[..., AlgorithmRunner],
    instance: TSPTWInstance,
) -> pd.DataFrame:
    rows = []

    for value in values:
        cost_mean, cost_std, time_mean = _evaluate_config(
            make_runner(value), instance, NUM_RUNS_PER_CONFIG
        )
        rows.append(
            {
                "algorithm": algorithm_name,
                "parametro": parameter_name,
                "valor": value,
                "custo_medio": cost_mean,
                "custo_desvio_padrao": cost_std,
                "tempo_medio_segundos": time_mean,
            }
        )

    return pd.DataFrame(rows)


def study_tabu_search(instance: TSPTWInstance) -> pd.DataFrame:
    return _study_parameter(
        algorithm_name="TS",
        parameter_name="tabu_tenure",
        values=[5, 15, 30],
        make_runner=lambda tenure: (
            lambda inst, rng: tabu_search(inst, TabuSearchParams(tabu_tenure=tenure), rng)
        ),
        instance=instance,
    )


def study_aco(instance: TSPTWInstance) -> pd.DataFrame:
    beta_study = _study_parameter(
        algorithm_name="ACO",
        parameter_name="beta",
        values=[3.0, 5.0, 8.0],
        make_runner=lambda beta: (
            lambda inst, rng: ant_colony_optimization(inst, ACOParams(beta=beta), rng)
        ),
        instance=instance,
    )
    evaporation_study = _study_parameter(
        algorithm_name="ACO",
        parameter_name="evaporation_rate",
        values=[0.3, 0.5, 0.7],
        make_runner=lambda rate: (
            lambda inst, rng: ant_colony_optimization(inst, ACOParams(evaporation_rate=rate), rng)
        ),
        instance=instance,
    )
    return pd.concat([beta_study, evaporation_study], ignore_index=True)


def study_pso(instance: TSPTWInstance) -> pd.DataFrame:
    return _study_parameter(
        algorithm_name="PSO",
        parameter_name="seeded_fraction",
        values=[0.0, 0.3, 0.6],
        make_runner=lambda fraction: (
            lambda inst, rng: particle_swarm_optimization(inst, PSOParams(seeded_fraction=fraction), rng)
        ),
        instance=instance,
    )


def main() -> None:
    instance = load_instance(INSTANCES_DIR / STUDY_INSTANCE_FILE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.concat(
        [study_tabu_search(instance), study_aco(instance), study_pso(instance)],
        ignore_index=True,
    )
    results.to_csv(RESULTS_DIR / "parameter_study.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
