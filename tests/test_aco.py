# Testes de sanidade para o ACO (Ant System)
#
# Uso: python -m tests.test_aco

import numpy as np

from src.aco import ACOParams, ant_colony_optimization
from src.instance_loader import generate_tsptw_instance


def test_aco_returns_valid_permutation() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = ACOParams(num_ants=10, max_iterations=10)

    result = ant_colony_optimization(instance, params, rng)

    assert sorted(result.best_route) == list(range(1, 16))


def test_aco_cost_history_is_non_increasing() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = ACOParams(num_ants=10, max_iterations=15)

    result = ant_colony_optimization(instance, params, rng)

    assert all(
        later <= earlier
        for earlier, later in zip(result.cost_history[:-1], result.cost_history[1:])
    )
    assert result.cost_history[-1] == result.best_cost


def main() -> None:
    test_aco_returns_valid_permutation()
    test_aco_cost_history_is_non_increasing()
    print("Todos os testes de sanidade do ACO passaram.")


if __name__ == "__main__":
    main()
