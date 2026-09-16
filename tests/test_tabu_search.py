# Testes de sanidade para a Busca Tabu
#
# Uso: python -m tests.test_tabu_search

import numpy as np

from src.instance_loader import generate_tsptw_instance
from src.tabu_search import TabuSearchParams, tabu_search


def test_tabu_search_returns_valid_permutation() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = TabuSearchParams(tabu_tenure=5, max_iterations=30)

    result = tabu_search(instance, params, rng)

    assert sorted(result.best_route) == list(range(1, 16))


def test_tabu_search_improves_over_random_start() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = TabuSearchParams(tabu_tenure=5, max_iterations=50)

    result = tabu_search(instance, params, rng)

    assert result.best_cost <= result.cost_history[0]
    assert result.cost_history[-1] == result.best_cost


def main() -> None:
    test_tabu_search_returns_valid_permutation()
    test_tabu_search_improves_over_random_start()
    print("Todos os testes de sanidade da Busca Tabu passaram.")


if __name__ == "__main__":
    main()
