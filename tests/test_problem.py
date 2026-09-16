# Testes de sanidade para o modelo do problema e geração de instâncias
#
# Uso: python -m tests.test_problem

import numpy as np

from src.instance_loader import _nearest_neighbor_route, generate_tsptw_instance
from src.problem import evaluate_route, semi_greedy_route


def test_reference_route_is_feasible() -> None:
    instance = generate_tsptw_instance(num_cities=20, seed=42)
    reference_route = _nearest_neighbor_route(instance.distance_matrix)[1:]

    evaluation = evaluate_route(instance, reference_route)

    assert evaluation.is_feasible, "rota de referência deveria ser viável por construção"
    assert evaluation.total_penalty == 0.0


def test_semi_greedy_route_is_a_valid_permutation() -> None:
    instance = generate_tsptw_instance(num_cities=20, seed=42)
    rng = np.random.default_rng(0)
    route = semi_greedy_route(instance, rng)

    assert sorted(route) == list(range(1, 21))


def test_evaluate_route_detects_time_window_violation() -> None:
    instance = generate_tsptw_instance(num_cities=20, seed=42, time_window_width=1.0)
    reversed_route = list(range(20, 0, -1))

    evaluation = evaluate_route(instance, reversed_route)

    assert evaluation.total_penalty > 0.0
    assert not evaluation.is_feasible


def main() -> None:
    test_reference_route_is_feasible()
    test_semi_greedy_route_is_a_valid_permutation()
    test_evaluate_route_detects_time_window_violation()
    print("Todos os testes de sanidade passaram.")


if __name__ == "__main__":
    main()
