# Testes de sanidade para o PSO com Random Keys
#
# Uso: python -m tests.test_pso

import numpy as np

from src.instance_loader import generate_tsptw_instance
from src.pso import PSOParams, decode_particle_position, particle_swarm_optimization


def test_decode_particle_position_is_a_valid_permutation() -> None:
    rng = np.random.default_rng(0)
    position = rng.uniform(0.0, 1.0, size=15)

    route = decode_particle_position(position)

    assert sorted(route) == list(range(1, 16))


def test_pso_returns_valid_permutation() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = PSOParams(num_particles=10, max_iterations=20)

    result = particle_swarm_optimization(instance, params, rng)

    assert sorted(result.best_route) == list(range(1, 16))


def test_pso_cost_history_is_non_increasing() -> None:
    instance = generate_tsptw_instance(num_cities=15, seed=7)
    rng = np.random.default_rng(0)
    params = PSOParams(num_particles=10, max_iterations=20)

    result = particle_swarm_optimization(instance, params, rng)

    assert all(
        later <= earlier
        for earlier, later in zip(result.cost_history[:-1], result.cost_history[1:])
    )
    assert result.cost_history[-1] == result.best_cost


def main() -> None:
    test_decode_particle_position_is_a_valid_permutation()
    test_pso_returns_valid_permutation()
    test_pso_cost_history_is_non_increasing()
    print("Todos os testes de sanidade do PSO passaram.")


if __name__ == "__main__":
    main()
