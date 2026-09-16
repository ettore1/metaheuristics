# PSO (Particle Swarm Optimization) com Random Keys para o TSPTW

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.problem import FloatArray, TSPTWInstance, evaluate_route, semi_greedy_route


@dataclass
class PSOParams:
    """Parâmetros do PSO."""

    num_particles: int = 30
    max_iterations: int = 400
    inertia_weight: float = 0.7
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    velocity_clamp: float = 0.5  # limite absoluto de velocidade, evita explosão numérica
    penalty_weight: float = 1000.0
    seeded_fraction: float = 0.3  # fração do enxame inicializada com construção semi-gulosa


@dataclass
class PSOResult:
    """Resultado de uma execução do PSO."""

    best_route: list[int]
    best_cost: float
    cost_history: list[float]


def decode_particle_position(position: FloatArray) -> list[int]:
    """Decodifica uma posição em uma rota via Random Keys (ranking/argsort).

    Esta é a adaptação do PSO ao domínio combinatório do TSPTW: cada dimensão
    da partícula é uma "chave aleatória" contínua associada a uma cidade. A
    ordem relativa dessas chaves — não seus valores absolutos — define a
    permutação resultante, o que permite usar as equações de velocidade e
    posição clássicas do PSO (definidas para espaços contínuos) diretamente
    sobre um problema de sequenciamento combinatório.
    """
    visiting_order = np.argsort(position)
    return (visiting_order + 1).tolist()


def _encode_route_as_random_keys(
    route: list[int],
    rng: np.random.Generator,
) -> FloatArray:
    # codifica uma rota como chaves aleatórias: a cidade na posição 'rank' da
    # rota recebe a chave (rank + ruído) / num_cidades, de forma que
    # decode_particle_position(keys) reproduz exatamente 'route'
    num_cities = len(route)
    noise = rng.uniform(0.0, 0.05, size=num_cities)
    keys = np.empty(num_cities)

    for rank, city in enumerate(route):
        keys[city - 1] = rank + noise[rank]

    return keys / num_cities


def _initialize_positions(
    instance: TSPTWInstance,
    num_particles: int,
    seeded_fraction: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Inicializa as posições do enxame.

    Uma fração das partículas é semeada com rotas construídas pela heurística
    semi-gulosa (a mesma usada na Busca Tabu), convertidas em random keys; as
    demais permanecem aleatórias, preservando diversidade exploratória.

    Essa semeadura se mostrou necessária na prática: um enxame 100%
    inicializado aleatoriamente não converge para soluções viáveis do TSPTW
    (mesmo após ajustes de parâmetros como número de partículas e velocidade),
    pois o espaço de chaves aleatórias puro raramente contém, por acaso,
    permutações que respeitem janelas de tempo apertadas. Partículas semeadas
    dão ao enxame um ponto de partida razoável ao redor do qual convergir.
    """
    num_cities = instance.num_cities
    positions = rng.uniform(0.0, 1.0, size=(num_particles, num_cities))
    num_seeded_particles = int(num_particles * seeded_fraction)

    for particle_index in range(num_seeded_particles):
        seed_route = semi_greedy_route(instance, rng)
        positions[particle_index] = _encode_route_as_random_keys(seed_route, rng)

    return positions


def particle_swarm_optimization(
    instance: TSPTWInstance,
    params: PSOParams,
    rng: np.random.Generator,
) -> PSOResult:
    """Executa o PSO com Random Keys para o TSPTW.

    Cada partícula é um vetor de reais (chaves aleatórias) decodificado em uma
    rota. As equações de velocidade/posição são as clássicas do PSO contínuo;
    somente a decodificação (posição -> rota) é específica do domínio
    combinatório.
    """
    num_cities = instance.num_cities

    positions = _initialize_positions(instance, params.num_particles, params.seeded_fraction, rng)
    velocities = rng.uniform(-1.0, 1.0, size=(params.num_particles, num_cities))

    personal_best_positions = positions.copy()
    personal_best_costs = np.array(
        [_route_cost(instance, position, params.penalty_weight) for position in positions]
    )

    global_best_index = int(np.argmin(personal_best_costs))
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = float(personal_best_costs[global_best_index])
    cost_history = [global_best_cost]

    for _ in range(params.max_iterations):
        positions, velocities = _update_swarm(
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            global_best_position=global_best_position,
            params=params,
            rng=rng,
        )

        for particle_index in range(params.num_particles):
            cost = _route_cost(instance, positions[particle_index], params.penalty_weight)

            if cost < personal_best_costs[particle_index]:
                personal_best_costs[particle_index] = cost
                personal_best_positions[particle_index] = positions[particle_index].copy()

                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = positions[particle_index].copy()

        cost_history.append(global_best_cost)

    global_best_route = decode_particle_position(global_best_position)
    return PSOResult(best_route=global_best_route, best_cost=global_best_cost, cost_history=cost_history)


def _route_cost(instance: TSPTWInstance, position: FloatArray, penalty_weight: float) -> float:
    route = decode_particle_position(position)
    return evaluate_route(instance, route, penalty_weight).total_cost


def _update_swarm(
    positions: FloatArray,
    velocities: FloatArray,
    personal_best_positions: FloatArray,
    global_best_position: FloatArray,
    params: PSOParams,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    num_particles, num_cities = positions.shape
    cognitive_random = rng.uniform(0.0, 1.0, size=(num_particles, num_cities))
    social_random = rng.uniform(0.0, 1.0, size=(num_particles, num_cities))

    new_velocities = (
        params.inertia_weight * velocities
        + params.cognitive_coefficient * cognitive_random * (personal_best_positions - positions)
        + params.social_coefficient * social_random * (global_best_position - positions)
    )
    new_velocities = np.clip(new_velocities, -params.velocity_clamp, params.velocity_clamp)
    new_positions = positions + new_velocities

    return new_positions, new_velocities
