# Otimização por Colônia de Formigas (ACO / Ant System) para o TSPTW

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.problem import FloatArray, TSPTWInstance, evaluate_route


@dataclass
class ACOParams:
    """Parâmetros do Ant System."""

    num_ants: int = 30
    max_iterations: int = 200
    alpha: float = 1.0  # peso do feromônio na regra de transição
    beta: float = 8.0  # peso da heurística (proximidade) na regra de transição
    evaporation_rate: float = 0.5  # rho: fração de feromônio evaporada por iteração
    pheromone_deposit: float = 1.0  # Q: constante de depósito de feromônio
    penalty_weight: float = 1000.0


@dataclass
class ACOResult:
    """Resultado de uma execução do ACO."""

    best_route: list[int]
    best_cost: float
    cost_history: list[float]


def ant_colony_optimization(
    instance: TSPTWInstance,
    params: ACOParams,
    rng: np.random.Generator,
) -> ACOResult:
    """Executa o Ant System para o TSPTW.

    Construção: cada formiga constrói uma rota partindo do depósito, escolhendo
    a próxima cidade probabilisticamente a partir do feromônio (tau) e da
    heurística de proximidade (eta = 1/distância).
    Atualização de feromônio: evaporação global seguida do depósito de cada
    formiga, proporcional à qualidade (1/custo) da rota construída.
    """
    num_points = instance.num_cities + 1
    heuristic = 1.0 / (instance.distance_matrix + np.finfo(float).eps)
    pheromone = np.ones((num_points, num_points))

    best_route: list[int] | None = None
    best_cost = math.inf
    cost_history: list[float] = []

    for _ in range(params.max_iterations):
        ant_routes = []
        ant_costs = []

        for _ant in range(params.num_ants):
            route = _construct_route(instance, pheromone, heuristic, params, rng)
            cost = evaluate_route(instance, route, params.penalty_weight).total_cost
            ant_routes.append(route)
            ant_costs.append(cost)

            if cost < best_cost:
                best_route = list(route)
                best_cost = cost

        _update_pheromone(pheromone, ant_routes, ant_costs, params)
        cost_history.append(best_cost)

    return ACOResult(best_route=best_route, best_cost=best_cost, cost_history=cost_history)


def _construct_route(
    instance: TSPTWInstance,
    pheromone: FloatArray,
    heuristic: FloatArray,
    params: ACOParams,
    rng: np.random.Generator,
) -> list[int]:
    # a cada passo, escolhe a próxima cidade entre as não visitadas com
    # probabilidade proporcional a (feromônio^alpha) * (heurística^beta)
    unvisited = set(range(1, instance.num_cities + 1))
    route: list[int] = []
    current_city = 0

    while unvisited:
        candidates = list(unvisited)
        attractiveness = np.array(
            [
                (pheromone[current_city, city] ** params.alpha) * (heuristic[current_city, city] ** params.beta)
                for city in candidates
            ]
        )
        probabilities = attractiveness / attractiveness.sum()

        next_city = int(rng.choice(candidates, p=probabilities))
        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return route


def _update_pheromone(
    pheromone: FloatArray,
    ant_routes: list[list[int]],
    ant_costs: list[float],
    params: ACOParams,
) -> None:
    pheromone *= 1.0 - params.evaporation_rate

    for route, cost in zip(ant_routes, ant_costs):
        deposit_amount = params.pheromone_deposit / cost
        full_path = [0, *route, 0]

        for from_city, to_city in zip(full_path[:-1], full_path[1:]):
            pheromone[from_city, to_city] += deposit_amount
            pheromone[to_city, from_city] += deposit_amount
