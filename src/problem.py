# Modelo do problema TSPTW (Traveling Salesman Problem with Time Windows)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class TSPTWInstance:
    """Instância do TSPTW. O índice 0 é sempre o depósito (início/fim da rota)."""

    name: str
    coordinates: FloatArray
    distance_matrix: FloatArray
    ready_time: FloatArray
    due_time: FloatArray
    service_time: FloatArray

    @property
    def num_cities(self) -> int:
        # cidades a visitar, sem contar o depósito
        return len(self.coordinates) - 1


@dataclass
class RouteEvaluation:
    """Resultado da avaliação de custo/viabilidade de uma rota."""

    total_cost: float
    total_distance: float
    total_penalty: float
    is_feasible: bool


def evaluate_route(
    instance: TSPTWInstance,
    route: list[int],
    penalty_weight: float = 1000.0,
) -> RouteEvaluation:
    """Avalia uma rota (permutação das cidades 1..n, sem incluir o depósito).

    O trajeto completo percorrido é depósito -> route -> depósito. Chegar
    antes da janela de tempo gera espera (sem custo); chegar depois de
    'due_time' é penalizado proporcionalmente ao atraso, em vez de tornar a
    solução inteira descartável. Isso permite que as metaheurísticas também
    naveguem por soluções levemente inviáveis durante a busca.
    """
    full_path = [0, *route, 0]
    total_distance = 0.0
    total_penalty = 0.0
    current_time = 0.0

    for from_city, to_city in zip(full_path[:-1], full_path[1:]):
        travel_time = instance.distance_matrix[from_city, to_city]
        arrival_time = current_time + travel_time
        total_distance += travel_time

        wait_time = max(0.0, instance.ready_time[to_city] - arrival_time)
        service_start_time = arrival_time + wait_time
        lateness = max(0.0, service_start_time - instance.due_time[to_city])
        total_penalty += lateness * penalty_weight

        current_time = service_start_time + instance.service_time[to_city]

    return RouteEvaluation(
        total_cost=total_distance + total_penalty,
        total_distance=total_distance,
        total_penalty=total_penalty,
        is_feasible=total_penalty == 0.0,
    )


def semi_greedy_route(
    instance: TSPTWInstance,
    rng: np.random.Generator,
    candidate_list_size: int = 3,
) -> list[int]:
    """Constrói uma rota semi-gulosa (estilo GRASP): a cada passo, escolhe
    aleatoriamente entre as 'candidate_list_size' cidades não visitadas mais
    próximas da cidade atual.

    Serve como solução inicial para buscas locais (e.g. Tabu Search): produz
    soluções diversas entre execuções (mantendo a natureza estocástica exigida
    na análise experimental), mas evita o ponto de partida muito ruim de uma
    permutação totalmente aleatória, que pode aprisionar a busca local em um
    ótimo local de baixa qualidade.
    """
    unvisited = set(range(1, instance.num_cities + 1))
    route: list[int] = []
    current_city = 0

    while unvisited:
        nearest_candidates = sorted(unvisited, key=lambda city: instance.distance_matrix[current_city, city])
        candidate_list = nearest_candidates[:candidate_list_size]
        next_city = candidate_list[rng.integers(len(candidate_list))]

        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return route
