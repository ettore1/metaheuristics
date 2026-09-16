# Busca Tabu (Tabu Search) para o TSPTW, com vizinhança 2-opt + Or-opt e lista tabu

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.problem import TSPTWInstance, evaluate_route, semi_greedy_route

MoveAttribute = frozenset[int]


@dataclass
class TabuSearchParams:
    """Parâmetros da Busca Tabu."""

    tabu_tenure: int = 15
    max_iterations: int = 200
    penalty_weight: float = 1000.0
    candidate_list_size: int = 3  # tamanho da lista de candidatos na construção inicial semi-gulosa


@dataclass
class TabuSearchResult:
    """Resultado de uma execução da Busca Tabu."""

    best_route: list[int]
    best_cost: float
    cost_history: list[float]


def tabu_search(
    instance: TSPTWInstance,
    params: TabuSearchParams,
    rng: np.random.Generator,
) -> TabuSearchResult:
    """Executa a Busca Tabu para o TSPTW.

    Vizinhança: combina movimentos 2-opt (inversão de segmentos) e Or-opt
    (realocação de uma única cidade para outra posição da rota). O Or-opt é
    essencial aqui: ao contrário do 2-opt, ele não inverte a ordem de visita
    das cidades dentro de um segmento, o que evita conflitos desnecessários
    com a sequência exigida pelas janelas de tempo.
    Memória de curto prazo: lista tabu que proíbe desfazer movimentos
    recentes por 'tabu_tenure' iterações, com critério de aspiração (aceita
    um movimento tabu se ele resultar na melhor solução já encontrada).
    Solução inicial: construção semi-gulosa (ver 'semi_greedy_route'), que
    evita aprisionar a busca local em um ótimo local de baixa qualidade logo
    na primeira iteração.
    """
    current_route = semi_greedy_route(instance, rng, params.candidate_list_size)
    current_cost = evaluate_route(instance, current_route, params.penalty_weight).total_cost

    best_route = list(current_route)
    best_cost = current_cost
    cost_history = [best_cost]

    # mapeia um atributo de movimento para a iteração em que deixa de ser tabu
    tabu_expiration: dict[MoveAttribute, int] = {}

    for iteration in range(params.max_iterations):
        neighbor_route, neighbor_cost, move_attribute = _best_admissible_neighbor(
            instance=instance,
            route=current_route,
            tabu_expiration=tabu_expiration,
            current_iteration=iteration,
            best_cost=best_cost,
            penalty_weight=params.penalty_weight,
        )

        if neighbor_route is None:
            break  # toda a vizinhança está tabu (raro em prática)

        current_route = neighbor_route
        current_cost = neighbor_cost
        tabu_expiration[move_attribute] = iteration + params.tabu_tenure

        if current_cost < best_cost:
            best_route = list(current_route)
            best_cost = current_cost

        cost_history.append(best_cost)

    return TabuSearchResult(best_route=best_route, best_cost=best_cost, cost_history=cost_history)


def _best_admissible_neighbor(
    instance: TSPTWInstance,
    route: list[int],
    tabu_expiration: dict[MoveAttribute, int],
    current_iteration: int,
    best_cost: float,
    penalty_weight: float,
) -> tuple[list[int] | None, float, MoveAttribute | None]:
    # avalia toda a vizinhança (2-opt + Or-opt) e retorna o melhor vizinho
    # não-tabu (ou tabu, mas aspirado por superar o recorde global)
    best_neighbor_route: list[int] | None = None
    best_neighbor_cost = math.inf
    best_neighbor_attribute: MoveAttribute | None = None

    for move_attribute, neighbor_route in _candidate_moves(route):
        is_tabu = tabu_expiration.get(move_attribute, -1) > current_iteration

        neighbor_cost = evaluate_route(instance, neighbor_route, penalty_weight).total_cost
        is_aspirated = neighbor_cost < best_cost

        if is_tabu and not is_aspirated:
            continue

        if neighbor_cost < best_neighbor_cost:
            best_neighbor_route = neighbor_route
            best_neighbor_cost = neighbor_cost
            best_neighbor_attribute = move_attribute

    return best_neighbor_route, best_neighbor_cost, best_neighbor_attribute


def _candidate_moves(route: list[int]) -> Iterator[tuple[MoveAttribute, list[int]]]:
    yield from _two_opt_neighborhood(route)
    yield from _or_opt_neighborhood(route)


def _two_opt_neighborhood(route: list[int]) -> Iterator[tuple[MoveAttribute, list[int]]]:
    """Vizinhos 2-opt: inverte o segmento route[i:j+1] para cada par i<j."""
    num_cities = len(route)

    for i in range(num_cities - 1):
        for j in range(i + 1, num_cities):
            neighbor_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
            move_attribute = frozenset({route[i], route[j]})
            yield move_attribute, neighbor_route


def _or_opt_neighborhood(route: list[int]) -> Iterator[tuple[MoveAttribute, list[int]]]:
    """Vizinhos Or-opt: remove a cidade da posição i e a reinsere em outra posição."""
    num_cities = len(route)

    for i in range(num_cities):
        relocated_city = route[i]
        remaining_route = route[:i] + route[i + 1 :]

        for j in range(len(remaining_route) + 1):
            if j == i:
                continue  # mesma posição original: não é um movimento de fato

            neighbor_route = remaining_route[:j] + [relocated_city] + remaining_route[j:]
            move_attribute = frozenset({relocated_city})
            yield move_attribute, neighbor_route
