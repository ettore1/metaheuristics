# Geração, persistência e carregamento de instâncias do TSPTW

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.problem import FloatArray, TSPTWInstance


def build_distance_matrix(coordinates: FloatArray) -> FloatArray:
    """Matriz de distâncias euclidianas entre todos os pares de pontos."""
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def generate_tsptw_instance(
    num_cities: int,
    seed: int,
    grid_size: float = 100.0,
    time_window_width: float = 40.0,
    name: str | None = None,
) -> TSPTWInstance:
    """Gera uma instância sintética de TSPTW no estilo Solomon (1987).

    Metodologia (padrão na literatura de TSPTW, e.g. Da Silva & Urrutia, 2010):
    1. Sorteia coordenadas aleatórias para o depósito e as cidades em um grid 2D.
    2. Constrói uma rota de referência via heurística do vizinho mais próximo.
    3. Centraliza a janela de tempo de cada cidade no horário de chegada dessa
       rota de referência, com largura 'time_window_width'.

    Isso garante que toda instância gerada tenha ao menos uma rota viável
    conhecida (a própria rota de referência), evitando instâncias degeneradas.
    """
    rng = np.random.default_rng(seed)
    num_points = num_cities + 1
    coordinates = rng.uniform(0.0, grid_size, size=(num_points, 2))
    distance_matrix = build_distance_matrix(coordinates)

    # rota fechada de referência (inclui o retorno ao depósito) para calcular
    # os tempos de chegada que servirão de base às janelas de tempo
    reference_route = _nearest_neighbor_route(distance_matrix)
    closed_route = [*reference_route, 0]
    arrival_times = _compute_arrival_times(distance_matrix, closed_route)

    half_width = time_window_width / 2.0
    ready_time = np.zeros(num_points)
    due_time = np.zeros(num_points)

    # janela de cada cidade centrada no horário de chegada da rota de referência
    for city, arrival_time in zip(closed_route[1:-1], arrival_times[:-1]):
        ready_time[city] = max(0.0, arrival_time - half_width)
        due_time[city] = arrival_time + half_width

    # depósito: sempre aberto a partir de t=0, fecha após o retorno do veículo
    return_arrival_time = arrival_times[-1]
    due_time[0] = return_arrival_time + half_width
    service_time = np.zeros(num_points)

    return TSPTWInstance(
        name=name or f"synthetic_n{num_cities}_seed{seed}",
        coordinates=coordinates,
        distance_matrix=distance_matrix,
        ready_time=ready_time,
        due_time=due_time,
        service_time=service_time,
    )


def _nearest_neighbor_route(distance_matrix: FloatArray) -> list[int]:
    # rota gulosa: a partir da cidade atual, visita sempre a mais próxima ainda não visitada
    num_points = len(distance_matrix)
    visited = {0}
    route = [0]
    current_city = 0

    while len(visited) < num_points:
        unvisited = [city for city in range(num_points) if city not in visited]
        nearest_city = min(unvisited, key=lambda city: distance_matrix[current_city, city])
        route.append(nearest_city)
        visited.add(nearest_city)
        current_city = nearest_city

    return route


def _compute_arrival_times(distance_matrix: FloatArray, route: list[int]) -> list[float]:
    # tempo de chegada acumulado em cada cidade da rota (exclui o depósito em t=0)
    arrival_times = []
    current_time = 0.0

    for from_city, to_city in zip(route[:-1], route[1:]):
        current_time += distance_matrix[from_city, to_city]
        arrival_times.append(current_time)

    return arrival_times


def save_instance(instance: TSPTWInstance, filepath: Path) -> None:
    """Salva a instância em formato JSON, para reprodutibilidade dos experimentos."""
    payload = {
        "name": instance.name,
        "coordinates": instance.coordinates.tolist(),
        "ready_time": instance.ready_time.tolist(),
        "due_time": instance.due_time.tolist(),
        "service_time": instance.service_time.tolist(),
    }
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_instance(filepath: Path) -> TSPTWInstance:
    """Carrega uma instância previamente salva com 'save_instance'."""
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    coordinates = np.array(payload["coordinates"], dtype=np.float64)

    return TSPTWInstance(
        name=payload["name"],
        coordinates=coordinates,
        distance_matrix=build_distance_matrix(coordinates),
        ready_time=np.array(payload["ready_time"], dtype=np.float64),
        due_time=np.array(payload["due_time"], dtype=np.float64),
        service_time=np.array(payload["service_time"], dtype=np.float64),
    )
