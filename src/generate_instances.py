# Gera e salva as instâncias de TSPTW (pequena/média/grande) usadas nos experimentos
#
# Uso: python -m src.generate_instances

from pathlib import Path

from src.instance_loader import (
    _nearest_neighbor_route,
    generate_tsptw_instance,
    save_instance,
)
from src.problem import evaluate_route

INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"

INSTANCE_SPECS = [
    {"size_label": "small", "num_cities": 25, "seed": 1},
    {"size_label": "medium", "num_cities": 50, "seed": 2},
    {"size_label": "large", "num_cities": 100, "seed": 3},
]


def main() -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    for spec in INSTANCE_SPECS:
        instance_name = f"tsptw_{spec['size_label']}_n{spec['num_cities']}"
        instance = generate_tsptw_instance(
            num_cities=spec["num_cities"],
            seed=spec["seed"],
            name=instance_name,
        )

        filepath = INSTANCES_DIR / f"{instance_name}.json"
        save_instance(instance, filepath)

        # sanidade: a rota de referência usada para construir as janelas de
        # tempo deve ser viável e sem penalidade, por construção
        reference_route = _nearest_neighbor_route(instance.distance_matrix)[1:]
        evaluation = evaluate_route(instance, reference_route)
        assert evaluation.is_feasible, f"{instance_name}: rota de referência deveria ser viável"

        print(
            f"{instance_name}: salva em {filepath.name} "
            f"(custo rota de referência={evaluation.total_distance:.2f}, "
            f"viável={evaluation.is_feasible})"
        )


if __name__ == "__main__":
    main()
