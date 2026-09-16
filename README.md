# TSPTW com PSO, Busca Tabu e ACO

Implementação e comparação de três metaheurísticas — Otimização por Enxame de
Partículas (PSO), Busca Tabu (TS) e Otimização por Colônia de Formigas (ACO) —
aplicadas ao Problema do Caixeiro Viajante com Janelas de Tempo (TSPTW).

Trabalho final da disciplina de Metaheurísticas e Aplicações.

## Requisitos

- **Python 3.11 ou superior** (desenvolvido e testado em Python 3.14).
- Dependências (listadas em [`requirements.txt`](requirements.txt)): `numpy`,
  `pandas`, `matplotlib` e `networkx`.

## Instalação

A partir da **raiz do repositório**, crie um ambiente virtual e instale as
dependências a partir do `requirements.txt`.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para sair do ambiente virtual ao terminar, execute `deactivate`. Se preferir não
usar um ambiente virtual, basta rodar `pip install -r requirements.txt`
diretamente.

## Estrutura do projeto

```
main.py                 CLI: roda metaheurísticas/instâncias selecionadas e gera os gráficos
instances/              instâncias do TSPTW usadas nos experimentos (.json)
src/
  problem.py             modelo do TSPTW: representação, custo, viabilidade
  instance_loader.py     geração e serialização de instâncias
  generate_instances.py  script que gera as 3 instâncias oficiais (pequena/média/grande)
  tabu_search.py         Busca Tabu (2-opt + Or-opt)
  aco.py                 Ant System (ACO)
  pso.py                 PSO com Random Keys
  experiment.py          framework de execução dos experimentos (10 execuções/algoritmo/instância)
  parameter_study.py     estudo do efeito de parâmetros-chave de cada metaheurística
  visualize.py           geração dos gráficos a partir dos resultados
  plot_example_instance.py  figura de exemplo (networkx) de uma instância do TSPTW
tests/                   testes de sanidade de cada módulo
results/                 CSVs e gráficos gerados pelos experimentos
report/                  relatório técnico em LaTeX
```

Todos os módulos em `src/` usam imports absolutos (`from src.problem import ...`),
portanto os comandos abaixo devem ser executados a partir da **raiz do
repositório**, com `src` no `PYTHONPATH` (os comandos `python -m src.<modulo>`
já cuidam disso automaticamente).

## Uso rápido (CLI)

A forma recomendada de rodar os experimentos é a interface de linha de comando
`main.py`, executada a partir da raiz do repositório:

```bash
python main.py                     # roda tudo (3 algoritmos x 3 instâncias, 10 execuções)
python main.py -a ts -n 25,50      # Busca Tabu nas instâncias n=25 e n=50
python main.py -a ts,aco -n 25,50  # TS e ACO nas instâncias n=25 e n=50
python main.py -a pso -n 100 -r 5  # PSO na instância n=100, 5 execuções
python main.py --no-plots          # roda tudo sem exibir os gráficos ao final
```

Opções (todas com valores padrão; sem argumentos, roda tudo):

| Opção | Alias | Valores | Padrão | Significado |
|---|---|---|---|---|
| `--algorithms` | `-a` | `ts`, `aco`, `pso` (separados por vírgula) | todos | metaheurísticas a rodar |
| `--sizes` | `-n` | `25`, `50`, `100` (separados por vírgula) | todos | tamanhos de instância |
| `--runs` | `-r` | inteiro ≥ 1 | 10 | execuções independentes por combinação |
| `--no-plots` | | — | desligado | não gerar nem exibir os gráficos |

Ao final, a CLI grava `results/summary.csv` e `results/convergence.csv` e, salvo
`--no-plots`, gera e **exibe** as figuras (escalabilidade, convergência e
distribuição) em `results/`, encerrando quando as janelas são fechadas. Rode
`python main.py --help` para a ajuda completa.

> **Atenção:** rodar um subconjunto sobrescreve `results/summary.csv`,
> `results/convergence.csv` e as figuras com os dados desse subconjunto. Para
> regenerar os resultados completos usados no relatório, rode `python main.py`
> sem argumentos.

## Execução por etapas (uso avançado)

### 1. Gerar as instâncias

```bash
python -m src.generate_instances
```

Gera as três instâncias oficiais (`n=25`, `n=50`, `n=100`) em `instances/`. Já
estão incluídas no repositório; rode este passo apenas se quiser regerá-las.

### 2. Rodar os testes de sanidade

```bash
python -m tests.test_problem
python -m tests.test_tabu_search
python -m tests.test_aco
python -m tests.test_pso
```

### 3. Rodar uma metaheurística isoladamente

```python
import numpy as np
from src.instance_loader import load_instance
from src.tabu_search import TabuSearchParams, tabu_search
from pathlib import Path

instance = load_instance(Path("instances/tsptw_small_n25.json"))
rng = np.random.default_rng(0)
result = tabu_search(instance, TabuSearchParams(), rng)

print(result.best_cost, result.best_route)
```

O mesmo padrão vale para `ant_colony_optimization` (`src/aco.py`) e
`particle_swarm_optimization` (`src/pso.py`), cada um com sua própria classe de
parâmetros (`ACOParams`, `PSOParams`).

### 4. Rodar a bateria completa de experimentos

```bash
python -m src.experiment
```

Executa cada algoritmo 10 vezes em cada uma das 3 instâncias, salvando
`results/summary.csv` (custo, distância, viabilidade e tempo por execução) e
`results/convergence.csv` (histórico de convergência por iteração). Os
resultados são salvos incrementalmente após cada algoritmo — a Busca Tabu na
instância grande é a etapa mais demorada (dezenas de minutos).

### 5. Rodar o estudo de parâmetros

```bash
python -m src.parameter_study
```

Testa variações de `tabu_tenure` (TS), `beta`/`evaporation_rate` (ACO) e
`seeded_fraction` (PSO) na instância média, salvando `results/parameter_study.csv`.

### 6. Gerar os gráficos

```bash
python -m src.visualize
```

Lê `results/summary.csv` e `results/convergence.csv` e gera
`results/fig_scalability.pdf` (custo, distância e tempo vs. tamanho da
instância) e `results/fig_convergence.pdf` (convergência por iteração).

## Principais parâmetros

| Algoritmo | Parâmetro | Padrão | Significado |
|---|---|---|---|
| TS | `tabu_tenure` | 15 | duração (em iterações) de um movimento na lista tabu |
| TS | `max_iterations` | 200 | número de iterações da busca |
| TS | `candidate_list_size` | 3 | tamanho da lista de candidatos na construção inicial semi-gulosa |
| ACO | `num_ants` | 30 | número de formigas por iteração |
| ACO | `alpha` | 1.0 | peso do feromônio na regra de transição |
| ACO | `beta` | 8.0 | peso da heurística de distância na regra de transição |
| ACO | `evaporation_rate` | 0.5 | fração do feromônio evaporada por iteração |
| PSO | `num_particles` | 30 | tamanho do enxame |
| PSO | `max_iterations` | 400 | número de iterações |
| PSO | `seeded_fraction` | 0.3 | fração do enxame inicializada via construção semi-gulosa |
| PSO | `inertia_weight`, `cognitive_coefficient`, `social_coefficient` | 0.7 / 1.5 / 1.5 | coeficientes clássicos do PSO |

Todos os parâmetros têm valor padrão nas respectivas classes
(`TabuSearchParams`, `ACOParams`, `PSOParams`) e podem ser sobrescritos na
criação do objeto, e.g. `ACOParams(beta=5.0, num_ants=50)`.

## Autor

Ettore Gabriel Braga
