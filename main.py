import argparse
import pandas as pd
from pandas.api.types import is_numeric_dtype
from config import Config
from genetic_algorithm import GeneticAlgorithm
from regression_evaluator import RegressionEvaluator

def load_data(file_path: str, target_name: str):
    data = pd.read_csv(file_path)
    if target_name not in data.columns:
        raise ValueError(f"No existe la columna: {target_name}")

    if not is_numeric_dtype(data[target_name]):
        raise ValueError("La variable objetivo debe ser numerica")
    variable_names = [
        column for column in data.columns if(column != target_name and is_numeric_dtype(data[column]))
    ]
    if not variable_names:
        raise ValueError("No existen variables predictoras")
    ignored_columns = [column for column in data.columns if(column != target_name and column not in variable_names)]
    model_data = data[variable_names + [target_name]].dropna()
    if len(model_data) < 3:
        raise ValueError("No existen sufucientes datos validos")
    x = model_data[variable_names].to_numpy(dtype=float)
    y = model_data[target_name].to_numpy(dtype=float)
    return x, y, variable_names, ignored_columns

def create_equation(target_name: str, variable_names: list[str], coefficients) -> str:
    equation = f"{target_name} = {coefficients[0]:.6f}"
    for variable_name, coefficient in zip(variable_names, coefficients[1:]):
        sign = "+" if coefficient >= 0 else "-"
        equation += (
            f" {sign} "
            f"{abs(coefficient):.6f}"
            f"({variable_name})"
        )
    return equation

def print_model(title: str, target_name: str, chromosome: list[int], result: dict):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Cromosoma: {chromosome}")
    print(
        "Variables seleccionadas: "
        f"{len(result['selected_names'])}"
    )
    for name in result["selected_names"]:
        print(f"  - {name}")
    equation = create_equation(target_name, result["selected_names"], result["coefficients"])

    print("\nEcuación:")
    print(equation)
    print("\nMetricas:")
    print(f"  BIC:          {result['bic']:.6f}")
    print(f"  RMSE:         {result['rmse']:.6f}")
    print(f"  R²:           {result['r_squared']:.6f}")
    print(
        "  R² ajustado:  "
        f"{result['adjusted_r_squared']:.6f}"
    )

def main():
    parser = argparse.ArgumentParser(description="Selección de variables mediante un algoritmo genetico")
    parser.add_argument("archivo", help="Archivo CSV que contiene los datos")
    parser.add_argument("objetivo", help="Nombre de la variable objetivo.")
    arguments = parser.parse_args()
    x, y, variable_names, ignored_columns = load_data(arguments.archivo, arguments.objetivo)

    chromosome_length = len(variable_names)
    if Config.MUTATION_PROBABILITY is None:
        mutation_probability = 1/chromosome_length
    else: 
        mutation_probability = (Config.MUTATION_PROBABILITY)
    evaluator = RegressionEvaluator(x, y, variable_names)
    algorithm = GeneticAlgorithm(
        chromosome_length=chromosome_length,
        population_size=Config.POPULATION_SIZE,
        generations=Config.GENERATIONS,
        crossover_probability=(Config.CROSSOVER_PROBABILITY),
        mutation_probability=mutation_probability,
        tournament_size=Config.TOURNAMENT_SIZE,
        evaluator=evaluator.evaluate,
        random_seed=Config.RANDOM_SEED
    )
    print("=" * 70)
    print("SELECCIÓN DE VARIABLES CON ALGORITMO GENÉTICO")
    print("=" * 70)
    print(f"Archivo: {arguments.archivo}")
    print(f"Variable objetivo: {arguments.objetivo}")
    print(f"Observaciones: {len(y)}")
    print(f"Variables candidatas: {chromosome_length}")
    print(f"Población: {Config.POPULATION_SIZE}")
    print(f"Generaciones: {Config.GENERATIONS}")
    print(
        "Probabilidad de cruce: "
        f"{Config.CROSSOVER_PROBABILITY}"
    )
    print(
        "Probabilidad de mutación: "
        f"{mutation_probability:.6f}"
    )

    if ignored_columns:
        print(
            "Columnas ignoradas: "
            + ", ".join(ignored_columns)
        )

    print("=" * 70)
    best_chromosome, best_bic, history = (algorithm.run())
    best_result = evaluator.fit(best_chromosome)
    print_model("Mejor modelo encontrado", arguments.objetivo, best_chromosome, best_result)
    full_chromosome = [1 for _ in variable_names]
    full_result = evaluator.fit(full_chromosome)
    print("\n" + "=" * 70)
    print("COMPARACIÓN")
    print("=" * 70)
    print(
        f"BIC del algoritmo genético: "
        f"{best_result['bic']:.6f}"
    )
    print(
        f"BIC del modelo completo:    "
        f"{full_result['bic']:.6f}"
    )
    print(
        "Variables eliminadas:        "
        f"{len(variable_names) - sum(best_chromosome)}"
    )
    if best_result["bic"] < full_result["bic"]:
        print(
            "Resultado: el algoritmo encontró "
            "un modelo más conveniente según el BIC."
        )
    else:
        print(
            "Resultado: el modelo completo obtuvo "
            "el menor BIC."
        )
    print("=" * 70)


if __name__ == "__main__":
    main()