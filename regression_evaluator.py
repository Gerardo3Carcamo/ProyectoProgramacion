import math
import numpy as np

class RegressionEvaluator:

    def __init__(self, x: np.ndarray, y: np.ndarray, variables_names: list[str]):
        self.x = x
        self.y = y
        self.variables_names = variables_names
        self._validate_data()

    def _validate_data(self):
        if self.x.ndim != 2:
            raise ValueError("X debe de ser una matriz")
        if self.y.ndim != 1:
            raise ValueError("Y debe de ser un vector")
        if len(self.x) != len(self.y):
            raise ValueError("X y Y deben de tener el mismo numero de observaciones")
        if self.x.shape[1] != len(self.variables_names):
            raise ValueError("La cantidad de nombres no coincide con la columna de las X's")

    def fit(self, chromosome: list[int]) -> dict:
        if len(chromosome) != self.x.shape[1]:
            raise ValueError("Longitud del cromosoma incorrecta")
        selected_indices = [
            index for index, bit in enumerate(chromosome) if bit == 1
        ]
        if not selected_indices:
            raise ValueError("El cromosoma no seleccionó ninguna variable")
        selected_x = self.x[:, selected_indices]
        design_matrix = np.column_stack([
            np.ones(len(self.y)),
            selected_x
        ])

        number_of_observations = len(self.y)
        number_of_parameters = design_matrix.shape[1]

        if number_of_observations <= number_of_parameters:
            raise ValueError("No existen suficientes observaciones")
        coefficients, _, rank, _ = np.linalg.lstsq(
            design_matrix, self.y, rcond=None
        )
        if rank < number_of_parameters:
            raise ValueError("El modelo presenta colinealidad perfecta")

        predictions = design_matrix @ coefficients
        residuals = self.y - predictions

        rss = float(np.sum(residuals ** 2))
        safe_rss = max(rss, 1e-12)

        bic = (
            number_of_observations 
            * math.log(safe_rss / number_of_observations) 
            + number_of_parameters 
            * math.log(number_of_observations)
        )

        rmse = math.sqrt(rss/number_of_observations)
        total_sum_squares = float(np.sum((self.y - np.mean(self.y))**2))

        if total_sum_squares == 0:
            r_squared = 0.0
        else: 
            r_squared = 1 - (rss/total_sum_squares)
        number_of_predictors = len(selected_indices)
        adjusted_r_squared = 1 - ((1 - r_squared) * (number_of_observations - 1) / (number_of_observations - number_of_predictors - 1))
        selected_names = [self.variables_names[index] for index in selected_indices]
        return {
            "bic": bic,
            "rmse": rmse,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "rss": rss,
            "coefficients": coefficients,
            "predictions": predictions,
            "selected_indices": selected_indices,
            "selected_names": selected_names
        }

    def evaluate(self, chromosome: list[int]) -> float:
        try:
            result = self.fit(chromosome)
            return result["bic"]
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
    