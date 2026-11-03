import random
from collections.abc import Callable

class GeneticAlgorithm:

    def __init__(
            self, 
            chromosome_length: int, 
            population_size: int, 
            generations: int, 
            crossover_probability: float,
            mutation_probability: float,
            tournament_size: int,
            evaluator: Callable[[list[int]], float],
            random_seed: int = 42):
        if chromosome_length <= 0:
            raise ValueError(
                "El cromosoma debe tener al menos un gen."
            )

        if population_size < 2:
            raise ValueError(
                "La población debe tener al menos dos individuos."
            )

        if generations <= 0:
            raise ValueError(
                "Debe existir al menos una generación."
            )

        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.evaluator = evaluator

        self.random = random.Random(random_seed)

    def create_individual(self) -> list[int]:
        chromosome = [
            self.random.randint(0, 1) for _ in range(self.chromosome_length)
        ]

        if not any(chromosome):
            random_index = self.random.randrange(
                self.chromosome_length
            )
            chromosome[random_index] = 1
        return chromosome

    def create_population(self) -> list[list[int]]:
        return [
            self.create_individual() for _ in range(self.population_size)
        ]

    def tournament_selection(self, population: list[list[int]], costs: list[float]) -> list[int]:
        competitor_indices = [
            self.random.randrange(len(population))
            for _ in range(self.tournament_size)
        ]

        winner_index = min(competitor_indices, key=lambda index: costs[index])
        return list(population[winner_index])

    def crossover(self, father_1: list[int], father_2: list[int]):
        if(self.chromosome_length < 2 or self.random.random() >= self.crossover_probability):
            return list(father_1), list(father_2)

        cutting_point = self.random.randint(1, self.chromosome_length-1)
        child_1 = (father_1[:cutting_point]+father_2[cutting_point:])
        child_2 = (father_2[:cutting_point]+father_1[cutting_point:])

        return child_1, child_2

    def mutate(self, chromosome: list[int]) -> None:
        mutated = list(chromosome)
        for index in range(len(mutated)):
            if(
                self.random.random() < self.mutation_probability
            ):
                mutated[index] ^= 1
        return mutated

    def create_next_generation(self, population: list[list[int]], costs: list[float], elite: list[int]) -> list[list[int]]:
        new_population = [list(elite)]
        while len(new_population) < self.population_size:
            father_1 = self.tournament_selection(population, costs)
            father_2 = self.tournament_selection(population, costs)

            child_1, child_2 = self.crossover(father_1, father_2)
            child_1 = self.mutate(child_1)
            child_2 = self.mutate(child_2)

            new_population.append(child_1)

            if len(new_population) < self.population_size:
                new_population.append(child_2)

        return new_population

    def run(self):
        population = self.create_population()

        best_chromosome = None
        best_cost = float("inf")
        history = []

        for generation in range(1, self.generations+1):
            costs = [self.evaluator(chromosome) for chromosome in population]
            generation_best_index = min(range(len(costs)), key=lambda index: costs[index])
            generation_best_cost = costs[generation_best_index]
            generation_best_chromosome = population[generation_best_index]
            if(best_chromosome is None or generation_best_cost < best_cost):
                best_cost = generation_best_cost
                best_chromosome = list(generation_best_chromosome)
            history.append(best_cost)
            print(
                f"Generación: {generation:3d} | "
                f"Mejor BIC: {best_cost:12.6f}"
                f"Variables: {sum(best_chromosome)}"
            )
            if generation < self.generations:
                population = self.create_next_generation(
                    population, 
                    costs,
                    best_chromosome
                )
        return best_chromosome, best_cost, history

    