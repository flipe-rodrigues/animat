import numpy as np
from rnn import RNN
from fitness import evaluate_fitness

import concurrent.futures

# Assuming these functions are defined elsewhere in your code

def evolve_rnn_population(population_size, generations, mutation_rate):
    # Initialize population with random RNNs
    population = [RNN() for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation}")

        # Evaluate fitness of each individual in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fitness_results = list(executor.map(evaluate_fitness, population))

        # Select the top individuals based on fitness
        sorted_population = [x for _, x in sorted(zip(fitness_results, population), key=lambda pair: pair[0], reverse=True)]
        population = sorted_population[:population_size // 2]

        # Generate new individuals through mutation
        new_population = []
        for individual in population:
            new_individual = individual.mutate(mutation_rate)
            new_population.append(new_individual)

        # Add new individuals to the population
        population.extend(new_population)

    # Return the best individual from the final population
    best_individual = max(population, key=evaluate_fitness)
    return best_individual

if __name__ == "__main__":
    population_size = 100
    generations = 50
    mutation_rate = 0.01

    best_rnn = evolve_rnn_population(population_size, generations, mutation_rate)
    print("Best RNN found:", best_rnn)