import gymnasium as gym
import numpy as np
import random

# Set the random seed
seed_value = 42

# Initialize the environment and set the seed
env = gym.make('LunarLander-v2', render_mode="none")
# env.seed(seed_value)
env.action_space.seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Define the fitness function
def evaluate_fitness(individual, env):
    state = env.reset(seed=seed_value)
    total_reward = 0
    for i in range(len(individual)):
        action = individual[i]  # Simple linear policy
        # print ("action", action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

# Initialize the population
def initialize_population(pop_size, individual_length, env):
    population = []
    for _ in range(pop_size):
        individual = [env.action_space.sample() for _ in range(individual_length)]
        # print ("individual", individual)
        population.append(individual)
    return population

# Perform selection
def sort_population(population, fitnesses):

    # Combinar as duas listas usando zip e ordenar com base nos valores de fitness
    zipped = zip(fitnesses, population)
    sorted_pairs = sorted(zipped, reverse=True, key=lambda x: x[0])

    # Separar as listas ordenadas
    sorted_fitnesses, sorted_individuals = zip(*sorted_pairs)

    # Converter para lista, se necessário
    sorted_fitnesses = list(sorted_fitnesses)
    sorted_individuals = list(sorted_individuals)

    # print("Indivíduos ordenados:", sorted_individuals)
    # print("Aptidões ordenadas:", sorted_fitnesses)
    # selected = random.choices(population, weights=fitnesses, k=len(population))
    return sorted_individuals, sorted_fitnesses

# Perform crossover
def crossover(parent1, parent2):
    idx = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2

# Perform mutation
def mutate(env, individual, mutation_rate):
    if random.random() < mutation_rate:
        # mutate
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] =  env.action_space.sample()
    return individual

# Main GA loop
population_size = 500
generations = 1000
individual_length = 200
mutation_rate = 0.2
# state_space = env.observation_space.shape[0]
# action_space = env.action_space.n
# print ("state_space", state_space)
# print ("action_space", action_space)

population = initialize_population(population_size, individual_length, env)
for generation in range(generations):
    print("generation", generation)
    fitnesses = [evaluate_fitness(ind, env) for ind in population]
    # print ("fitnesses", fitnesses)
    sorted_population, sorted_fitness = sort_population(population, fitnesses)
    
    print("Best fitness:", sorted_fitness[0])
    new_population = [sorted_population[0]] # elitism
    # print ("new_population", new_population)
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(population[:10], k=2)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([mutate(env, child1, mutation_rate), mutate(env, child2, mutation_rate)])
    
    population = new_population

best_individual = max(population, key=lambda ind: evaluate_fitness(ind, env))
print("Best individual:", best_individual)
#visualize best individual
env = gym.make('LunarLander-v2', render_mode="human")
evaluate_fitness(best_individual, env)