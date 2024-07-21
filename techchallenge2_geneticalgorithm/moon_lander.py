import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random



# Define the fitness function
def evaluate_fitness(individual, env):
    state = env.reset(seed=seed_value)
    total_reward = 0
    for i in range(len(individual)):
        # get next action to execute from individual
        action = individual[i]
        # execute the action and get the reward
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # check if execution has terminated
        if terminated or truncated:
            break
    return total_reward

# Initialize the population
def initialize_population(pop_size, individual_length, env):
    population = []
    for _ in range(pop_size):
        # create an individual codified as an array of random actions
        individual = [env.action_space.sample() for _ in range(individual_length)]
        # append new individual to the population under cosntruction
        population.append(individual)
    return population

# Sort the population to simplify selection
def sort_population(population, fitnesses):
    # Combine the population and fitness lists usinf zip and sort it based on the fitness
    zipped = zip(fitnesses, population)
    sorted_pairs = sorted(zipped, reverse=True, key=lambda x: x[0])

    # Separate the sorted lists
    sorted_fitnesses, sorted_individuals = zip(*sorted_pairs)
    sorted_fitnesses = list(sorted_fitnesses)
    sorted_individuals = list(sorted_individuals)

    return sorted_individuals, sorted_fitnesses

# select random parents for crossover
def select_parents(sorted_population, selected_parents):
    # choose 2 random individuals from the top 10 of the sorted population
    return random.choices(sorted_population[:selected_parents], k=2)

# Perform crossover
def crossover(parent1, parent2):
    # Select a random index to split the parents
    idx = np.random.randint(1, len(parent1))
    # generate 2 children by combining the parents
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2

# Perform mutation
def mutate(env, individual, mutation_rate):
    # decide if mutation will occure based on the mutation rate˜
    if random.random() < mutation_rate:
        # mutate using the mutation rate to mutate cromossomes randonly
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] =  env.action_space.sample()
    return individual

# Function to update the plot
def update_plot(gen, fitness_scores):
    line.set_data(range(len(fitness_scores)), fitness_scores)
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Rescale view
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot
    # clear_output(wait=True)
    plt.show()



# Set the random seed
seed_value = 42

# Initialize the environment and set the seed
env = gym.make('LunarLander-v2', render_mode=None)
env.action_space.seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Main Genetic Algorithm loop
population_size = 100
mutation_rate = 0.4
selected_parents = 60
generations = 1000
individual_length = 200

# initialize chart to visualize fintness evolution
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')  # Initialize an empty line
ax.set_xlim(0, generations)
ax.set_ylim(-300, 300)  # Set appropriate y-axis limits
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness/Score')

# initialize a random population to be the first generation
population = initialize_population(population_size, individual_length, env)
best_individuals = []
fitness_scores = []
best_individual = []
best_fitness = 0
for generation in range(generations):
    # calculate fitness for each individual in the population
    fitnesses = [evaluate_fitness(ind, env) for ind in population]
    # sort the population based on fitness
    sorted_population, sorted_fitness = sort_population(population, fitnesses)
    # save best individual of this generation for later comparison
    best_individual = sorted_population[0]
    best_fitness = sorted_fitness[0]
    best_individuals.append(best_individual);
    fitness_scores.append(best_fitness);
    update_plot(generation, fitness_scores)
    print("Generation:", generation," Best fitness:", sorted_fitness[0])
    # Initialize next generation using elitism - keep the best 2 individual in the next generation
    new_population = [sorted_population[0], sorted_population[1]]
    while len(new_population) < population_size:
        # select parents for crossover
        parent1, parent2 = select_parents(sorted_population, selected_parents)
        # perform crossover and generate 2 new childs for the new generation
        child1, child2 = crossover(parent1, parent2)
        # mutate the childs with a given mutation rate
        new_population.extend([mutate(env, child1, mutation_rate), mutate(env, child2, mutation_rate)])    
    population = new_population

print("Best fitness:", best_score)
print("Best individual:", best_individual)
# visualize/render best individual
env = gym.make('LunarLander-v2', render_mode="human")
evaluate_fitness(best_individual, env)

plt.ioff()  # Turn off interactive mode
plt.show()