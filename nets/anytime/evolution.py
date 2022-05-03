import tensorflow as tf
import numpy as np
import copy
import time

from ..helpers import get_convolutional_model
from ..models import Sequential
from main import AnytimeAlgorithm, AnytimeAlgorithmResult, interruptible
from ..schedule import Schedule, StaticEpochNoRegularization, DynamicEpoch
from helpers import get_data_for_run


class EvolutionHyperparameter:
    def __init__(self, initial_value, min_value, max_value, strategy):
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.strategy = strategy


class Individual:
    def __init__(self, x, layer_sizes, output_neurons, hyperparameters):
        self.genome = np.array([
            hyperparameters['regularization_penalty'].initial_value,
            hyperparameters['learning_rate'].initial_value,
            hyperparameters['batch_size'].initial_value,
        ])
        self.model = get_convolutional_model(x, layer_sizes, output_neurons=output_neurons)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.get_learning_rate())
        self.history = Sequential.ParameterContainer.prepare_history()
        self.age = 0

        self.model.build(x.shape)

    def copy(self):
        individual_copy = Individual.__new__(Individual)
        individual_copy.genome = self.genome.copy()
        individual_copy.model = self.model.copy()
        individual_copy.optimizer = copy.deepcopy(self.optimizer)
        individual_copy.history = copy.deepcopy(self.history)
        individual_copy.age = self.age
        return individual_copy

    def mutate(self, mutation_strength):
        if mutation_strength > 0:
            self.model.mutate(mutation_strength)

    def correct(self, hyperparameters):
        self.genome[0] = np.clip(self.genome[0], hyperparameters['regularization_penalty'].min_value,
                                 hyperparameters['regularization_penalty'].max_value)
        self.genome[1] = np.clip(self.genome[1], hyperparameters['learning_rate'].min_value,
                                 hyperparameters['learning_rate'].max_value)
        self.genome[2] = np.clip(self.genome[2], hyperparameters['batch_size'].min_value,
                                 hyperparameters['batch_size'].max_value)

    def get_loss(self):
        return self.history['loss'][-1]

    def get_metric(self):
        return self.history['metric'][-1]

    def get_val_loss(self):
        return self.history['val_loss'][-1]

    def get_val_metric(self):
        return self.history['val_metric'][-1]

    def get_hidden_layer_sizes(self):
        return self.history['hidden_layer_sizes'][-1]

    def get_age(self):
        #         return len(self.history['val_metric'])
        return self.age

    def get_age_penalty_coefficient(self, age_penalty_period):
        age = self.get_age()
        #         return 1 / (2 ** max(0, (age - age_penalty_period) / age_penalty_period))
        if age <= age_penalty_period:
            return 1
        return 1 / (1 + 0.005 * 1.8 ** (age - age_penalty_period))

    def get_fitness(self, age_penalty_period):
        if age_penalty_period is None:
            return self.get_val_metric()
        return self.get_val_metric() * self.get_age_penalty_coefficient(age_penalty_period)

    def get_regularization_penalty(self):
        return 10. ** -self.genome[0]

    def get_learning_rate(self):
        return self.genome[1]

    def get_batch_size(self):
        return int(self.genome[2])


class Evolution(AnytimeAlgorithm):
    @staticmethod
    def initialize_population(population_size, x, layer_sizes, output_neurons, hyperparameters):
        population = [Individual(x, layer_sizes, output_neurons, hyperparameters) for _ in range(population_size)]
        return population

    @staticmethod
    def introduce_new_individuals(population, n_introduced, x, layer_sizes, output_neurons, hyperparameters):
        introduced_individuals = [Individual(x, layer_sizes, output_neurons, hyperparameters) for _ in
                                  range(n_introduced)]
        return population + introduced_individuals

    @staticmethod
    def get_best_individual_by_fitness(population, age_penalty_period):
        return max(population, key=lambda x: x.get_fitness(age_penalty_period))

    @staticmethod
    def get_best_individual_by_val_metric(population):
        return max(population, key=lambda x: x.get_val_metric())

    @staticmethod
    def get_strategy(hyperparameters):
        return [
            hyperparameters['regularization_penalty'].strategy,
            hyperparameters['learning_rate'].strategy,
            hyperparameters['batch_size'].strategy
        ]

    @staticmethod
    def crossover(population, n_parents, mutation_strength, hyperparameters):
        novel_population = list()
        for individual in population:
            parents_selection = np.random.choice(list(range(len(population))), size=n_parents, replace=False)
            parent_genomes = [population[index].genome for index in parents_selection]
            offspring_genome = np.mean(np.vstack(parent_genomes), axis=0)
            offspring = individual.copy()
            offspring.genome = offspring_genome
            offspring.genome += np.random.normal(0, 1, offspring.genome.shape) * Evolution.get_strategy(hyperparameters)
            offspring.correct(hyperparameters)
            offspring.optimizer.learning_rate.assign(offspring.get_learning_rate())
            offspring.mutate(mutation_strength)
            novel_population.append(offspring)
        return population + novel_population

    # @staticmethod
    # def mutation(population, mutation_strength):
    #     new_population = list()
    #     for individual in population:
    #         individual_copy = individual.copy()
    #         individual_copy.mutate(mutation_strength)
    #         new_population.extend([individual, individual_copy])
    #     return new_population

    @staticmethod
    def extend_history(old_history, new_history):
        for key in old_history.keys():
            old_history[key].extend(new_history[key])

    @staticmethod
    def training(population, x, y, validation_data, min_new_neurons, growth_percentage, verbose, use_static_graph,
                 age_penalty_period, fine_tuning):
        for individual in population:
            model = individual.model
            optimizer = individual.optimizer
            # schedule = Schedule([StaticEpochNoRegularization()])
            if fine_tuning and individual.get_age() >= age_penalty_period:
                schedule = Schedule([StaticEpochNoRegularization()])
            else:
                schedule = Schedule([DynamicEpoch(individual.get_regularization_penalty(), 'weighted_l1')])
            x_train_sample, y_train_sample = x, y
            x_test_sample, y_test_sample = validation_data[0], validation_data[1]
            history = model.fit(x=x_train_sample, y=y_train_sample, optimizer=optimizer, schedule=schedule,
                                batch_size=individual.get_batch_size(),
                                min_new_neurons=min_new_neurons, validation_data=(x_test_sample, y_test_sample),
                                growth_percentage=growth_percentage,
                                verbose=verbose, use_static_graph=use_static_graph)
            Evolution.extend_history(individual.history, history)
        return population

    @staticmethod
    def tournament_selection(population, population_size, tournament_size, age_penalty_period):
        new_population = list()

        while len(new_population) < population_size:
            selection = np.random.choice(list(range(len(population))), size=tournament_size, replace=False)
            best_individual = None
            best_fitness = - np.inf
            for individual_index in selection:
                individual = population[individual_index]
                fitness = individual.get_fitness(age_penalty_period)
                if fitness > best_fitness:
                    best_individual = individual
                    best_fitness = fitness
            new_population.append(best_individual.copy())

        return new_population

    @staticmethod
    def age_population(population):
        for individual in population:
            individual.age += 1
        return population

    @staticmethod
    def measure_fitnesses(population, age_penalty_period):
        fitnesses = list()
        for individual in population:
            fitnesses.append(individual.get_fitness(age_penalty_period))
        return fitnesses

    def print_generation_statistics(self, generation, population, duration, age_penalty_period):
        population_sorted_by_fitness = sorted(population, key=lambda x: x.get_fitness(age_penalty_period), reverse=True)
        individuals = [
            (
                individual.get_age(),
                round(individual.get_val_metric(), 4),
                round(individual.get_fitness(age_penalty_period), 4),
                individual.get_regularization_penalty(),
                individual.get_learning_rate(),
                individual.get_batch_size(),
                individual.get_hidden_layer_sizes(),
            )
            for individual in population_sorted_by_fitness
        ]
        population_sorted_by_val_metric = sorted(population, key=lambda x: x.get_val_metric(), reverse=True)
        best_individual = population_sorted_by_val_metric[0]
        result = AnytimeAlgorithmResult(
            loss=best_individual.get_loss(),
            metric=best_individual.get_metric(),
            val_loss=best_individual.get_val_loss(),
            val_metric=best_individual.get_val_metric(),
            hidden_layer_sizes=best_individual.get_hidden_layer_sizes(),
            duration=duration,
            args=list(),
            kwargs={'regularization_penalty': best_individual.get_regularization_penalty(),
                    'learning_rate': best_individual.get_learning_rate(),
                    'batch_size': best_individual.get_batch_size()},
        )
        self.log_result(result)
        print(
            f"Generation {generation}: {round(duration, 1)} s, best val metric {round(best_individual.get_val_metric(), 4)}, {individuals}")
        print(
            f"#### Total duration {round(self.get_total_duration(), 1)}, overall best val metric {self.best_val_metric} ####")

    @interruptible
    def run(self, x, y, validation_data, layer_sizes, output_neurons, hyperparameters, n_parents, population_size=10,
            n_generations=10,
            tournament_size=3, elitism=True, n_introduced=0, age_penalty_period=None, min_new_neurons=20,
            growth_percentage=0.2, use_static_graph=False,
            mutation_strength=0., fine_tuning=False, fraction=None, test_size=None):
        super().run()

        x, y, validation_data = get_data_for_run(x, y, validation_data, fraction, test_size)
        population = self.initialize_population(population_size, x, layer_sizes, output_neurons, hyperparameters)
        best_individual = None
        fitnesses_history = list()
        generation = 0
        while True:
            if generation >= n_generations:
                break
            start_time = time.time()
            population = self.crossover(population, n_parents, mutation_strength, hyperparameters)
            # population = mutation(population, mutation_strength)
            population = self.introduce_new_individuals(population, n_introduced, x, layer_sizes, output_neurons,
                                                        hyperparameters)
            population = self.training(population, x, y, validation_data, min_new_neurons, growth_percentage,
                                       verbose=False,
                                       use_static_graph=use_static_graph, age_penalty_period=age_penalty_period,
                                       fine_tuning=fine_tuning)
            population = self.tournament_selection(population, population_size, tournament_size, age_penalty_period)
            if elitism:
                if best_individual is not None:
                    population.append(best_individual)
            population = self.age_population(population)
            if elitism:
                best_individual = self.get_best_individual_by_fitness(population, age_penalty_period).copy()
            fitnesses = self.measure_fitnesses(population, age_penalty_period)
            duration = time.time() - start_time
            self.print_generation_statistics(generation, population, duration, age_penalty_period)
            generation += 1
