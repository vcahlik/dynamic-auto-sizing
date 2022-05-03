import numpy as np
import time
import random
import collections
import itertools
import datetime

from helpers import save_results, get_data_for_run
from ..helpers import get_statistics_from_history
from main import AnytimeAlgorithm, Range, interruptible, AnytimeAlgorithmResult, identity_postprocess


class RandomSearch(AnytimeAlgorithm):
    @staticmethod
    def sample_combination(values):
        combination = list()
        for value in values:
            if isinstance(value, Range):
                combination.append(value.sample())
            else:
                combination.append(value)
        return tuple(combination)

    @interruptible
    def run(self, train_fn, x, y, validation_data=None, postprocess_fn=identity_postprocess, fraction=None,
            test_size=None, save=False,
            max_duration=None, complete_profile=False, *args, **kwargs):
        super().run()

        filename = f"results/RandomSearch{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        start_time = time.time()
        histories = list()

        best_overall_val_metric = -np.inf
        best_overall_combination = None

        while True:
            start_time = time.time()
            combination_args = self.sample_combination([*args])
            combination_kwargs = dict(zip(kwargs.keys(), self.sample_combination(list(kwargs.values()))))
            combination_args, combination_kwargs = postprocess_fn(combination_args, combination_kwargs)

            combination = combination_args + tuple(combination_kwargs.values())
            print(f"Run with parameters {combination} started...")

            x_iter, y_iter, validation_data_iter = get_data_for_run(x, y, validation_data, fraction, test_size)

            history = train_fn(x_iter, y_iter, validation_data_iter, *combination_args, **combination_kwargs)
            history['parameters'] = combination
            histories.append(history)

            duration = time.time() - start_time
            best_loss, best_metric, best_val_loss, best_val_metric, best_hidden_layer_sizes = get_statistics_from_history(
                history)
            print(
                f"Run with parameters {combination} completed, duration {round(duration, 1)}, best_val_loss: {best_val_loss}, best_val_metric: {best_val_metric}, best_hidden_layer_sizes: {best_hidden_layer_sizes}")

            if best_val_metric > best_overall_val_metric:
                best_overall_val_metric = best_val_metric
                best_overall_combination = combination

            if complete_profile:
                history_length = len(history['val_metric'])
                epoch_duration = duration / history_length
                _time = 0
                for i in range(history_length):
                    _time += epoch_duration
                    result = AnytimeAlgorithmResult(loss=history['loss'][i], metric=history['metric'][i],
                                                    val_loss=history['val_loss'][i],
                                                    val_metric=history['val_metric'][i],
                                                    hidden_layer_sizes=history['hidden_layer_sizes'][i],
                                                    duration=epoch_duration, args=combination_args,
                                                    kwargs=combination_kwargs)
                    self.log_result(result, _time=_time)
                return

            result = AnytimeAlgorithmResult(loss=best_loss, metric=best_metric, val_loss=best_val_loss,
                                            val_metric=best_val_metric,
                                            hidden_layer_sizes=best_hidden_layer_sizes, duration=duration,
                                            args=combination_args, kwargs=combination_kwargs)
            self.log_result(result)
            print(
                f'Total duration {round(self.get_total_duration(), 1)}, mean val metric {self.get_mean_val_metric()}, mean duration {round(self.get_mean_duration(), 1)}, best overall combination: {best_overall_combination}, val_metric: {best_overall_val_metric}')

            if save:
                save_results([result.to_dict() for result in self.results], full_name=filename)

            if max_duration is not None and self.get_total_duration() > max_duration:
                break


class AnytimeGridSearch(AnytimeAlgorithm):
    @staticmethod
    def get_range_arguments(arguments):
        return [argument for argument in arguments if isinstance(argument, Range)]

    @staticmethod
    def _raw_relative_combinations_for_level(level, n_range_arguments):
        l = level + 1
        single_argument_values = tuple(i / (2 ** l) for i in range(1, 2 ** l))
        all_argument_values = [single_argument_values for _ in range(n_range_arguments)]
        return list(itertools.product(*all_argument_values))

    @staticmethod
    def get_relative_combinations_for_level(level, n_range_arguments, randomize):
        relative_combinations = set(AnytimeGridSearch._raw_relative_combinations_for_level(level, n_range_arguments))
        previous_combinations = set()
        for i in range(0, level):
            previous_combinations = previous_combinations.union(
                set(AnytimeGridSearch._raw_relative_combinations_for_level(i, n_range_arguments)))
        relative_combinations = list(relative_combinations.difference(previous_combinations))
        if randomize:
            random.shuffle(relative_combinations)
        return relative_combinations

    @staticmethod
    def relative_combination_to_range_argument_combination(relative_combination, range_arguments):
        return [range_argument.sample(x=relative_value) for range_argument, relative_value in
                zip(range_arguments, relative_combination)]

    @staticmethod
    def range_argument_combination_to_combination(range_argument_combination, arguments):
        combination = list()
        range_argument_combination = collections.deque(range_argument_combination)
        for argument in arguments:
            if isinstance(argument, Range):
                # Range argument, take its value from the generated relative argument combination
                value = range_argument_combination.popleft()
            else:
                # Normal argument, use it directly
                value = argument
            combination.append(value)
        return tuple(combination)

    @staticmethod
    def get_combinations_for_level(level, arguments, randomize):
        range_arguments = AnytimeGridSearch.get_range_arguments(arguments)
        relative_combinations = AnytimeGridSearch.get_relative_combinations_for_level(level, len(range_arguments),
                                                                                      randomize)
        range_argument_combinations = [
            AnytimeGridSearch.relative_combination_to_range_argument_combination(relative_combination, range_arguments)
            for relative_combination in relative_combinations]
        combinations = [
            AnytimeGridSearch.range_argument_combination_to_combination(range_argument_combination, arguments)
            for range_argument_combination in range_argument_combinations]
        return combinations

    @interruptible
    def run(self, train_fn, x, y, validation_data=None, postprocess_fn=identity_postprocess, fraction=None,
            test_size=None, randomize=True,
            max_duration=None, complete_profile=False, *args, **kwargs):
        super().run()

        start_time = time.time()
        histories = list()

        best_overall_val_metric = -np.inf
        best_overall_combination = None

        level = 0
        interrupt = False
        arguments = [*args] + list(kwargs.values())
        while True:
            for combination in AnytimeGridSearch.get_combinations_for_level(level, arguments, randomize):
                start_time = time.time()
                combination_args = combination[:len(args)]
                combination_kwargs = dict(zip(kwargs.keys(), combination[len(args):]))
                combination_args, combination_kwargs = postprocess_fn(combination_args, combination_kwargs)

                combination = combination_args + tuple(combination_kwargs.values())
                print(f"Run with parameters {combination} started...")

                x_iter, y_iter, validation_data_iter = get_data_for_run(x, y, validation_data, fraction, test_size)

                history = train_fn(x_iter, y_iter, validation_data_iter, *combination_args, **combination_kwargs)
                history['parameters'] = combination
                histories.append(history)

                duration = time.time() - start_time
                best_loss, best_metric, best_val_loss, best_val_metric, best_hidden_layer_sizes = get_statistics_from_history(
                    history)
                print(
                    f"Run with parameters {combination} completed, duration {round(duration, 1)}, best_val_loss: {best_val_loss}, best_val_metric: {best_val_metric}, best_hidden_layer_sizes: {best_hidden_layer_sizes}")

                if best_val_metric > best_overall_val_metric:
                    best_overall_val_metric = best_val_metric
                    best_overall_combination = combination

                if complete_profile:
                    history_length = len(history['val_metric'])
                    epoch_duration = duration / history_length
                    _time = 0
                    for i in range(history_length):
                        _time += epoch_duration
                        result = AnytimeAlgorithmResult(loss=history['loss'][i], metric=history['metric'][i],
                                                        val_loss=history['val_loss'][i],
                                                        val_metric=history['val_metric'][i],
                                                        hidden_layer_sizes=history['hidden_layer_sizes'][i],
                                                        duration=epoch_duration, args=combination_args,
                                                        kwargs=combination_kwargs)
                        self.log_result(result, _time=_time)
                    return

                result = AnytimeAlgorithmResult(loss=best_loss, metric=best_metric, val_loss=best_val_loss,
                                                val_metric=best_val_metric,
                                                hidden_layer_sizes=best_hidden_layer_sizes, duration=duration,
                                                args=combination_args, kwargs=combination_kwargs,
                                                level=level)
                self.log_result(result)
                print(
                    f'Total duration {round(self.get_total_duration(), 1)}, mean val metric {self.get_mean_val_metric()}, mean duration {round(self.get_mean_duration(), 1)}, best overall combination: {best_overall_combination}, val_metric: {best_overall_val_metric}')

                if max_duration is not None and self.get_total_duration() > max_duration:
                    interrupt = True
                    break
            if interrupt:
                break
            level += 1
