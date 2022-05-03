import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import pickle
import datetime
from sklearn import model_selection

from main import AnytimeAlgorithm, AnytimeAlgorithmResult
from ..datasets import get_dataset_sample


def save_results(obj, name=None, full_name=None):
    if full_name is None:
        filename = f"results/{name}{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        assert full_name is not None
        filename = full_name
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to file {filename}")


def load_results(file_paths):
    results = list()
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            current_results = pickle.load(f)
        results.extend([AnytimeAlgorithmResult.from_dict(result) for result in current_results])
    return results


def load_results_multi(file_paths):
    all_results = list()
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            results_multi = pickle.load(f)
        for results in results_multi:
            results = [AnytimeAlgorithmResult.from_dict(x) for x in results]
            all_results.append(results)
    return all_results


def anytime_multirun(anytime_cls, n_runs, save, *args, **kwargs):
    filename = f"results/{anytime_cls.__name__}{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    all_results = list()
    all_dict_results = list()
    for run in range(n_runs):
        print()
        print(f"###########################################")
        print(f"Started run {run}...")
        algorithm = anytime_cls()
        algorithm.run(*args, **kwargs)
        print(
            f"Total run duration {algorithm.get_total_duration()}, mean duration {algorithm.get_mean_duration()}, mean val metric {algorithm.get_mean_val_metric()}")
        all_results.append(algorithm.results)
        all_dict_results.append([result.to_dict() for result in algorithm.results])

        if save:
            save_results(all_dict_results, full_name=filename)

    print("Completed.")
    return all_results


def sample_random_search(results, min_duration, profiles=None):
    shuffled_results = copy.deepcopy(results)
    random.shuffle(shuffled_results)
    sampled_results = list()
    best_val_metric = - np.inf
    time = 0

    if profiles is not None:
        profile = copy.deepcopy(random.choice(profiles))
        for result in profile:
            if AnytimeAlgorithm.get_total_duration_from_results(sampled_results) >= min_duration:
                return sampled_results
            if result.val_metric > best_val_metric:
                best_val_metric = result.val_metric
            result.best_val_metric = best_val_metric
            time += result.duration
            result.time = time
            sampled_results.append(result)

    for result in shuffled_results:
        if AnytimeAlgorithm.get_total_duration_from_results(sampled_results) >= min_duration:
            return sampled_results
        if result.val_metric > best_val_metric:
            best_val_metric = result.val_metric
        result.best_val_metric = best_val_metric
        time += result.duration
        result.time = time
        sampled_results.append(result)
    raise Exception("Not enough results to reach specified duration.")


def get_mean_best_val_metric(all_results, time, initial_best_val_metric):
    best_val_metrics = list()
    for results in all_results:
        best_val_metric = initial_best_val_metric
        for result in results:
            if result.time > time:
                break
            best_val_metric = result.best_val_metric
        best_val_metrics.append(best_val_metric)
    return np.mean(best_val_metrics)


def get_average_results(all_results, initial_best_val_metric=0):
    all_times = list()
    for results in all_results:
        all_times.extend([result.time for result in results])
    all_times = sorted(all_times)

    times = list()
    best_val_metrics = list()
    for time in all_times:
        times.append(time)
        best_val_metrics.append(
            get_mean_best_val_metric(all_results, time, initial_best_val_metric=initial_best_val_metric))
    return times, best_val_metrics


def plot_results(all_results, xlim, ylim, ylabel, xlabel="Time (s)", initial_best_val_metric=0,
                 performance_trace_alpha=0.1,
                 trace_label=None, profile_label=None):
    average_times, average_best_val_metrics = get_average_results(all_results,
                                                                  initial_best_val_metric=initial_best_val_metric)

    for i, results in enumerate(all_results):
        x = [0] + [result.time for result in results]
        x += [x[-1] + (x[-1] - x[-2]) / 10]
        y = [initial_best_val_metric, initial_best_val_metric] + [result.best_val_metric for result in results]
        if i == 0:
            label = trace_label
        else:
            label = None
        plt.step(x, y, alpha=performance_trace_alpha, c='black', label=label)

    x = [0] + average_times
    x += [x[-1] + (x[-1] - x[-2]) / 10]
    y = [initial_best_val_metric, initial_best_val_metric] + average_best_val_metrics
    plt.step(x, y, alpha=1, c='black', label=profile_label)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)


def fix_grid_search(all_results, profiles):
    all_fixed_results = list()
    for results in all_results:
        fixed_results = list()
        best_val_metric = - np.inf
        time = 0
        profile = copy.deepcopy(random.choice(profiles))

        for result in profile:
            if result.val_metric > best_val_metric:
                best_val_metric = result.val_metric
            result.best_val_metric = best_val_metric
            time += result.duration
            result.time = time
            fixed_results.append(result)

        for result in results[1:]:
            if result.val_metric > best_val_metric:
                best_val_metric = result.val_metric
            result.best_val_metric = best_val_metric
            time += result.duration
            result.time = time
            fixed_results.append(result)

        all_fixed_results.append(fixed_results)

    return all_fixed_results


def get_data_for_run(x, y, validation_data, fraction, test_size):
    assert (fraction is None) != (validation_data is None)
    assert not ((validation_data is not None) and (test_size is not None))

    if fraction is not None:
        x_iter, y_iter = get_dataset_sample(x, y, fraction, seed=None)
    else:
        x_iter, y_iter = x, y

    if validation_data is None:
        x_iter, x_val_iter, y_iter, y_val_iter = model_selection.train_test_split(x_iter, y_iter, test_size=test_size)
        validation_data_iter = (x_val_iter, y_val_iter)
    else:
        validation_data_iter = validation_data

    #     print(len(y_iter), len(validation_data_iter[1]))
    return x_iter, y_iter, validation_data_iter
