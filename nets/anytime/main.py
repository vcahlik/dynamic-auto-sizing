import numpy as np
import time
import abc


def interruptible(f):
    def function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            print("Interrupted by user.")

    return function


class Range:
    def __init__(self, min_value, max_value, transformation=None, integer=False):
        self.min_value = min_value
        self.max_value = max_value
        self.transformation = transformation
        self.integer = integer

    def sample(self, x=None):
        y = self._raw_sample(x)
        if self.integer:
            y = np.rint(y).astype(int)
        if self.transformation is not None:
            result = self.transformation(y)
            #             print(f"Transformed value {y} to {result}.")
            return result
        return y

    @abc.abstractmethod
    def _raw_sample(self):
        pass


class UniformRange(Range):
    def _raw_sample(self, x):
        if x is None:
            return np.random.uniform(self.min_value, self.max_value)
        else:
            return self.min_value + x * (self.max_value - self.min_value)


class PowerRange(Range):
    def _raw_sample(self, x):
        if x is None:
            y = np.random.uniform(self.min_value, self.max_value)
        else:
            y = self.min_value + x * (self.max_value - self.min_value)
        return 10. ** y


class AnytimeAlgorithmResult:
    def __init__(self, loss, metric, val_loss, val_metric, hidden_layer_sizes, duration, args, kwargs,
                 level=None, regularization_penalty=None):
        self.loss = loss
        self.metric = metric
        self.val_loss = val_loss
        self.val_metric = val_metric
        self.hidden_layer_sizes = hidden_layer_sizes
        self.duration = duration
        self.args = args
        self.kwargs = kwargs
        self.level = level
        self.regularization_penalty = regularization_penalty
        self.time = None
        self.best_val_metric = None

    def to_dict(self):
        return {
            'loss': self.loss,
            'metric': self.metric,
            'val_loss': self.val_loss,
            'val_metric': self.val_metric,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'duration': self.duration,
            'args': self.args,
            'kwargs': self.kwargs,
            'level': self.level,
            'regularization_penalty': self.regularization_penalty,
            'time': self.time,
            'best_val_metric': self.best_val_metric,
        }

    @staticmethod
    def from_dict(_dict):
        result = AnytimeAlgorithmResult.__new__(AnytimeAlgorithmResult)
        result.loss = _dict['loss']
        result.metric = _dict['metric']
        result.val_loss = _dict['val_loss']
        result.val_metric = _dict['val_metric']
        result.hidden_layer_sizes = _dict['hidden_layer_sizes']
        result.duration = _dict['duration']
        result.args = _dict['args']
        result.kwargs = _dict['kwargs']
        result.level = _dict['level']
        result.regularization_penalty = _dict.get('regularization_penalty')
        result.time = _dict['time']
        result.best_val_metric = _dict['best_val_metric']
        return result


class AnytimeAlgorithm:
    def __init__(self):
        self.results = list()
        self.best_val_metric = -np.inf
        self.start_time = None

    def log_result(self, result, _time=None):
        if result.val_metric > self.best_val_metric:
            self.best_val_metric = result.val_metric
        if _time is None:
            _time = time.time() - self.start_time
        result.time = _time
        result.best_val_metric = self.best_val_metric
        self.results.append(result)

    @staticmethod
    def get_total_duration_from_results(results):
        return sum([result.duration for result in results])

    def get_total_duration(self):
        return self.get_total_duration_from_results(self.results)

    @staticmethod
    def get_mean_duration_from_results(results):
        return np.mean([result.duration for result in results])

    def get_mean_duration(self):
        return self.get_mean_duration_from_results(self.results)

    @staticmethod
    def get_mean_val_metric_from_results(results):
        return np.mean([result.val_metric for result in results])

    def get_mean_val_metric(self):
        return self.get_mean_val_metric_from_results(self.results)

    @staticmethod
    def get_best_val_metric_from_results(results):
        return max([result.val_metric for result in results])

    def get_best_val_metric(self):
        return self.get_best_val_metric_from_results(self.results)

    def run(self):
        self.start_time = time.time()


def identity_postprocess(args, kwargs):
    return args, kwargs
