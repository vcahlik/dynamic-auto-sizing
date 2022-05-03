import tensorflow as tf
import numpy as np
import copy
import time
import datetime

from ..helpers import get_dense_model, get_convolutional_model, squared_error, negative_squared_error
from main import AnytimeAlgorithm, AnytimeAlgorithmResult, Range, identity_postprocess, interruptible
from helpers import get_data_for_run, save_results


class ProgressiveModelGrowth(AnytimeAlgorithm):
    @staticmethod
    def sample_combination(values):
        combination = list()
        for value in values:
            if isinstance(value, Range):
                combination.append(value.sample())
            else:
                combination.append(value)
        return tuple(combination)

    def _progressive_growth(self, model_type, x, y, validation_data, combination_args, combination_kwargs,
                            min_regularization_penalty, regularization_penalty_multiplier, max_stall_epochs,
                            learning_rate, schedule, layer_sizes, output_neurons, min_new_neurons=20,
                            growth_percentage=0.2,
                            verbose=False, use_static_graph=True, batch_size=128):
        assert len(schedule) == 1
        assert model_type in ('dense', 'convolutional')

        schedule = copy.deepcopy(schedule)
        start_time = time.time()

        if model_type == 'dense':
            assert output_neurons == 1
            model = get_dense_model(x, layer_sizes)
        else:
            model = get_convolutional_model(x, layer_sizes, output_neurons)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        combination = combination_args + tuple(combination_kwargs.values())
        print(f"Run with parameters {combination} started...")

        best_val_loss = np.inf
        stall_epochs = 0
        regularization_penalty = schedule.epochs[0].regularization_penalty
        while True:
            epoch_start_time = time.time()
            if model_type == 'dense':
                epoch_history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                                          min_new_neurons=min_new_neurons,
                                          validation_data=validation_data, growth_percentage=growth_percentage,
                                          verbose=verbose,
                                          use_static_graph=use_static_graph, loss_fn=squared_error,
                                          metric_fn=negative_squared_error)
            else:
                epoch_history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                                          min_new_neurons=min_new_neurons,
                                          validation_data=validation_data, growth_percentage=growth_percentage,
                                          verbose=verbose,
                                          use_static_graph=use_static_graph)
            epoch_duration = time.time() - epoch_start_time
            result = AnytimeAlgorithmResult(loss=epoch_history['loss'][0], metric=epoch_history['metric'][0],
                                            val_loss=epoch_history['val_loss'][0],
                                            val_metric=epoch_history['val_metric'][0],
                                            hidden_layer_sizes=epoch_history['hidden_layer_sizes'][0],
                                            duration=epoch_duration, args=combination_args, kwargs=combination_kwargs)
            self.log_result(result)
            print(
                f"Next epoch, val_loss {epoch_history['val_loss'][0]}, val_metric {epoch_history['val_metric'][0]}, regularization_penalty {regularization_penalty}, hidden_layer_sizes {epoch_history['hidden_layer_sizes'][0]}")
            if epoch_history['val_loss'][0] >= best_val_loss:
                if stall_epochs == 0:
                    print("Training stalled...")
                    regularization_penalty = regularization_penalty * regularization_penalty_multiplier
                    schedule.epochs[0].regularization_penalty = regularization_penalty
                stall_epochs += 1
            else:
                if stall_epochs > 0:
                    print("Stall ended.")
                best_val_loss = epoch_history['val_loss'][0]
                stall_epochs = 0
            if regularization_penalty < min_regularization_penalty or stall_epochs > max_stall_epochs:
                break

        duration = time.time() - start_time
        print(
            f"Run with parameters {combination} completed, duration {round(duration, 1)}, best_val_loss: {best_val_loss}")

    @interruptible
    def run(self, model_type, x, y, validation_data=None, postprocess_fn=identity_postprocess, fraction=None,
            test_size=None, save=False,
            max_duration=None, *args, **kwargs):
        assert model_type in ('dense', 'convolutional')
        super().run()

        filename = f"results/RandomSearch{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        histories = list()

        while True:
            combination_args = self.sample_combination([*args])
            combination_kwargs = dict(zip(kwargs.keys(), self.sample_combination(list(kwargs.values()))))
            combination_args, combination_kwargs = postprocess_fn(combination_args, combination_kwargs)

            x_iter, y_iter, validation_data_iter = get_data_for_run(x, y, validation_data, fraction, test_size)

            self._progressive_growth(
                model_type, x_iter, y_iter, validation_data_iter, combination_args,
                combination_kwargs, *combination_args, **combination_kwargs)

            print(f'Total duration {round(self.get_total_duration(), 1)}')

            if save:
                save_results([result.to_dict() for result in self.results], full_name=filename)

            if max_duration is not None and self.get_total_duration() > max_duration:
                break
