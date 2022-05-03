import tensorflow as tf
import numpy as np

from models import Sequential, Conv2D, Flatten, Dense


def get_statistics_from_history(history):
    best_epoch_number = np.argmax(history['val_metric'])
    best_loss = history['loss'][best_epoch_number]
    best_metric = history['metric'][best_epoch_number]
    best_val_loss = history['val_loss'][best_epoch_number]
    best_val_metric = history['val_metric'][best_epoch_number]
    best_hidden_layer_sizes = history['hidden_layer_sizes'][best_epoch_number]
    return best_loss, best_metric, best_val_loss, best_val_metric, best_hidden_layer_sizes


def get_statistics_from_histories(histories):
    best_val_losses = list()
    best_val_metrics = list()
    all_best_hidden_layer_sizes = list()

    for history in histories:
        _, _, best_val_loss, best_val_metric, best_hidden_layer_sizes = get_statistics_from_history(history)
        best_val_losses.append(best_val_loss)
        best_val_metrics.append(best_val_metric)
        all_best_hidden_layer_sizes.append(best_hidden_layer_sizes)

    mean_best_val_loss = np.mean(best_val_losses)
    mean_best_val_metric = np.mean(best_val_metrics)
    mean_best_hidden_layer_sizes = [np.mean(layer) for layer in list(zip(*all_best_hidden_layer_sizes))]

    return mean_best_val_loss, mean_best_val_metric, mean_best_hidden_layer_sizes


def cross_validate(train_fn, x, y, n_splits, random_state=42, *args, **kwargs):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    histories = list()
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        xtrain, xtest = x[train_index], x[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        history = train_fn(xtrain, ytrain, validation_data=(xtest, ytest), *args, **kwargs)
        histories.append(history)

        _, _, best_val_loss, best_val_metric, best_hidden_layer_sizes = get_statistics_from_history(history)
        print(
            f"Run {i} completed, best_val_loss: {best_val_loss}, best_val_metric: {best_val_metric}, best_hidden_layer_sizes: {best_hidden_layer_sizes}")

    mean_best_val_loss, mean_best_val_metric, mean_best_hidden_layer_sizes = get_statistics_from_histories(histories)
    print(f'mean_best_val_loss: {mean_best_val_loss}')
    print(f'mean_best_val_metric: {mean_best_val_metric}')
    print(f'mean_best_hidden_layer_sizes: {mean_best_hidden_layer_sizes}')

    return histories, mean_best_hidden_layer_sizes


def hyperparameter_search(train_fn, x, y, validation_data, *args, **kwargs):
    from itertools import product

    all_params = [*args] + list(kwargs.values())
    histories = list()

    best_overall_val_loss = np.inf
    best_overall_val_metric = None
    best_overall_combination = None

    for combination in product(*all_params):
        combination_args = combination[:len(args)]

        combination_kwargs_values = combination[len(args):]
        combination_kwargs = dict(zip(kwargs.keys(), combination_kwargs_values))

        history = train_fn(x, y, validation_data, *combination_args, **combination_kwargs)
        history['parameters'] = combination
        histories.append(history)

        _, _, best_val_loss, best_val_metric, best_hidden_layer_sizes = get_statistics_from_history(history)
        print(
            f"Run with parameters {combination} completed, best_val_loss: {best_val_loss}, best_val_metric: {best_val_metric}, best_hidden_layer_sizes: {best_hidden_layer_sizes}")

        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_val_metric = best_val_metric
            best_overall_combination = combination

    print(f'Best overall combination: {best_overall_combination}, val_metric: {best_overall_val_metric}')

    return histories, best_overall_combination



def merge_histories(history1, history2):
    merged_history = dict()
    for key in history1.keys():
        merged_history[key] = history1[key] + history2[key]
    return merged_history


def get_convolutional_model(x, layer_sizes, output_neurons=10):
    model = Sequential([
        Conv2D(layer_sizes[0], filter_size=(3, 3), activation='selu', strides=(1, 1), padding='SAME',
               kernel_initializer='lecun_normal', input_shape=x[0, :, :, :].shape),
        Conv2D(layer_sizes[1], filter_size=(3, 3), activation='selu', strides=(2, 2), padding='SAME',
               kernel_initializer='lecun_normal'),
        tf.keras.layers.Dropout(0.2),
        Conv2D(layer_sizes[2], filter_size=(3, 3), activation='selu', strides=(1, 1), padding='SAME',
               kernel_initializer='lecun_normal'),
        Conv2D(layer_sizes[3], filter_size=(3, 3), activation='selu', strides=(2, 2), padding='SAME',
               kernel_initializer='lecun_normal'),
        tf.keras.layers.Dropout(0.5),
        Flatten(),
        Dense(layer_sizes[4], activation='selu', kernel_initializer='lecun_normal'),
        Dense(output_neurons, activation='softmax', fixed_size=True),
    ])
    return model


def get_dense_model(x, layer_sizes):
    layers = list()

    layers.append(
        Dense(layer_sizes[0], activation='selu', kernel_initializer='lecun_normal', input_shape=x[0, :].shape))
    for layer_size in layer_sizes[1:]:
        layers.append(Dense(layer_size, activation='selu', kernel_initializer='lecun_normal'))
    layers.append(Dense(1, activation=None, kernel_initializer='lecun_normal', fixed_size=True))

    model = Sequential(layers)
    return model


def train_fn_conv(x, y, validation_data, learning_rate, schedule, layer_sizes, output_neurons=10, min_new_neurons=20,
                  growth_percentage=0.2, verbose=False, use_static_graph=True, batch_size=128):
    model = get_convolutional_model(x, layer_sizes, output_neurons)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                        min_new_neurons=min_new_neurons,
                        validation_data=validation_data, growth_percentage=growth_percentage, verbose=verbose,
                        use_static_graph=use_static_graph)

    return history


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def negative_squared_error(y_true, y_pred):
    return - ((y_true - y_pred) ** 2)


def train_fn_dense(x, y, validation_data, learning_rate, schedule, layer_sizes, min_new_neurons=20,
                   growth_percentage=0.2, verbose=False, use_static_graph=True, batch_size=128):
    model = get_dense_model(x, layer_sizes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                        min_new_neurons=min_new_neurons,
                        validation_data=validation_data, growth_percentage=growth_percentage, verbose=verbose,
                        use_static_graph=use_static_graph,
                        loss_fn=squared_error, metric_fn=negative_squared_error)

    return history


def early_stopping_conv(x, y, validation_data, learning_rate, schedule, layer_sizes, output_neurons=10,
                        min_new_neurons=20,
                        growth_percentage=0.2, verbose=False, use_static_graph=True, batch_size=128, max_setbacks=2):
    assert len(schedule) == 1

    model = get_convolutional_model(x, layer_sizes, output_neurons)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = Sequential.ParameterContainer.prepare_history()

    best_val_loss = np.inf
    n_setbacks = 0
    while True:
        epoch_history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                                  min_new_neurons=min_new_neurons,
                                  validation_data=validation_data, growth_percentage=growth_percentage, verbose=verbose,
                                  use_static_graph=use_static_graph)
        history = merge_histories(history, epoch_history)
        val_loss = epoch_history['val_loss'][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            n_setbacks = 0
        else:
            n_setbacks += 1
            if n_setbacks > max_setbacks:
                break

    return history


def early_stopping_dense(x, y, validation_data, learning_rate, schedule, layer_sizes, output_neurons=1,
                         min_new_neurons=20,
                         growth_percentage=0.2, verbose=False, use_static_graph=True, batch_size=128, max_setbacks=2):
    assert len(schedule) == 1
    assert output_neurons == 1

    model = get_dense_model(x, layer_sizes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = Sequential.ParameterContainer.prepare_history()

    best_val_loss = np.inf
    n_setbacks = 0
    while True:
        epoch_history = model.fit(x=x, y=y, optimizer=optimizer, schedule=schedule, batch_size=batch_size,
                                  min_new_neurons=min_new_neurons,
                                  validation_data=validation_data, growth_percentage=growth_percentage, verbose=verbose,
                                  use_static_graph=use_static_graph,
                                  loss_fn=squared_error, metric_fn=negative_squared_error)
        history = merge_histories(history, epoch_history)
        val_loss = epoch_history['val_loss'][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            n_setbacks = 0
        else:
            n_setbacks += 1
            if n_setbacks > max_setbacks:
                break

    return history


def layer_sizes_join_postprocess(args, kwargs):
    kwargs['layer_sizes'] = kwargs['layer_1_size'], kwargs['layer_2_size'], kwargs['layer_3_size'], kwargs[
        'layer_4_size'], kwargs['layer_5_size']
    del kwargs['layer_1_size'], kwargs['layer_2_size'], kwargs['layer_3_size'], kwargs['layer_4_size'], kwargs[
        'layer_5_size']
    return args, kwargs
