import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import copy


dtype = 'float32'
tf.keras.backend.set_floatx(dtype)


class Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self):
        self.n_new_neurons = 0
        self.scaling_tensor = None
        self.set_regularization_penalty(0.)
        self.set_regularization_method(None)

    def copy(self):
        regularizer_copy = Regularizer.__new__(Regularizer)
        regularizer_copy.n_new_neurons = self.n_new_neurons
        regularizer_copy.scaling_tensor = self.scaling_tensor
        regularizer_copy.set_regularization_penalty(self.regularization_penalty)
        regularizer_copy.set_regularization_method(self.regularization_method)
        return regularizer_copy

    def __call__(self, x):
        if self.regularization_method is None or self.regularization_penalty == 0:
            return 0
        elif self.regularization_method == 'weighted_l1':
            return self.weighted_l1(x)
        elif self.regularization_method == 'weighted_l1_reordered':
            return self.weighted_l1_reordered(x)
        elif self.regularization_method == 'group_sparsity':
            return self.group_sparsity(x)
        elif self.regularization_method == 'l1':
            return self.l1(x)
        else:
            raise NotImplementedError(f"Unknown regularization method {self.regularization_method}")

    def weighted_l1(self, x):
        # I.e. for a parameter matrix of 4 input and 10 output neurons:
        #
        # [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        #
        # the scaling tensor, as well as the resulting weighted values, could be:
        #
        # [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        #  [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        #  [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        #  [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]]
        #
        # Therefore every additional output neuron is regularized more.

        scaling_tensor = tf.cumsum(tf.constant(self.regularization_penalty, shape=x.shape, dtype=dtype), axis=-1)
        weighted_values = scaling_tensor * tf.abs(x)
        return tf.reduce_sum(weighted_values)

    def weighted_l1_reordered(self, x):
        if self.update_scaling_tensor:
            scaling_tensor_raw = tf.cumsum(tf.constant(self.regularization_penalty, shape=x.shape, dtype=dtype),
                                           axis=-1)

            scaling_tensor_old_neurons = scaling_tensor_raw[:, :-self.n_new_neurons]
            scaling_tensor_new_neurons = scaling_tensor_raw[:, -self.n_new_neurons:]
            scaling_tensor_old_neurons_shuffled = tf.transpose(
                tf.random.shuffle(tf.transpose(scaling_tensor_old_neurons)))
            self.scaling_tensor = tf.concat([scaling_tensor_old_neurons_shuffled, scaling_tensor_new_neurons], axis=-1)
            self.update_scaling_tensor = False

        weighted_values = self.scaling_tensor * tf.abs(x)
        return tf.reduce_sum(weighted_values)

    def group_sparsity(self, x):
        # I.e. for a parameter matrix of 3 input and 5 output neurons:
        #
        # [[1., 1., 1., 1., 1.],
        #  [1., 2., 2., 1., 2.],
        #  [2., 2., 3., 1., 3.]]
        #
        # The resulting vector of group norms is [2., 2., 3., 1., 3.], therefore for
        # every output neuron, its incoming connections form a group.

        group_norms = tf.norm(x, ord=2, axis=0)
        # assert group_norms.shape[0] == x.shape[1]
        return self.regularization_penalty * tf.reduce_sum(group_norms)

    def l1(self, x):
        weighted_values = self.regularization_penalty * tf.abs(x)
        return tf.reduce_sum(weighted_values)

    def prune(self):
        self.n_new_neurons = 0
        if self.regularization_method == 'weighted_l1_reordered':
            self.update_scaling_tensor = True

    def grow(self, n_new_neurons):
        self.n_new_neurons = n_new_neurons
        if self.regularization_method == 'weighted_l1_reordered':
            self.update_scaling_tensor = True

    def set_regularization_penalty(self, regularization_penalty):
        self.regularization_penalty = regularization_penalty

    def set_regularization_method(self, regularization_method):
        self.regularization_method = regularization_method
        if self.regularization_method == 'weighted_l1_reordered':
            self.update_scaling_tensor = True
        else:
            self.update_scaling_tensor = None

    def get_config(self):
        return {'regularization_penalty': float(self.regularization_penalty)}


class DASLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super().__init__()

        self._input_shape = input_shape
        self._built = False


class Dense(DASLayer):
    def __init__(self, units, activation, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', input_shape=None, fixed_size=False,
                 regularizer=None):
        super().__init__(input_shape)

        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.fixed_size = fixed_size

        self.A = tf.keras.activations.get(activation)
        self.W_init = tf.keras.initializers.get(kernel_initializer)
        self.b_init = tf.keras.initializers.get(bias_initializer)
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = Regularizer()

    def copy(self):
        layer_copy = Dense.__new__(Dense)
        super(Dense, layer_copy).__init__(self._input_shape)

        layer_copy.units = self.units
        layer_copy.activation = self.activation
        layer_copy.kernel_initializer = self.kernel_initializer
        layer_copy.bias_initializer = self.bias_initializer
        layer_copy.fixed_size = self.fixed_size

        layer_copy.A = self.A
        layer_copy.W_init = self.W_init
        layer_copy.b_init = self.b_init
        layer_copy.regularizer = self.regularizer.copy()

        layer_copy.W = tf.Variable(
            name='W',
            initial_value=self.W,
            trainable=True)

        layer_copy.b = tf.Variable(
            name='b',
            initial_value=self.b,
            trainable=True)

        layer_copy.add_regularizer_loss()

        layer_copy._built = True
        return layer_copy

    def build(self, input_shape):
        if self._built:
            return

        input_units = input_shape[-1]

        self.W = tf.Variable(
            name='W',
            initial_value=self.W_init(shape=(input_units, self.units), dtype=dtype),
            trainable=True)

        self.b = tf.Variable(
            name='b',
            initial_value=self.b_init(shape=(self.units,), dtype=dtype),
            trainable=True)

        self.add_regularizer_loss()

        self._built = True

    def call(self, inputs, training=None):
        return self.A(tf.matmul(inputs, self.W) + self.b)

    def add_regularizer_loss(self):
        self.add_loss(lambda: self.regularizer(tf.concat([self.W, tf.reshape(self.b, (1, -1))], axis=0)))

    def get_size(self):
        return self.W.shape[0], self.W.shape[1]

    def prune(self, threshold, active_input_units_indices):
        # Remove connections from pruned units in previous layer
        new_W = tf.gather(self.W.value(), active_input_units_indices, axis=0)

        if self.fixed_size:
            active_output_neurons_indices = list(range(new_W.shape[1]))
        else:
            # Prune units in this layer
            weights_with_biases = tf.concat([new_W, tf.reshape(self.b.value(), (1, -1))], axis=0)
            neurons_are_active = tf.math.reduce_max(tf.abs(weights_with_biases), axis=0) >= threshold
            active_output_neurons_indices = tf.reshape(tf.where(neurons_are_active), (-1,))

            new_W = tf.gather(new_W, active_output_neurons_indices, axis=1)
            new_b = tf.gather(self.b.value(), active_output_neurons_indices, axis=0)

            self.b = tf.Variable(name='b', initial_value=new_b, trainable=True)

        self.W = tf.Variable(name='W', initial_value=new_W, trainable=True)

        self.regularizer.prune()
        return active_output_neurons_indices

    def grow(self, n_new_input_units, percentage, min_new_units, scaling_factor):
        if n_new_input_units > 0:
            # Add connections to grown units in previous layer
            W_growth = self.W_init(shape=(self.W.shape[0] + n_new_input_units, self.W.shape[1]), dtype=dtype)[
                       -n_new_input_units:,
                       :] * scaling_factor  # TODO is it better to be multiplying here by scaling_factor? It does help with not increasing the max weights of existing neurons when new neurons are added.
            new_W = tf.concat([self.W.value(), W_growth], axis=0)
        else:
            new_W = self.W.value()

        if self.fixed_size:
            n_new_output_units = 0
        else:
            # Grow new units in this layer
            n_new_output_units = max(min_new_units, int(new_W.shape[1] * percentage))
            if n_new_output_units > 0:
                W_growth = self.W_init(shape=(new_W.shape[0], new_W.shape[1] + n_new_output_units), dtype=dtype)[:,
                           -n_new_output_units:] * scaling_factor
                b_growth = self.b_init(shape=(n_new_output_units,),
                                       dtype=dtype)  # TODO for all possible bias initializers to work properly, the whole bias vector should be initialized at once
                new_W = tf.concat([new_W, W_growth], axis=1)
                new_b = tf.concat([self.b.value(), b_growth], axis=0)

                self.b = tf.Variable(name='b', initial_value=new_b, trainable=True)

        self.W = tf.Variable(name='W', initial_value=new_W, trainable=True)

        self.regularizer.grow(n_new_output_units)
        return n_new_output_units

    def mutate(self, mutation_strength):
        self.W.assign_add(tf.random.normal(self.W.shape, mean=0.0, stddev=mutation_strength))
        self.b.assign_add(tf.random.normal(self.b.shape, mean=0.0, stddev=mutation_strength))

    def set_regularization_penalty(self, regularization_penalty):
        if not self.fixed_size:
            self.regularizer.set_regularization_penalty(regularization_penalty)

    def set_regularization_method(self, regularization_method):
        if not self.fixed_size:
            self.regularizer.set_regularization_method(regularization_method)

    def get_param_string(self):
        param_string = ""
        weights_with_bias = tf.concat([self.W, tf.reshape(self.b, (1, -1))], axis=0)
        max_parameters = tf.math.reduce_max(tf.abs(weights_with_bias), axis=0).numpy()
        magnitudes = np.floor(np.log10(max_parameters))
        for m in magnitudes:
            if m > 0:
                m = 0
            param_string += str(int(-m))
        return param_string


class Conv2D(DASLayer):
    def __init__(self, filters, filter_size, activation, strides=(1, 1),
                 padding='SAME', kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', input_shape=None, fixed_size=False,
                 regularizer=None):
        super().__init__(input_shape)

        self.filters = filters
        self.filter_size = filter_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.fixed_size = fixed_size

        self.A = tf.keras.activations.get(activation)
        self.F_init = tf.keras.initializers.get(kernel_initializer)
        self.b_init = tf.keras.initializers.get(bias_initializer)
        if regularizer is not None:
            self.regularizer = regularizer
        else:
            self.regularizer = Regularizer()

    def copy(self):
        layer_copy = Conv2D.__new__(Conv2D)
        super(Conv2D, layer_copy).__init__(self._input_shape)

        layer_copy.filters = self.filters
        layer_copy.filter_size = self.filter_size
        layer_copy.activation = self.activation
        layer_copy.strides = self.strides
        layer_copy.padding = self.padding
        layer_copy.kernel_initializer = self.kernel_initializer
        layer_copy.bias_initializer = self.bias_initializer
        layer_copy.fixed_size = self.fixed_size

        layer_copy.A = self.A
        layer_copy.F_init = self.F_init
        layer_copy.b_init = self.b_init
        layer_copy.regularizer = self.regularizer.copy()

        layer_copy.F = tf.Variable(
            name='F',
            initial_value=self.F,
            trainable=True)

        layer_copy.b = tf.Variable(
            name='b',
            initial_value=self.b,
            trainable=True)

        layer_copy.add_regularizer_loss()

        layer_copy._built = True
        return layer_copy

    def build(self, input_shape):
        if self._built:
            return

        input_filters = input_shape[-1]

        self.F = tf.Variable(
            name='F',
            initial_value=self.F_init(
                shape=(self.filter_size[0], self.filter_size[1], input_filters, self.filters), dtype=dtype
            ),
            trainable=True)

        self.b = tf.Variable(
            name='b',
            initial_value=self.b_init(shape=(self.filters,), dtype=dtype),
            trainable=True)

        self.add_regularizer_loss()

        self._built = True

    def call(self, inputs, training=None):
        y = tf.nn.conv2d(inputs, self.F, strides=self.strides, padding=self.padding)
        y = tf.nn.bias_add(y, self.b)
        y = self.A(y)
        return y

    def add_regularizer_loss(self):
        self.add_loss(lambda: self.regularizer(
            tf.concat([tf.reshape(self.F, (-1, self.F.shape[-1])), tf.reshape(self.b, (1, -1))], axis=0)))

    def get_size(self):
        return self.F.shape[-2], self.F.shape[-1]

    def prune(self, threshold, active_input_units_indices):
        # Remove connections from pruned units in previous layer
        new_F = tf.gather(self.F.value(), active_input_units_indices, axis=-2)

        if self.fixed_size:
            active_output_filters_indices = list(range(new_F.shape[-1]))
        else:
            # Prune units in this layer
            F_reduced_max = tf.reshape(tf.math.reduce_max(tf.abs(new_F), axis=(0, 1, 2)), (1, -1))
            F_reduced_max_with_biases = tf.concat([F_reduced_max, tf.reshape(self.b.value(), (1, -1))], axis=0)
            filters_are_active = tf.math.reduce_max(tf.abs(F_reduced_max_with_biases), axis=0) >= threshold
            active_output_filters_indices = tf.reshape(tf.where(filters_are_active), (-1,))

            new_F = tf.gather(new_F, active_output_filters_indices, axis=-1)
            new_b = tf.gather(self.b.value(), active_output_filters_indices, axis=0)

            self.b = tf.Variable(name='b', initial_value=new_b, trainable=True)

        self.F = tf.Variable(name='F', initial_value=new_F, trainable=True)

        self.regularizer.prune()
        return active_output_filters_indices

    def grow(self, n_new_input_units, percentage, min_new_units, scaling_factor):
        if n_new_input_units > 0:
            # Add connections to grown units in previous layer
            F_growth = self.F_init(
                shape=(self.F.shape[0], self.F.shape[1], self.F.shape[2] + n_new_input_units, self.F.shape[3]),
                dtype=dtype)[:, :, -n_new_input_units:,
                       :] * scaling_factor  # TODO is it better to be multiplying here by scaling_factor? It does help with not increasing the max weights of existing neurons when new neurons are added.
            new_F = tf.concat([self.F.value(), F_growth], axis=-2)
        else:
            new_F = self.F.value()

        if self.fixed_size:
            n_new_output_units = 0
        else:
            # Grow new units in this layer
            n_new_output_units = max(min_new_units, int(new_F.shape[-1] * percentage))
            if n_new_output_units > 0:
                F_growth = self.F_init(
                    shape=(new_F.shape[0], new_F.shape[1], new_F.shape[2], new_F.shape[3] + n_new_output_units),
                    dtype=dtype)[:, :, :, -n_new_output_units:] * scaling_factor
                b_growth = self.b_init(shape=(n_new_output_units,),
                                       dtype=dtype)  # TODO for all possible bias initializers to work properly, the whole bias vector should be initialized at once
                new_F = tf.concat([new_F, F_growth], axis=-1)
                new_b = tf.concat([self.b.value(), b_growth], axis=0)

                self.b = tf.Variable(name='b', initial_value=new_b, trainable=True)

        self.F = tf.Variable(name='F', initial_value=new_F, trainable=True)

        self.regularizer.grow(n_new_output_units)
        return n_new_output_units

    def mutate(self, mutation_strength):
        self.F.assign_add(tf.random.normal(self.F.shape, mean=0.0, stddev=mutation_strength))
        self.b.assign_add(tf.random.normal(self.b.shape, mean=0.0, stddev=mutation_strength))

    def set_regularization_penalty(self, regularization_penalty):
        if not self.fixed_size:
            self.regularizer.set_regularization_penalty(regularization_penalty)

    def set_regularization_method(self, regularization_method):
        if not self.fixed_size:
            self.regularizer.set_regularization_method(regularization_method)

    def get_param_string(self):
        param_string = ""
        # TODO
        return param_string


class Flatten(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        return tf.reshape(tf.transpose(inputs, perm=[0, 3, 1, 2]), (inputs.shape[0], -1))


class Sequential(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()

        self.lrs = layers

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.lrs:
            x = layer(x, training=training)
        return x

    def copy(self):
        copied_layers = list()
        for layer in self.lrs:
            if isinstance(layer, DASLayer):
                layer_copy = layer.copy()
            else:
                layer_copy = copy.deepcopy(layer)
            copied_layers.append(layer_copy)

        model_copy = Sequential(copied_layers)
        return model_copy

    def get_layer_input_shape(self, target_layer):
        if target_layer._input_shape is not None:
            return target_layer._input_shape

        input = np.random.normal(size=(1,) + self.lrs[0]._input_shape)
        for layer in self.lrs:
            if layer is target_layer:
                return tuple(input.shape[1:])
            input = layer(input)
        raise Exception("Layer not found in the model.")

    def get_layer_output_shape(self, target_layer):
        input = np.random.normal(size=(1,) + self.lrs[0]._input_shape)
        for layer in self.lrs:
            output = layer(input)
            if layer is target_layer:
                return tuple(output.shape[1:])
            input = output
        raise Exception("Layer not found in the model.")

    def get_layer_sizes(self):
        """
        Returns the sizes of all layers in the model, including the input and output layer.
        """
        layer_sizes = list()
        first_layer = True
        for l in range(len(self.lrs)):
            layer = self.lrs[l]
            if isinstance(layer, DASLayer):
                layer_size = layer.get_size()
                if first_layer:
                    layer_sizes.append(layer_size[0])
                    first_layer = False
                layer_sizes.append(layer_size[1])
        return layer_sizes

    def get_hidden_layer_sizes(self):
        return self.get_layer_sizes()[1:-1]

    def get_regularization_penalty(self):
        # TODO improve
        return self.lrs[-2].regularizer.regularization_penalty

    def set_regularization_penalty(self, regularization_penalty):
        for layer in self.lrs:
            if isinstance(layer, DASLayer) and not layer.fixed_size:
                layer.set_regularization_penalty(regularization_penalty)

    def set_regularization_method(self, regularization_method):
        for layer in self.lrs:
            if isinstance(layer, DASLayer) and not layer.fixed_size:
                layer.set_regularization_method(regularization_method)

    def prune(self, params):
        input_shape = self.get_layer_input_shape(self.lrs[0])
        n_input_units = input_shape[-1]
        active_units_indices = list(range(n_input_units))

        last_custom_layer = None
        for layer in self.lrs:
            if isinstance(layer, DASLayer):
                if last_custom_layer is not None and type(last_custom_layer) != type(layer):
                    if type(last_custom_layer) == Conv2D and type(layer) == Dense:
                        convolutional_shape = self.get_layer_output_shape(last_custom_layer)
                        active_units_indices = self.convert_channel_indices_to_flattened_indices(active_units_indices,
                                                                                                 convolutional_shape)
                    else:
                        raise Exception("Incorrect order of custom layer types.")
                active_units_indices = layer.prune(params.pruning_threshold, active_units_indices)
                last_custom_layer = layer

    def grow(self, params):
        n_new_units = 0

        last_custom_layer = None
        for layer in self.lrs:
            if isinstance(layer, DASLayer):
                if last_custom_layer is not None and type(last_custom_layer) != type(layer):
                    if type(last_custom_layer) == Conv2D and type(layer) == Dense:
                        convolutional_shape = self.get_layer_output_shape(last_custom_layer)
                        n_new_units = n_new_units * convolutional_shape[0] * convolutional_shape[1]
                    else:
                        raise Exception("Incorrect order of custom layer types.")
                n_new_units = layer.grow(n_new_units, params.growth_percentage, min_new_units=params.min_new_neurons,
                                         scaling_factor=params.pruning_threshold)
                last_custom_layer = layer

    def mutate(self, mutation_strength):
        for layer in self.lrs:
            if isinstance(layer, DASLayer):
                layer.mutate(mutation_strength)

    @staticmethod
    def convert_channel_indices_to_flattened_indices(channel_indices, convolutional_shape):
        dense_indices = list()
        units_per_channel = convolutional_shape[0] * convolutional_shape[1]
        for channel_index in channel_indices:
            for iter in range(units_per_channel):
                dense_indices.append(channel_index * units_per_channel + iter)
        return dense_indices

    def print_neurons(self):
        for layer in self.lrs[:-1]:
            print(layer.get_param_string())

    def evaluate(self, params, summed_training_loss, summed_training_metric):
        # Calculate training loss and metric
        if summed_training_loss is not None:
            loss = summed_training_loss / params.x.shape[0]
        else:
            loss = None

        if summed_training_metric is not None:
            metric = summed_training_metric / params.x.shape[0]
        else:
            metric = None

        # Calculate val loss and metric
        summed_val_loss = 0
        summed_val_metric = 0
        n_val_instances = 0

        for step, (x_batch, y_batch) in enumerate(params.val_dataset):
            # y_pred = tf.reshape(self(x_batch, training=False), y_batch.shape)
            y_pred = self(x_batch, training=False)
            summed_val_loss += tf.reduce_sum(params.loss_fn(y_batch, y_pred))
            summed_val_metric += float(tf.reduce_sum(params.metric_fn(y_batch, y_pred)))
            n_val_instances += x_batch.shape[0]

        val_loss = summed_val_loss / n_val_instances
        val_metric = summed_val_metric / n_val_instances

        return loss, metric, val_loss, val_metric

    def list_params(self):
        trainable_count = np.sum([K.count_params(w) for w in self.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.non_trainable_weights])
        total_count = trainable_count + non_trainable_count

        print('Total params: {:,}'.format(total_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

        return total_count, trainable_count, non_trainable_count

    def print_epoch_statistics(self, params, summed_training_loss, summed_training_metric, message=None,
                               require_result=False):
        if not params.verbose:
            if require_result:
                return self.evaluate(params, summed_training_loss, summed_training_metric)
            else:
                return

        loss, metric, val_loss, val_metric = self.evaluate(params, summed_training_loss, summed_training_metric)

        if message is not None:
            print(message)

        print(
            f"loss: {loss} - metric: {metric} - val_loss: {val_loss} - val_metric: {val_metric} - penalty: {self.get_regularization_penalty()}")
        hidden_layer_sizes = self.get_hidden_layer_sizes()
        print(f"hidden layer sizes: {hidden_layer_sizes}, total units: {sum(hidden_layer_sizes)}")
        if params.print_neurons:
            self.print_neurons()

        if require_result:
            return loss, metric, val_loss, val_metric

    def update_history(self, params, loss, metric, val_loss, val_metric):
        params.history['loss'].append(float(loss))
        params.history['metric'].append(float(metric))
        params.history['val_loss'].append(float(val_loss))
        params.history['val_metric'].append(float(val_metric))
        params.history['hidden_layer_sizes'].append(self.get_hidden_layer_sizes())

    @staticmethod
    def prepare_datasets(x, y, batch_size, validation_data):
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=20000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
        return train_dataset.prefetch(tf.data.AUTOTUNE), val_dataset.prefetch(tf.data.AUTOTUNE)

    def manage_dynamic_regularization(self, params, val_loss):
        if val_loss >= params.best_conditional_val_loss * params.stall_coefficient:
            # Training is currently in stall
            if not params.training_stalled:
                penalty = self.get_regularization_penalty() * params.regularization_penalty_multiplier
                print("Changing penalty...")
                # TODO this must be modified, penalty can differ for each layer
                self.set_regularization_penalty(penalty)
                params.training_stalled = True
        else:
            params.best_conditional_val_loss = val_loss
            params.training_stalled = False

    def grow_wrapper(self, params):
        dynamic_reqularization_active = params.regularization_penalty_multiplier != 1.
        if dynamic_reqularization_active:
            loss, metric, val_loss, val_metric = self.print_epoch_statistics(params, None, None, "Before growing:",
                                                                             require_result=True)
            self.manage_dynamic_regularization(params, val_loss)
        else:
            self.print_epoch_statistics(params, None, None, "Before growing:")

        self.grow(params)
        self.print_epoch_statistics(params, None, None, "After growing:")

    def prune_wrapper(self, params, summed_loss, summed_metric):
        loss, metric, _, _ = self.print_epoch_statistics(params, summed_loss, summed_metric, "Before pruning:",
                                                         require_result=True)
        self.prune(params)
        _, _, val_loss, val_metric = self.print_epoch_statistics(params, None, None, "After pruning:",
                                                                 require_result=True)
        self.update_history(params, loss, metric, val_loss, val_metric)

    class ParameterContainer:
        def __init__(self, x, y, optimizer, batch_size, min_new_neurons, validation_data, pruning_threshold,
                     regularization_penalty_multiplier,
                     stall_coefficient, growth_percentage, mini_epochs_per_epoch, verbose, print_neurons,
                     use_static_graph, loss_fn, metric_fn):
            self.x = x
            self.y = y
            self.optimizer = optimizer
            self.batch_size = batch_size
            self.min_new_neurons = min_new_neurons
            self.validation_data = validation_data
            self.pruning_threshold = pruning_threshold
            self.regularization_penalty_multiplier = regularization_penalty_multiplier
            self.stall_coefficient = stall_coefficient
            self.growth_percentage = growth_percentage
            self.mini_epochs_per_epoch = mini_epochs_per_epoch
            self.verbose = verbose
            self.print_neurons = print_neurons
            self.use_static_graph = use_static_graph
            self.loss_fn = loss_fn
            self.metric_fn = metric_fn

            self.train_dataset, self.val_dataset = Sequential.prepare_datasets(x, y, batch_size, validation_data)
            self.history = self.prepare_history()

            self.best_conditional_val_loss = np.inf
            self.training_stalled = False

        @staticmethod
        def prepare_history():
            history = {
                'loss': list(),
                'metric': list(),
                'val_loss': list(),
                'val_metric': list(),
                'hidden_layer_sizes': list(),
            }
            return history

    def fit_single_step(self, x_batch, y_batch, optimizer, loss_fn, metric_fn):
        with tf.GradientTape() as tape:
            # y_pred = tf.reshape(self(x_batch, training=True), y_batch.shape)
            y_pred = self(x_batch, training=True)
            raw_loss = loss_fn(y_batch, y_pred)
            loss_value = tf.reduce_mean(raw_loss)
            loss_value += sum(self.losses)  # Add losses registered by model.add_loss

            loss = tf.reduce_sum(raw_loss)
            metric = float(tf.reduce_sum(metric_fn(y_batch, y_pred)))

        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss, metric

    def fit_single_epoch(self, params):
        summed_loss = 0
        summed_metric = 0

        for mini_epoch in range(params.mini_epochs_per_epoch):
            summed_loss = 0
            summed_metric = 0

            if params.use_static_graph:
                fit_single_step_function = tf.function(self.fit_single_step)
            else:
                fit_single_step_function = self.fit_single_step
            for step, (x_batch, y_batch) in enumerate(params.train_dataset):
                loss, metric = fit_single_step_function(x_batch, y_batch, params.optimizer, params.loss_fn,
                                                        params.metric_fn)
                summed_loss += loss
                summed_metric += metric

        return summed_loss, summed_metric

    def fit(self, x, y, optimizer, schedule, batch_size, min_new_neurons, validation_data, pruning_threshold=0.001,
            regularization_penalty_multiplier=1.,
            stall_coefficient=1, growth_percentage=0.2, mini_epochs_per_epoch=1, verbose=True, print_neurons=False,
            use_static_graph=True,
            loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
            metric_fn=tf.keras.metrics.sparse_categorical_accuracy):
        params = self.ParameterContainer(x=x, y=y, optimizer=optimizer, batch_size=batch_size,
                                         min_new_neurons=min_new_neurons, validation_data=validation_data,
                                         pruning_threshold=pruning_threshold,
                                         regularization_penalty_multiplier=regularization_penalty_multiplier,
                                         stall_coefficient=stall_coefficient,
                                         growth_percentage=growth_percentage,
                                         mini_epochs_per_epoch=mini_epochs_per_epoch, verbose=verbose,
                                         print_neurons=print_neurons,
                                         use_static_graph=use_static_graph, loss_fn=loss_fn, metric_fn=metric_fn)
        self.build(x.shape)  # Necessary when verbose == False

        for epoch_no, epoch in enumerate(schedule):
            if verbose:
                print("##########################################################")
                print(f"Epoch {epoch_no + 1}/{len(schedule)}")

            self.set_regularization_penalty(epoch.regularization_penalty)
            self.set_regularization_method(epoch.regularization_method)

            if epoch.grow:
                self.grow_wrapper(params)

            summed_loss, summed_metric = self.fit_single_epoch(params)

            if epoch.prune:
                self.prune_wrapper(params, summed_loss, summed_metric)
            else:
                loss, metric, val_loss, val_metric = self.print_epoch_statistics(params, summed_loss, summed_metric,
                                                                                 require_result=True)
                self.update_history(params, loss, metric, val_loss, val_metric)

        return params.history
