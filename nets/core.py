from .utils import measure_accuracy

import numpy as np
import math


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def get_scaling_matrix(l1_term, n_output_neurons, self_scale_coef):
    if self_scale_coef:
        scaling_matrix = np.cumprod(np.full((n_output_neurons,), self_scale_coef)).reshape((n_output_neurons, 1))
        scaling_matrix = scaling_matrix / scaling_matrix[-1] * l1_term
    else:
        scaling_matrix = np.linspace(0, l1_term * (n_output_neurons-1), n_output_neurons).reshape((1, -1)).T
    return scaling_matrix


def compute_cost(AL, Y, parameters, l1_term, self_scale, self_scale_coef):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    L = len(parameters) // 2  # number of layers in the neural network

    # Compute loss from aL and y.
    error_cost = -1 / m * np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL)))
    reg_cost = 0
    for l in range(L):
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]
        if self_scale:
            scaling_matrix = get_scaling_matrix(l1_term, W.shape[0], self_scale_coef)
            reg_cost += 1 / m * np.sum(scaling_matrix * np.abs(W))
            reg_cost += 1 / m * np.sum(scaling_matrix * np.abs(b))
        else:
            reg_cost += l1_term / m * np.sum(np.abs(W))
            reg_cost += l1_term / m * np.sum(np.abs(b))
    cost = error_cost + reg_cost

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache, l1_term, self_scale, self_scale_coef):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    if self_scale:
        scaling_matrix = get_scaling_matrix(l1_term, W.shape[0], self_scale_coef)
        dW = 1 / m * (dZ.dot(A_prev.T)) + scaling_matrix / m * np.sign(W)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True) + scaling_matrix / m * np.sign(b)
    else:
        dW = 1 / m * (dZ.dot(A_prev.T)) + l1_term / m * np.sign(W)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True) + l1_term / m * np.sign(b)
    dA_prev = W.T.dot(dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, l1_term, self_scale, self_scale_coef):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, l1_term, self_scale, self_scale_coef)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, l1_term, self_scale, self_scale_coef)

    return dA_prev, dW, db


def backward(AL, Y, caches, l1_term, self_scale, self_scale_coef):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
        dAL, current_cache, "sigmoid", l1_term, self_scale, self_scale_coef
    )

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu", l1_term, self_scale, self_scale_coef
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def train_model(
        X,
        Y,
        parameters,
        learning_rate=0.01,
        l1_term=0,
        self_scale=False,
        self_scale_coef=None,
        num_epochs=100,
        print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # Loop (gradient descent)
    for epoch_no in range(0, num_epochs):
        mini_batches = random_mini_batches(X, Y, 64)

        for X_batch, y_batch in mini_batches:
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = forward(X_batch, parameters)

            # Compute cost.
            cost = compute_cost(AL, y_batch, parameters, l1_term, self_scale, self_scale_coef)

            # Backward propagation.
            grads = backward(AL, y_batch, caches, l1_term, self_scale, self_scale_coef)

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost:
            print("Cost after epoch %i: %f" % (epoch_no, cost))

    return parameters


def neuron_is_inactive(idx, W, b, threshold):
    W_is_inactive = np.amax(np.abs(W), 1)[idx] < threshold
    b_is_inactive = b[idx] < threshold
    return W_is_inactive and b_is_inactive


def prune_neurons(parameters, threshold=0.001):
    L = len(parameters) // 2

    for l in range(L - 1):
        W_cur = parameters["W" + str(l + 1)]
        b_cur = parameters["b" + str(l + 1)]
        W_next = parameters["W" + str(l + 2)]

        n_neurons = W_cur.shape[0]
        for idx in reversed(range(0, n_neurons, 1)):
            if neuron_is_inactive(idx, W_cur, b_cur, threshold):
                # Delete the neuron
                W_cur = np.delete(W_cur, idx, axis=0)
                b_cur = np.delete(b_cur, idx, axis=0)
                W_next = np.delete(W_next, idx, axis=1)

        parameters["W" + str(l + 1)] = W_cur
        parameters["b" + str(l + 1)] = b_cur
        parameters["W" + str(l + 2)] = W_next


def grow_neurons(parameters, scaling_factor=0.1):
    L = len(parameters) // 2

    for l in range(L - 1):
        W_cur = parameters["W" + str(l + 1)]
        b_cur = parameters["b" + str(l + 1)]
        W_next = parameters["W" + str(l + 2)]

        n_neurons = W_cur.shape[0]
        n_new_neurons = max(n_neurons // 10, 5)

        # Grow the neurons
        W_cur = np.concatenate((W_cur, np.random.randn(n_new_neurons, W_cur.shape[1]) * scaling_factor * np.sqrt(2/W_cur.shape[1])), axis=0)
        b_cur = np.concatenate((b_cur, np.zeros((n_new_neurons, 1))), axis=0)
        W_next = np.concatenate((W_next, np.random.randn(W_next.shape[0], n_new_neurons) * np.sqrt(2/(W_next.shape[1]+n_new_neurons))), axis=1)

        parameters["W" + str(l + 1)] = W_cur
        parameters["b" + str(l + 1)] = b_cur
        parameters["W" + str(l + 2)] = W_next


def get_param_string(parameters_array):
    param_string = ""
    max_parameters = np.amax(np.abs(parameters_array), 1)
    magnitudes = np.floor(np.log10(max_parameters))
    for m in magnitudes:
        if m > 0:
            m = 0
        param_string += str(int(-m))
    return param_string


def train_dynamic_model(X, y, parameters, learning_rate=0.01, l1_term=0.002, n_iterations=15):
    iteration = 1
    while iteration <= n_iterations:
        parameters = train_model(X, y, parameters, learning_rate=learning_rate, l1_term=l1_term, self_scale=True,
                                 self_scale_coef=None, num_epochs=5, print_cost=True)
        print(f"Iteration {iteration}: accuracy {measure_accuracy(parameters, X, y)}")
        prune_neurons(parameters)
        print(f"After pruning: {get_layer_sizes(parameters)}")
        print(get_param_string(parameters['W1']))
        print(get_param_string(parameters['W2']))
        grow_neurons(parameters)
        print(f"After growing: {get_layer_sizes(parameters)}")
        print(get_param_string(parameters['W1']))
        print(get_param_string(parameters['W2']))
        iteration += 1
        print("-------------------")
    return parameters


def get_layer_sizes(parameters):
    layer_sizes = list()
    L = len(parameters) // 2

    layer_sizes.append(parameters['W1'].shape[1])
    for l in range(L):
        layer_sizes.append(parameters["W" + str(l + 1)].shape[0])

    return layer_sizes
