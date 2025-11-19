import numpy as np
np.random.seed(42)

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    exps = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    x = exps / np.sum(exps)
    return x

def hidden_layer(x, params):
    W, b = params
    h = np.tanh(np.dot(W, x)+b)
    return h

def classifier_output(x, params):
    h = x
    for i in range(0, len(params) - 2, 2): # Iterates over all hidden layers (a step of 2 since params contains [W, b] pairs)
        params_i = (params[i], params[i+1])
        h = hidden_layer(h, params_i)
    # Final layer
    W, b = params[-2], params[-1]
    probs = softmax(np.dot(W, h) + b)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    in_act = [x]  # List of layers inputs
    h = x

    #  Forward prop:
    for i in range(0, len(params) - 2, 2): # hidden layers
        params_i = (params[i], params[i + 1])
        h = hidden_layer(h, params_i)
        in_act.append(h)

    # Output layer
    W_output, b_output = params[-2], params[-1]
    probs = softmax(np.dot(W_output, in_act[-1]) + b_output)
    loss = -np.log(probs[y])

    #  Back prop:
    grads = [None] * len(params)
    # Final output layer
    dl_do = probs.copy()
    dl_do[y] = dl_do[y] - 1 # dl/do = a - y
    grads[-2] = np.outer(dl_do, in_act[-1])
    grads[-1] = dl_do
    der_prev = np.dot(dl_do, params[-2]) * (1 - in_act[-1] ** 2)

    # Back prop through hidden layers
    for i in reversed(range(0, len(params) - 2, 2)):
        act_prev = in_act[i//2]
        grads[i] = np.outer(der_prev, act_prev)
        grads[i + 1] = der_prev
        if i != 0:
            W = params[i]
            der_prev = np.dot(der_prev, W) * (1 - act_prev ** 2)

    return loss, grads





def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i + 1], dims[i]) * 0.5
        b = np.zeros(dims[i + 1])
        params += [W, b]
    return params
