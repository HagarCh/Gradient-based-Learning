import numpy as np
np.random.seed(42)

STUDENTS = [
    {"name": "Hagar Chen Cohen", "ID": "204121461"},
    {"name": "Reut Meiri", "ID": "313191355"},
]

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
    W, b, _, _= params
    h = np.tanh(np.dot(W, x) + b)
    return h


def classifier_output(x, params):
    _, _, U, b_tag = params
    h = hidden_layer(x, params)
    probs = softmax(np.dot(U, h) + b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    #  Forward prop:
    probs = classifier_output(x, params)
    loss = -np.log(probs[y])
    #  Back prop:
    # l = softmax(o), o = U*h+b_tag,  h = tanh(z), z = Wx+b
    h =  np.tanh(np.dot(W, x) + b)
    dl_do = probs.copy()
    dl_do[y] =dl_do[y] - 1 # dl/do = a - y

    dl_dU = np.outer(dl_do, h)
    dl_db_tag = dl_do

    dl_dh = np.dot(dl_do, U)
    dl_dz = dl_dh * (1 - h ** 2)  #dl/dh * dh/dz

    dl_dW = np.outer(dl_dz, x)
    dl_db = dl_dz.copy()

    return loss, [dl_dW, dl_db, dl_dU, dl_db_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(hid_dim, in_dim) * 0.5
    b =np.zeros(hid_dim)
    U = np.random.randn(out_dim, hid_dim) * 0.5
    b_tag = np.zeros(out_dim)

    params = [W, b, U, b_tag]
    return params


