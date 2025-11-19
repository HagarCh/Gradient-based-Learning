import random
import numpy as np
import mlp1
from utils import read_data, text_to_bigrams, get_data_path, build_feature_vocab, build_label_vocab, text_to_unigrams

random.seed(42)
np.random.seed(42)

STUDENTS = [
    {"name": "Hagar Chen Cohen", "ID": "204121461"},
    {"name": "Reut Meiri", "ID": "313191355"},
]

def feats_to_vec(features, F2I):
    feat_array = np.zeros(len(F2I))
    for feat in features:
        if feat in F2I:
            feat_array[F2I[feat]] += 1
    return feat_array


def accuracy_on_dataset(dataset, params, F2I, L2I):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        features = feats_to_vec(features, F2I)
        y_true = L2I[label]
        y_pred = mlp1.predict(features, params)
        if y_true == y_pred:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, F2I, L2I):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for i in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features, F2I)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients and the learning rate.
            for j in range(len(params)):
                params[j] -= learning_rate * grads[j]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, F2I, L2I)
        dev_accuracy = accuracy_on_dataset(dev_data, params, F2I, L2I)
        print(i, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == "__main__":
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # Generalized paths
    train_path = get_data_path('train')
    dev_path = get_data_path('dev')

    TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data(train_path)]
    DEV = [(l, text_to_bigrams(t)) for l, t in read_data(dev_path)]
    # unigrams:
    # TRAIN = [(l, text_to_unigrams(t)) for l, t in read_data(train_path)]
    # DEV = [(l, text_to_unigrams(t)) for l, t in read_data(dev_path)]

    F2I = build_feature_vocab(TRAIN, max_features=600)
    L2I = build_label_vocab(TRAIN)

    in_dim = len(F2I)
    out_dim = len(L2I)
    hid_dim = 100
    num_iterations = 40
    learning_rate = 0.06

    params = mlp1.create_classifier(in_dim, hid_dim , out_dim)
    trained_params = train_classifier(
        TRAIN, DEV, num_iterations, learning_rate, params, F2I, L2I
    )
    #np.savez('model_params_mlp1.npz', W=trained_params[0], b=trained_params[1],  U=trained_params[2],
    #         b_tag=trained_params[3])