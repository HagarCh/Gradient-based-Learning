# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import os
from collections import Counter


def read_data(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]

def text_to_unigrams(text):
    return list(text)


def get_data_path(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, '..', 'data', folder_name))
    return data_dir

def build_feature_vocab(dataset, max_features=600):
    fc = Counter()
    for _, feats in dataset:
        fc.update(feats)
    # 600 most common bigrams in the training set.
    vocab = [x for x, _ in fc.most_common(max_features)]
    return {f: i for i, f in enumerate(sorted(vocab))}


def build_label_vocab(dataset):
    labels = sorted(set([label for label, _ in dataset]))
    return {label: i for i, label in enumerate(labels)}
