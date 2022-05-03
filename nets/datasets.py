import os
import tensorflow as tf
import numpy as np
import pandas as pd
import imageio

from models import dtype


def get_dataset_sample(X, y, fraction, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set random seed
    selection = np.random.choice([True, False], len(X), p=[fraction, 1 - fraction])
    if seed is not None:
        np.random.seed()  # Unset random seed
    X_sampled = X[selection]
    y_sampled = y[selection]
    return X_sampled, y_sampled


class Dataset:
    def __init__(self, X_train, y_train, X_test, y_test, shape, shape_flattened, fraction, vision=True,
                 standardize=True):
        if fraction is not None:
            X_train, y_train = get_dataset_sample(X_train, y_train, fraction, seed=42)
            X_test, y_test = get_dataset_sample(X_test, y_test, fraction, seed=42)

        X_train = X_train.astype(dtype)
        y_train = y_train.astype(dtype)
        X_test = X_test.astype(dtype)
        y_test = y_test.astype(dtype)

        if vision:
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        X_train = np.reshape(X_train, shape_flattened)
        X_test = np.reshape(X_test, shape_flattened)

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        if standardize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(X_train)  # Scaling each feature independently

            X_norm = scaler.transform(X)
            del X
            X_train_norm = scaler.transform(X_train)
            del X_train
            X_test_norm = scaler.transform(X_test)
            del X_test
        else:
            X_norm = X
            X_train_norm = X_train
            X_test_norm = X_test

        X_norm = np.reshape(X_norm, shape)
        X_train_norm = np.reshape(X_train_norm, shape)
        X_test_norm = np.reshape(X_test_norm, shape)

        # Shuffle X_norm and y
        assert len(X_norm) == len(y)
        p = np.random.permutation(len(X_norm))
        X_norm, y = X_norm[p], y[p]

        self.X_norm = X_norm
        self.y = y
        self.X_train_norm = X_train_norm
        self.y_train = y_train
        self.X_test_norm = X_test_norm
        self.y_test = y_test


def get_cifar_10_dataset(fraction=None):
    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    shape = (-1, 32, 32, 3)
    shape_flattened = (-1, 3072)  # Scaling each feature independently
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_cifar_100_dataset(fraction=None):
    cifar100 = tf.keras.datasets.cifar100
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    shape = (-1, 32, 32, 3)
    shape_flattened = (-1, 3072)  # Scaling each feature independently
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_svhn_dataset(fraction=None):
    from urllib.request import urlretrieve
    from scipy import io

    train_filename, _ = urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
    test_filename, _ = urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')

    X_train = io.loadmat(train_filename, variable_names='X').get('X')
    y_train = io.loadmat(train_filename, variable_names='y').get('y')
    X_test = io.loadmat(test_filename, variable_names='X').get('X')
    y_test = io.loadmat(test_filename, variable_names='y').get('y')

    X_train = np.moveaxis(X_train, -1, 0)
    y_train -= 1
    X_test = np.moveaxis(X_test, -1, 0)
    y_test -= 1

    shape = (-1, 32, 32, 3)
    shape_flattened = (-1, 3072)  # Scaling each feature independently
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_tiny_imagenet_dataset(fraction=None):
    """
    Original source: https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb
    Original author: sonugiri1043@gmail.com
    """

    if not os.path.isdir('IMagenet'):
        os.system('git clone https://github.com/seshuad/IMagenet')

    print("Processing the downloaded dataset...")

    path = 'IMagenet/tiny-imagenet-200/'

    id_dict = {}
    for i, line in enumerate(open(path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    train_data = list()
    test_data = list()
    train_labels = list()
    test_labels = list()

    for key, value in id_dict.items():
        train_data += [imageio.imread(path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i
                       in range(500)]
        train_labels_ = np.array([[0] * 200] * 500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    X_train = np.array(train_data)
    X_test = np.array(test_data)
    del train_data, train_labels_

    for line in open(path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(imageio.imread(path + 'val/images/{}'.format(img_name), pilmode='RGB'))
        test_labels_ = np.array([[0] * 200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    y_train = np.argmax(np.array(train_labels), axis=1)
    y_test = np.argmax(np.array(test_labels), axis=1)
    del train_labels
    del test_data, test_labels_, test_labels

    shape = (-1, 64, 64, 3)
    shape_flattened = (-1, 12288)  # Scaling each feature independently
    print("Calling Dataset()")
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_mnist_dataset(fraction=None):
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    shape = (-1, 28, 28, 1)
    shape_flattened = (-1, 1)  # Scaling all features together
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_fashion_mnist_dataset(fraction=None):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    shape = (-1, 28, 28, 1)
    shape_flattened = (-1, 1)  # Scaling all features together
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, fraction=fraction)


def get_fifteen_puzzle_dataset(path=None, fraction=None):
    from sklearn.model_selection import train_test_split

    if path is None:
        from google.colab import drive
        drive.mount('/content/gdrive')
        path = 'gdrive/MyDrive/15-costs-v3.csv'
    costs = pd.read_csv(path)
    costs = costs.sample(frac=fraction, random_state=42)

    X_raw = costs.iloc[:, :-1].values
    y = costs['cost'].values
    X = np.apply_along_axis(lambda x: np.eye(16)[x].ravel(), 1, X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, X_raw, y

    shape = (-1, 256)
    shape_flattened = (-1, 256)  # Scaling all features together
    return Dataset(X_train, y_train, X_test, y_test, shape=shape, shape_flattened=shape_flattened, vision=False,
                   fraction=None)
