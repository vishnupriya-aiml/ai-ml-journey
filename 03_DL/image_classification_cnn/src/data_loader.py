"""
data_loader.py
Utility functions to load and preprocess the CIFAR-10 dataset
using TensorFlow / Keras.

CIFAR-10:
- 60,000 color images (32x32x3)
- 10 classes (airplane, car, bird, etc.)
"""

import tensorflow as tf
from typing import Tuple


def load_cifar10() -> Tuple[tuple, tuple]:
    """
    Load CIFAR-10 dataset from Keras and return:
    (x_train, y_train), (x_test, y_test)

    Images are normalized to [0, 1] as floats.
    Labels are integers from 0 to 9.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Convert to float32 and normalize pixel values (0–255) -> (0–1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # y_train and y_test are shape (N, 1); flatten to shape (N,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test)


def get_class_names():
    """
    CIFAR-10 has 10 classes, usually in this order.
    """
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


if __name__ == "__main__":
    # Small self-test to verify shapes
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print("Train images shape:", x_train.shape)
    print("Train labels shape:", y_train.shape)
    print("Test images shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)
    print("Example class names:", get_class_names())
