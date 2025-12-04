"""
data_loader.py — load and filter CIFAR-10 for multi-class CNN

We will use 3 classes from CIFAR-10:
- cat (label 3)   -> class 0
- dog (label 5)   -> class 1
- frog (label 6)  -> class 2
"""

import numpy as np
from tensorflow.keras.datasets import cifar10

# Original CIFAR-10 label mapping for reference:
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

ORIG_CLASSES = [3, 5, 6]  # cat, dog, frog
CLASS_NAME_MAP = {0: "cat", 1: "dog", 2: "frog"}


def _filter_classes(X, y, orig_classes=ORIG_CLASSES):
    """
    Keep only the selected original classes and remap labels to [0, 1, 2].
    """
    mask = np.isin(y, orig_classes)
    X_filtered = X[mask]
    y_filtered_orig = y[mask]

    # Map original labels (3,5,6) -> (0,1,2)
    mapping = {orig_label: idx for idx, orig_label in enumerate(orig_classes)}
    y_mapped = np.vectorize(mapping.get)(y_filtered_orig)

    return X_filtered, y_mapped


def load_cifar_subset():
    """
    Load CIFAR-10, filter to 3 classes, normalize to [0,1].
    Returns:
        X_train, X_test: float32 arrays (N, 32, 32, 3)
        y_train, y_test: int arrays with labels in {0,1,2}
        class_names: list of class names in order
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # CIFAR labels are shape (N,1); flatten to (N,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    X_train, y_train = _filter_classes(X_train, y_train)
    X_test, y_test = _filter_classes(X_test, y_test)

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    class_names = [CLASS_NAME_MAP[i] for i in range(len(ORIG_CLASSES))]

    return X_train, X_test, y_train, y_test, class_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_names = load_cifar_subset()
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    print("Classes:", class_names)
