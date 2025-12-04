"""
train.py — Train multi-class CNN image classifier on CIFAR-10 subset.
"""

import os
import tensorflow as tf

from data_loader import load_cifar_subset
from model_builder import build_cnn_classifier


def train_cnn_multiclass(
    batch_size: int = 64,
    epochs: int = 20,
    model_dir: str = "models",
    reports_dir: str = "reports",
):
    # 1. Load data
    X_train, X_test, y_train, y_test, class_names = load_cifar_subset()

    # 2. Build model
    model = build_cnn_classifier(
        input_shape=X_train.shape[1:],
        num_classes=len(class_names),
    )

    # 3. Ensure dirs
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 4. Callbacks
    best_model_path = os.path.join(model_dir, "cnn_multiclass_best.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    # 5. Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
    )

    # 6. Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # 7. Save final model
    final_model_path = os.path.join(model_dir, "cnn_multiclass_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 8. Save simple training log
    log_path = os.path.join(reports_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Final test accuracy: {test_acc * 100:.2f}%\n")
        f.write("History keys:\n")
        for key in history.history.keys():
            f.write(f"- {key}\n")

    print(f"Training log written to: {log_path}")


if __name__ == "__main__":
    train_cnn_multiclass()
