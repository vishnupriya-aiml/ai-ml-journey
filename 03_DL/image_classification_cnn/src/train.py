"""
train.py
Train the CNN model on the CIFAR-10 dataset.
"""

import os
import tensorflow as tf
from data_loader import load_cifar10
from model_builder import build_cnn_model


def train_cnn(
    batch_size: int = 64,
    epochs: int = 5,
    model_dir: str = "models",
    reports_dir: str = "reports",
):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Build model
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=10)

    # Ensure output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Callbacks: save best model by validation accuracy
    checkpoint_path = os.path.join(model_dir, "cnn_cifar10_best.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # Optional: early stopping to avoid overfitting
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Save final model as well
    final_model_path = os.path.join(model_dir, "cnn_cifar10_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save simple training log
    log_path = os.path.join(reports_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Final test accuracy: {test_acc * 100:.2f}%\n")
        f.write("\nHistory keys:\n")
        for key in history.history.keys():
            f.write(f"- {key}\n")

    print(f"Training log written to: {log_path}")


if __name__ == "__main__":
    # You can adjust epochs if needed
    train_cnn(epochs=5)
