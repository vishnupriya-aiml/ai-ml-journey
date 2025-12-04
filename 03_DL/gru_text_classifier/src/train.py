"""
train.py — Train GRU sentiment classifier
"""

import os
import tensorflow as tf

from data_loader import load_gru_dataset
from tokenizer_builder import prepare_gru_data
from model_builder import build_gru_classifier


def train_gru_classifier(
    batch_size=2,
    epochs=15,
    model_dir="models",
    reports_dir="reports"
):
    # load dataset
    texts, labels = load_gru_dataset()

    # encode
    X_train, X_test, y_train, y_test, tokenizer = prepare_gru_data(texts, labels)

    # build model
    model = build_gru_classifier()

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # callbacks
    best_model_path = os.path.join(model_dir, "gru_sentiment_best.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    # train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # save final model
    final_model_path = os.path.join(model_dir, "gru_sentiment_final.keras")
    model.save(final_model_path)

    # save log
    log_path = os.path.join(reports_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Epochs: {epochs}\nBatch size: {batch_size}\n")
        f.write("Training complete.\n")

    print("Training finished.")


if __name__ == "__main__":
    train_gru_classifier()
