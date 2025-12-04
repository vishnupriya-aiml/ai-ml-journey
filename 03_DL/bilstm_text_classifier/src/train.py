"""
train.py — Train BiLSTM sentiment classifier
"""

import os
import tensorflow as tf

from data_loader import load_bilstm_dataset
from tokenizer_builder import prepare_bilstm_data
from model_builder import build_bilstm_classifier


def train_bilstm_classifier(
    batch_size: int = 2,
    epochs: int = 15,
    model_dir: str = "models",
    reports_dir: str = "reports",
):
    # 1. Load dataset
    texts, labels = load_bilstm_dataset()

    # 2. Tokenize + pad + split
    X_train, X_test, y_train, y_test, tokenizer = prepare_bilstm_data(
        texts, labels
    )

    # 3. Build model
    model = build_bilstm_classifier()

    # 4. Ensure dirs
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 5. Callbacks
    best_model_path = os.path.join(model_dir, "bilstm_sentiment_best.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # 6. Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
    )

    # 7. Save final model
    final_model_path = os.path.join(model_dir, "bilstm_sentiment_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 8. Save simple training log
    log_path = os.path.join(reports_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("Training completed for BiLSTM model.\n")
        f.write("History keys:\n")
        for key in history.history.keys():
            f.write(f"- {key}\n")

    print(f"Training log written to: {log_path}")


if __name__ == "__main__":
    train_bilstm_classifier()
