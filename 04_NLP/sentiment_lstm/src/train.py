"""
train.py
Train the LSTM model on the sentiment dataset.
"""

import os
import tensorflow as tf

from data_loader import load_sentiment_data
from text_preprocessor import prepare_data, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH
from model_builder import build_lstm_model


def train_lstm(
    batch_size: int = 4,
    epochs: int = 10,
    model_dir: str = "models",
    reports_dir: str = "reports",
):
    # 1. Load raw data
    texts, labels = load_sentiment_data()

    # 2. Preprocess data (tokenize, pad, encode, split)
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(
        texts,
        labels,
        num_words=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        test_size=0.2,
        random_state=42,
    )

    # 3. Build model
    model = build_lstm_model(
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=64,
        lstm_units=64,
        max_len=MAX_SEQUENCE_LENGTH,
    )

    # 4. Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 5. Callbacks
    best_model_path = os.path.join(model_dir, "lstm_sentiment_best.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
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
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
    )

    # 7. Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # 8. Save final model
    final_model_path = os.path.join(model_dir, "lstm_sentiment_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 9. Save training log
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
    train_lstm()
