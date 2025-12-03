"""
train.py
Train the Transformer-style sentiment classifier.
"""

import os
import tensorflow as tf

from data_loader import load_bert_dataset
from tokenizer_builder import prepare_bert_data, MAX_LENGTH
from model_builder import build_transformer_classifier


def train_transformer(
    batch_size: int = 2,
    epochs: int = 15,
    model_dir: str = "models",
    reports_dir: str = "reports",
):
    # 1. Load raw texts + labels
    texts, labels = load_bert_dataset()

    # 2. Encode with BERT tokenizer + split into train/test
    X_train, X_test, y_train, y_test, tokenizer = prepare_bert_data(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        max_length=MAX_LENGTH,
    )

    # 3. Build Transformer classifier (pure Keras)
    model = build_transformer_classifier(max_len=MAX_LENGTH)

    # 4. Ensure output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 5. Callbacks: save best model by validation accuracy + early stopping
    best_model_path = os.path.join(model_dir, "transformer_sentiment_best.keras")
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
        X_train["input_ids"],
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1,
    )

    # 7. Evaluate on test set
    test_loss, test_acc = model.evaluate(
        X_test["input_ids"],
        y_test,
        verbose=1,
    )
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # 8. Save final model
    final_model_path = os.path.join(model_dir, "transformer_sentiment_final.keras")
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
    train_transformer()
