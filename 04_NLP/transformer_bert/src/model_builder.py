"""
model_builder.py
Defines a BERT-based sentiment classifier using TensorFlow / Keras
and HuggingFace Transformers.
"""

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoConfig

from tokenizer_builder import MODEL_NAME


def build_bert_classifier(
    model_name: str = MODEL_NAME,
    num_labels: int = 2,
    learning_rate: float = 2e-5,
) -> tf.keras.Model:
    """
    Build and compile a BERT-based sequence classification model.

    Uses:
    - Pretrained BERT base model (bert-base-uncased by default)
    - Classification head for binary sentiment (positive/negative)
    """

    # Load configuration with correct number of labels
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Load pre-trained BERT model with classification head
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    # Use the built-in loss from the model (handles logits + labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # model.compute_loss will calculate appropriate loss when labels are passed
    model.compile(
        optimizer=optimizer,
        loss=model.compute_loss,
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Small self-test: build model and print summary
    bert_model = build_bert_classifier()
    bert_model.summary()
