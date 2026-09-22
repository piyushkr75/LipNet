"""LipNet model construction and checkpoint loading."""

import os
from pathlib import Path

# The supplied checkpoint was saved with Keras 2. This must be set before
# TensorFlow is imported so it uses the tf_keras compatibility package.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoint"


def load_model() -> tf.keras.Sequential:
    """Build LipNet and load the bundled legacy TensorFlow checkpoint."""
    if not CHECKPOINT_PATH.with_suffix(".index").is_file():
        raise FileNotFoundError(f"LipNet weights are missing: {CHECKPOINT_PATH.with_suffix('.index')}")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(75, 46, 140, 1)),
        tf.keras.layers.Conv3D(128, 3, padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.Conv3D(256, 3, padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.Conv3D(75, 3, padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool3D((1, 2, 2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(41, kernel_initializer="he_normal", activation="softmax"),
    ])
    model.load_weights(str(CHECKPOINT_PATH))
    return model
