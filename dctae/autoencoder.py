"""Module 5: Autoencoder — improved dense + convolutional autoencoder for Non-ROI blocks.

Loss function:
    L = L_reconstruction + λ · L_sparsity
    L = (1/N) Σ ||x - x̂||² + λ · Σ |a_j|

The sparsity term (L1 regularization on latent activations) forces most latent
units toward zero, producing a more compressible representation.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf


def build_dense_autoencoder(
    input_dim: int,
    latent_dim: int = 32,
    sparsity_weight: float = 1e-4,
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build an improved deep sparse autoencoder for non-ROI blocks.

    Architecture (deeper than original for better reconstruction):
        Encoder: input_dim → 192 → 128 → 64 → latent_dim
        Decoder: latent_dim → 64 → 128 → 192 → input_dim

    Each hidden layer uses BatchNormalization for training stability
    and Dropout for regularization.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="input_block")

    x = tf.keras.layers.Dense(192, name="enc_dense1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="enc_bn1")(x)
    x = tf.keras.layers.ReLU(name="enc_relu1")(x)
    x = tf.keras.layers.Dropout(0.1, name="enc_drop1")(x)

    x = tf.keras.layers.Dense(128, name="enc_dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="enc_bn2")(x)
    x = tf.keras.layers.ReLU(name="enc_relu2")(x)

    x = tf.keras.layers.Dense(64, name="enc_dense3")(x)
    x = tf.keras.layers.BatchNormalization(name="enc_bn3")(x)
    x = tf.keras.layers.ReLU(name="enc_relu3")(x)

    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="sigmoid",
        activity_regularizer=tf.keras.regularizers.L1(sparsity_weight),
        name="latent_code",
    )(x)

    x = tf.keras.layers.Dense(64, name="dec_dense1")(latent)
    x = tf.keras.layers.BatchNormalization(name="dec_bn1")(x)
    x = tf.keras.layers.ReLU(name="dec_relu1")(x)

    x = tf.keras.layers.Dense(128, name="dec_dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="dec_bn2")(x)
    x = tf.keras.layers.ReLU(name="dec_relu2")(x)

    x = tf.keras.layers.Dense(192, name="dec_dense3")(x)
    x = tf.keras.layers.BatchNormalization(name="dec_bn3")(x)
    x = tf.keras.layers.ReLU(name="dec_relu3")(x)
    x = tf.keras.layers.Dropout(0.1, name="dec_drop3")(x)

    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inputs, outputs, name="dense_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="dense_encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="decoder_input")
    dec = latent_inputs
    for layer_name in ["dec_dense1", "dec_bn1", "dec_relu1",
                       "dec_dense2", "dec_bn2", "dec_relu2",
                       "dec_dense3", "dec_bn3", "dec_relu3", "dec_drop3",
                       "reconstruction"]:
        dec = autoencoder.get_layer(layer_name)(dec)
    decoder = tf.keras.Model(latent_inputs, dec, name="dense_decoder")

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder, decoder


def build_conv_autoencoder(
    block_size: int = 16,
    latent_dim: int = 32,
    sparsity_weight: float = 1e-4,
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build a convolutional autoencoder for non-ROI blocks.

    Architecture:
        Encoder: (block_size, block_size, 1) → Conv(32) → Conv(64) → Flatten → latent_dim
        Decoder: latent_dim → Dense → Reshape → ConvTranspose(64) → ConvTranspose(32) → (block_size, block_size, 1)

    Convolutional layers better capture spatial patterns in image blocks.
    """
    spatial_dim = block_size // 4

    inputs = tf.keras.Input(shape=(block_size, block_size, 1), name="input_block")

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", name="enc_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="enc_bn1")(x)
    x = tf.keras.layers.ReLU(name="enc_relu1")(x)

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", name="enc_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="enc_bn2")(x)
    x = tf.keras.layers.ReLU(name="enc_relu2")(x)

    x = tf.keras.layers.Flatten(name="enc_flatten")(x)
    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="sigmoid",
        activity_regularizer=tf.keras.regularizers.L1(sparsity_weight),
        name="latent_code",
    )(x)

    x = tf.keras.layers.Dense(spatial_dim * spatial_dim * 64, name="dec_dense")(latent)
    x = tf.keras.layers.BatchNormalization(name="dec_bn1")(x)
    x = tf.keras.layers.ReLU(name="dec_relu1")(x)
    x = tf.keras.layers.Reshape((spatial_dim, spatial_dim, 64), name="dec_reshape")(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", name="dec_deconv1")(x)
    x = tf.keras.layers.BatchNormalization(name="dec_bn2")(x)
    x = tf.keras.layers.ReLU(name="dec_relu2")(x)

    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", name="dec_deconv2")(x)
    x = tf.keras.layers.BatchNormalization(name="dec_bn3")(x)
    x = tf.keras.layers.ReLU(name="dec_relu3")(x)

    outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inputs, outputs, name="conv_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="conv_encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="decoder_input")
    dec = latent_inputs
    for layer_name in ["dec_dense", "dec_bn1", "dec_relu1", "dec_reshape",
                       "dec_deconv1", "dec_bn2", "dec_relu2",
                       "dec_deconv2", "dec_bn3", "dec_relu3",
                       "reconstruction"]:
        dec = autoencoder.get_layer(layer_name)(dec)
    decoder = tf.keras.Model(latent_inputs, dec, name="conv_decoder")

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder, decoder


def train_autoencoder(
    autoencoder: tf.keras.Model,
    train_data: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.1,
) -> tf.keras.callbacks.History:
    """Train autoencoder with early stopping to prevent overfitting."""
    effective_batch = min(batch_size, max(1, len(train_data)))

    callbacks = []
    if len(train_data) > 10:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if len(train_data) > 20 else "loss",
                patience=10,
                restore_best_weights=True,
                min_delta=1e-5,
            )
        )

    use_validation = len(train_data) > 20
    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=effective_batch,
        verbose=0,
        shuffle=True,
        validation_split=validation_split if use_validation else 0.0,
        callbacks=callbacks,
    )
    return history


def compress_blocks(
    encoder: tf.keras.Model,
    blocks: np.ndarray,
    block_size: int,
    model_type: str = "dense",
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode blocks to latent space and quantize to uint8.

    Returns:
        quantized_latent: uint8 quantized latent codes (compressed representation)
        latent_codes: float32 raw latent codes
    """
    if model_type == "dense":
        vectors = blocks.astype(np.float32).reshape(len(blocks), -1) / 255.0
    else:
        vectors = blocks.astype(np.float32).reshape(-1, block_size, block_size, 1) / 255.0

    latent_codes = encoder.predict(vectors, verbose=0)
    quantized = np.round(latent_codes * 255.0).astype(np.uint8)
    return quantized, latent_codes


def decompress_blocks(
    decoder: tf.keras.Model,
    quantized_latent: np.ndarray,
    block_size: int,
    model_type: str = "dense",
) -> np.ndarray:
    """Dequantize latent codes and decode back to image blocks."""
    dequantized = quantized_latent.astype(np.float32) / 255.0
    reconstructed = decoder.predict(dequantized, verbose=0)

    if model_type == "dense":
        return reconstructed.reshape(-1, block_size, block_size) * 255.0
    else:
        return reconstructed.reshape(-1, block_size, block_size) * 255.0


def prepare_training_data(
    blocks: np.ndarray,
    block_size: int,
    model_type: str = "dense",
) -> np.ndarray:
    """Normalize block data for autoencoder training."""
    if model_type == "dense":
        return blocks.astype(np.float32).reshape(len(blocks), -1) / 255.0
    else:
        return blocks.astype(np.float32).reshape(-1, block_size, block_size, 1) / 255.0
