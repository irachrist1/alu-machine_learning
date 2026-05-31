#!/usr/bin/env python3
"""Variational autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims: dimensions of the model input.
        hidden_layers: list of nodes for each encoder hidden layer.
        latent_dims: dimensions of the latent space representation.

    Returns:
        A tuple containing the encoder, decoder, and autoencoder models.
    """
    input_layer = keras.Input(shape=(input_dims,))
    previous_layer = input_layer
    for nodes in hidden_layers:
        previous_layer = keras.layers.Dense(nodes, activation='relu')(
            previous_layer)

    mean_layer = keras.layers.Dense(latent_dims, activation=None)(
        previous_layer)
    log_variance_layer = keras.layers.Dense(latent_dims, activation=None)(
        previous_layer)

    def normal_sample(inputs):
        """Draw samples from a normal distribution."""
        mean, log_stddev = inputs
        std_norm = keras.backend.random_normal(
            shape=(keras.backend.shape(mean_layer)[0], latent_dims),
            mean=0, stddev=1)
        return mean + keras.backend.exp(log_stddev / 2) * std_norm

    sample_layer = keras.layers.Lambda(normal_sample)(
        [mean_layer, log_variance_layer])
    encoder = keras.Model(
        input_layer, [sample_layer, mean_layer, log_variance_layer])

    latent_space = keras.Input(shape=(latent_dims,))
    previous_layer = latent_space
    for nodes in reversed(hidden_layers):
        previous_layer = keras.layers.Dense(nodes, activation='relu')(
            previous_layer)
    decoder_layers = keras.layers.Dense(input_dims, activation='sigmoid')(
        previous_layer)
    decoder = keras.Model(latent_space, decoder_layers)

    def VAE_loss(inputs, outputs):
        """Custom loss function including a KL divergence term."""
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + log_variance_layer - keras.backend.square(mean_layer)
        kl_loss -= keras.backend.exp(log_variance_layer)
        kl_loss = keras.backend.sum(kl_loss, axis=-1) * -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto = keras.Model(input_layer, decoder(encoder(input_layer)[0]))
    auto.compile(optimizer='adam', loss=VAE_loss)

    return encoder, decoder, auto
