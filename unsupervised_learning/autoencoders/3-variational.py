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
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sample from the latent distribution via reparameterization."""
        mean, log_var = args
        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    def vae_loss(inputs, outputs):
        """Compute reconstruction loss plus KL divergence."""
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto_input = keras.Input(shape=(input_dims,))
    auto = keras.Model(auto_input, decoder(encoder(auto_input)[0]))
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
