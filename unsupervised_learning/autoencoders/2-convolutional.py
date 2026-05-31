#!/usr/bin/env python3
"""Convolutional autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.

    Args:
        input_dims: tuple of input dimensions (height, width, channels).
        filters: list of filter counts for each encoder conv layer.
        latent_dims: tuple of latent space dimensions.

    Returns:
        A tuple containing the encoder, decoder, and autoencoder models.
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for i, f in enumerate(reversed(filters)):
        padding = 'valid' if i == len(filters) - 1 else 'same'
        x = keras.layers.Conv2D(
            f, (3, 3), padding=padding, activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_output = keras.layers.Conv2D(
        input_dims[2], (3, 3), padding='same', activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=input_dims)
    auto = keras.Model(auto_input, decoder(encoder(auto_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
