import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model
from keras import utils
from training_visualization import weights_visualization_callback
from training_visualization import weights_history_visualizer


seed_val = 1029
utils.set_random_seed(seed_val)


def get_reconstruction_error(X_data, X_reconstructed):
    delta = X_data - X_reconstructed
    delta_sq = np.square(delta)
    mean_delta_sq = np.mean(delta_sq)
    return mean_delta_sq


if __name__ == "__main__":
    X_file = sys.argv[1]
    X = np.load(X_file)

    training_N = int(X.shape[0])  # int(33000)
    X_data = tf.convert_to_tensor(X[:training_N, :])
    input_dim = X_data.shape[1]
    encoding_dim = 16
    N = X_data.shape[0]

    X_test = tf.convert_to_tensor(X[training_N:, :])

    # Define the input layer
    input_layer = Input(shape=(input_dim,))

    # Define the encoder layers
    encoded = Dense(encoding_dim * 4, activation="relu")(input_layer)
    encoded = Dense(encoding_dim * 2, activation="relu")(encoded)
    encoded = Dense(encoding_dim, activation="relu")(encoded)

    # Define the decoder layers
    decoded = Dense(encoding_dim * 2, activation="relu")(encoded)
    decoded = Dense(encoding_dim * 4, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    # callback for visualization
    num_of_layers = 6
    callback = weights_visualization_callback(num_of_layers)
    # Define the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Define the encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                        loss="mse")
    autoencoder.build((N, input_dim))
    autoencoder.summary()

    # Train the autoencoder model

    history = autoencoder.fit(
        X_data,
        X_data,
        epochs=500,
        batch_size=64,
        shuffle=False,
        # validation_data=(X_test, X_test),
        callbacks=callback
    )

    plt.plot(history.history['loss'], label='cross entropy')
    plt.show()
    # Use the encoder to compress the data
    compressed_data = encoder.predict(X_data)
    np.save("compressed.npy", compressed_data)

    # Use the autoencoder to reconstruct the data
    reconstructed_data = autoencoder.predict(X_data)
    reconstruction_error = get_reconstruction_error(X_data, reconstructed_data)
    print(f"reconstruction error: {reconstruction_error}")
    weights_history = callback.get_weights()
    weights_visualizer = weights_history_visualizer(weights_history, mode='2d')
    weights_visualizer.visualize(interval=10)
