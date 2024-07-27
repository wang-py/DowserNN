import sys
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from training_visualization import weights_visualization_callback
from training_visualization import weights_history_visualizer

if __name__ == "__main__":
    # Define the dimensions of the input data
    input_dim = 784  # 28x28 images
    encoding_dim = 32  # bottleneck dimension

    # Define the input layer
    input_layer = Input(shape=(input_dim,))

    # Define the encoder layers
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(encoding_dim, activation="relu")(encoded)

    # Define the decoder layers
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    # callback for visualization
    num_of_layers = 6
    callback = weights_visualization_callback(num_of_layers)
    # Define the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Define the encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder model
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()

    # Train the autoencoder model
    from keras.datasets import mnist

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = x_train.reshape((len(x_train), input_dim))
    x_test = x_test.reshape((len(x_test), input_dim))

    history = autoencoder.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=callback
    )

    plt.plot(history.history['loss'], label='cross entropy')
    plt.show()
    # Use the encoder to compress the data
    compressed_data = encoder.predict(x_test)

    # Use the autoencoder to reconstruct the data
    reconstructed_data = autoencoder.predict(x_test)
    weights_history = callback.get_weights()
    weights_visualizer = weights_history_visualizer(weights_history, mode='2d')
    weights_visualizer.visualize()
