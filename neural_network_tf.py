import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate training and validation data
    X = np.load("test_data/X.npy")
    input_dim = X.shape[1]
    N = X.shape[0]
    y = np.load("test_data/y.npy")

    X_data = tf.convert_to_tensor(X)
    y_data = tf.convert_to_tensor(y)

    # Create a neural network model
    model = Sequential()
    model.add(
        Dense(
            80,
            activation="relu",
            kernel_initializer="he_normal",
        )
    )
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy")
    model.build((N, input_dim))

    model.summary()
    # Train the model
    history = model.fit(X, y, epochs=80, batch_size=32)
    y_predicted = model.predict(X_data)
    np.set_printoptions(precision=4, suppress=True)
    print("expected output:\n", y)
    print("predicted output:\n", y_predicted)

    # Generate new input data for prediction

    # Plot training loss
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(history.history['loss'], label='cross entropy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    # ax[1].plot(history.history['cross_entropy'], label='cross entropy')
    # ax[1].set_title('cross entropy')
    # ax[1].legend()
    plt.show()
