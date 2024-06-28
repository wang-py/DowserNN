import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# Generate training and validation data
X = np.load("test_data/X.npy")
input_dim = X.shape[1]
N = X.shape[0]
y = np.load("test_data/y.npy")

# Create a neural network model
model = Sequential()
model.add(Dense(80, input_shape=(N, input_dim),
                activation='relu', kernel_initializer='he_normal'))
model.add(Dense(2, input_shape=(80, 80), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy')

model.summary()
# Train the model
history = model.fit(X, y, epochs=800, batch_size=32)

# Generate new input data for prediction

# Plot the results
# plt.figure(figsize=(12, 6))

# Plot training loss
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# 
# plt.show()
