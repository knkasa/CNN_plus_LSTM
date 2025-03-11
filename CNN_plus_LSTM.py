import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, Flatten, concatenate

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 32
sequence_length = 6 
num_features = 5  # e.g., temperature, humidity, pressure, etc.
image_width = 24  # e.g., 24x24 image

# Create random time series data for LSTM
X_time_series = np.random.random((100, sequence_length, num_features))  # 100 samples

# Create random image data for CNN
X_images = np.random.random((100, image_width, image_width, 1))  # 100 samples

# Create random target values
y = np.random.random((100, 1))  # 100 target values

# Build the model
# LSTM branch
time_series_input = Input(shape=(sequence_length, num_features), name='time_series_input')
lstm_out = LSTM(64, return_sequences=False)(time_series_input)

# CNN branch
image_input = Input(shape=(image_width, image_width, 1), name='image_input')
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
flat = Flatten()(conv2)

# Combine both branches
combined = concatenate([lstm_out, flat])

# Output layer
dense1 = Dense(64, activation='relu')(combined)
output = Dense(1, activation='linear')(dense1)

# Create the model with multiple inputs
model = Model(inputs=[time_series_input, image_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    [X_time_series, X_images], y,
    validation_split=0.2,
    epochs=10,
    batch_size=batch_size,
    verbose=1
)

# Make predictions with the model
X_time_series_new = np.random.random((5, sequence_length, num_features))
X_images_new = np.random.random((5, image_width, image_width, 1))
predictions = model.predict([X_time_series_new, X_images_new])
print("Predictions:", predictions)

# Plot training history
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Matplotlib not available for plotting")