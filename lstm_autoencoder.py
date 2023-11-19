'''
This code contatins the model architecuture for the LSTM encoder-decoder with attention model for time series forecasting.
The model architecture includes LSTM layers, GLU layers, Multi-Head Attention, Layer Normalization, and Dropout layers.
Contains the preprocessing, implementation, and evaluation of the model.
 '''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, LSTM, LayerNormalization, MultiHeadAttention, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

# Import the velocity component from the CFD data
from ActualData import u

data = u
#Specify the wall-normal locations
y_loc = [40,45,50,55,60]
data = data[y_loc]


data_min = np.min(data)
data_max = np.max(data)

# Manual normalization
data_normalized = (data - data_min) / (data_max - data_min)

train_ratio = 0.8
train_size = int(data_normalized.shape[1] * train_ratio)

train_data = data_normalized[:, :train_size]
val_data = data_normalized[:, train_size:]

# Further preprocessing
def create_dataset(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i + timesteps])
    return np.array(X), np.array(y)

timesteps = 48
train_x, train_y = create_dataset(train_data.T, timesteps)
val_x, val_y = create_dataset(val_data.T, timesteps)

train_y = train_y.reshape(train_y.shape[0], 1, train_y.shape[1])
val_y = val_y.reshape(val_y.shape[0], 1, val_y.shape[1])

# Construct the Gated Linear Unit (GLU) layer
class GLU(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, axis=-1):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.axis = axis

    def build(self, input_shape):
        self.dense1 = Dense(self.output_dim, activation='linear')
        self.gate = Dense(self.output_dim, activation='sigmoid')
        super(GLU, self).build(input_shape)

    def call(self, inputs):
        x, gate_inputs = inputs
        return self.dense1(x) * self.gate(gate_inputs)

# Construct the simplified TFT Model
class TFT(Model):
    def __init__(self, hidden_size, lstm_layers, attention_heads, dropout_rate, input_dim):
        super(TFT, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim

        self.lstm_encoder = LSTM(hidden_size, return_sequences=True, return_state=True, input_shape=(None, input_dim))
        self.lstm_decoder = LSTM(hidden_size, return_sequences=True, input_shape=(None, input_dim))


        self.glu_encoder = GLU(input_dim=hidden_size, output_dim=hidden_size)
        self.glu_decoder = GLU(input_dim=hidden_size, output_dim=hidden_size)

        self.attention = MultiHeadAttention(num_heads=attention_heads, key_dim=hidden_size)

        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(hidden_size, activation="elu")
        self.dense2 = Dense(hidden_size, activation="elu")
        self.dense3 = Dense(int(len(y_loc)), activation="linear") # adjust the number of output neurons

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.concat = Concatenate(axis=-1)

    def call(self, inputs):
        x_encoder, x_decoder = inputs

        x, state_h, state_c = self.lstm_encoder(x_encoder)
        x = self.glu_encoder([x, x_encoder])
        x = self.dropout(x)

        x_decoder = self.lstm_decoder(x_decoder, initial_state=[state_h, state_c])
        x_decoder = self.glu_decoder([x_decoder, x_decoder])
        x = self.concat([x, x_decoder])

        attn_output = self.attention(x, x)
        x = self.layer_norm1(x + attn_output)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.layer_norm3(x)
        x = self.dropout(x)

        output = self.dense3(x)

        return output

class LossMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("Epoch:", epoch + 1, " Loss:", logs.get("loss"))

# Define callbacks
loss_monitor = LossMonitor()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)


# Define the hyperparameters
tft = TFT(hidden_size=64, lstm_layers=2, attention_heads=2, dropout_rate=0.1, input_dim=len(y_loc))
tft.build(input_shape=[(None, None, len(y_loc)), (None, None, len(y_loc))])

tft.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")

print("Model summary:")
tft.summary()

history = tft.fit((train_x, train_x), train_y, epochs=30, batch_size=64, validation_data=((val_x, val_x), val_y), callbacks=[loss_monitor, early_stopping, reduce_lr])

# x twice for since both the encoder and the decoder inputs are the same
predictions = tft.predict((val_x, val_x))


# Inverse_transform the predictions
predictions_unscaled = predictions * (data_max - data_min) + data_min

# Inverse_transform the validation data
val_y_unscaled = val_y * (data_max - data_min) + data_min

predictions = predictions_unscaled[:, -1, :]
val_y_unscaled = val_y_unscaled[:, -1, :]

# RMSE
mse = tf.keras.losses.mean_squared_error(val_y_unscaled, predictions)
rmse = tf.math.sqrt(mse)
print(f"RMSE is: {float(np.mean(rmse)):.7f}")

#MAE
mae = tf.keras.losses.mean_absolute_error(val_y_unscaled, predictions)
print(f"MAE is: {float(np.mean(mae)):.7f}")


#Plot  predictions vs reference
for i, loc in enumerate(y_loc):
    plt.figure(figsize=(12, 6))
    plt.plot(val_y_unscaled[:, i], label='Reference (y=' + str(loc) + ')', alpha=0.7)
    plt.plot(predictions[:, i], label= 'Predicted (y=' + str(loc) + ')', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Plot loss vs epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
