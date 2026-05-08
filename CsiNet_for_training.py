"""
Rewritten CsiNet training & evaluation script.
Functionality preserved from original CsiNet_for_training.py:
- Build residual autoencoder (channels_first)
- Train with x_train as both input and target
- Save losses, model, decoded outputs
- Compute inference time, NMSE and correlation(rho) using FFT + zero-padding
- Plot first N reconstructed samples
"""

import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback

# ──────────────────────── Configuration (same defaults as original) ─────────────────────── #
tf.keras.backend.clear_session()

envir = 'indoor'              # 'indoor' or 'outdoor'
img_height = 32               # CSI spatial height
img_width = 32                # CSI frequency width (compressed input width)
img_channels = 2              # real & imag
img_total = img_height * img_width * img_channels

residual_num = 2              # number of residual blocks in decoder (same as original)
encoded_dim = 512             # compression dim (512 default -> 1/4 in original context)

epochs = 1000
batch_size = 200

# Generate filename & paths similar to original
file = 'CsiNet_{}_dim{}{}'.format(envir, encoded_dim, time.strftime('_%m_%d'))
path = 'result/TensorBoard_{}'.format(file)
os.makedirs('result', exist_ok=True)
os.makedirs(path, exist_ok=True)


# ──────────────────────── Utility Blocks (kept same behavior) ─────────────────────── #
def add_common_layers(y):
    """Add BatchNormalization + LeakyReLU (shared in residual blocks)."""
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block_decoded(y_in):
    """Decoder residual block:
    shortcut + (Conv2D -> BN+LReLU) x2 -> Conv2D -> BN -> add -> LReLU
    Mirrors structure from original fragments.
    """
    shortcut = y_in

    y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y_in)
    y = add_common_layers(y)

    y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = BatchNormalization()(y)

    y = add([shortcut, y])
    y = LeakyReLU()(y)
    return y


# ──────────────────────── Build residual autoencoder (encoder + decoder) ─────────────────────── #
def residual_network(img_channels, img_height, img_width, residual_num, encoded_dim):
    image_tensor = Input(shape=(img_channels, img_height, img_width))  # channels_first

    # Encoder: initial conv + BN+LReLU -> flatten -> Dense(encoded_dim)
    x = Conv2D(2, (3, 3), padding='same', data_format='channels_first')(image_tensor)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)  # flatten
    encoded = Dense(encoded_dim, activation='linear')(x)

    # Decoder: Dense -> reshape -> residual blocks -> final conv(sigmoid)
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,))(x)

    for _ in range(residual_num):
        x = residual_block_decoded(x)

    # Output in [0,1] for real/imag parts (sigmoid)
    output = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format='channels_first')(x)

    autoencoder = Model(inputs=[image_tensor], outputs=[output])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# ──────────────────────── Custom Loss History Callback ─────────────────────── #
class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs=None):
        if logs is None: return
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        self.losses_val.append(logs.get('val_loss'))


# ──────────────────────── Data Loading & Preprocessing ─────────────────────── #
# Load train/val/test depending on environment (match original filenames)
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT']
else:
    mat = sio.loadmat('data/DATA_Htrainout.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT']

# ensure float32
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# reshape into channels_first: [batch, channels, height, width]
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))


# ──────────────────────── Model Construction & Training ─────────────────────── #
autoencoder = residual_network(img_channels, img_height, img_width, residual_num, encoded_dim)
print(autoencoder.summary())

history = LossHistory()

tensorboard_cb = TensorBoard(log_dir=path)
autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, x_val),
    callbacks=[history, tensorboard_cb]
)

# Save loss histories to CSV (same filenames as original pattern)
trainloss_file = 'result/trainloss_%s.csv' % file
valloss_file = 'result/valloss_%s.csv' % file
np.savetxt(trainloss_file, np.array(history.losses_train), delimiter=",")
np.savetxt(valloss_file, np.array(history.losses_val), delimiter=",")


# ──────────────────────── Inference on Test Data ─────────────────────── #
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))


# Save reconstructed CSI (flattened) similar to original
decoded_csv = "result/decoded_%s.csv" % file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(decoded_csv, x_hat1, delimiter=",")


# ──────────────────────── Prepare complex-domain CSI and compute NMSE & Correlation ─────────────────────── #
# Convert input/test & reconstructed from [0,1] to complex domain (-0.5~0.5)
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)   # original complex CSI (time-domain / image-domain)

x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)     # reconstructed complex CSI (image-domain)

# Load original frequency-domain CSI HF_all (for correlation calculation), then FFT reconstructed to frequency domain
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # original frequency-domain CSI array
else:
    mat = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']

# Reshape original frequency-domain CSI to [batch, img_height, 125] (as original)
X_test = np.reshape(X_test, (len(X_test), img_height, 125))

# Reconstruct frequency-domain CSI from reconstructed image-domain (zero-pad then FFT)
# First reshape x_hat_C back to [batch, img_height, img_width]
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
# Zero-pad along frequency axis to length 257 then FFT, then truncate to first 125 bins (same as original)
pad_len = 257 - img_width
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, pad_len))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]  # keep first 125 bins

# reshape for NMSE calculation
X_hat_flat = np.reshape(X_hat, (len(X_hat), -1))
X_test_flat = np.reshape(X_test, (len(X_test), -1))

# NMSE computation (original used power & mse on complex time-domain data too)
power = np.sum(np.abs(x_test_C)**2, axis=1)
mse = np.sum(np.abs(x_test_C - x_hat_C)**2, axis=1)
nmse_db = 10 * np.log10(np.mean(mse / power))

# Correlation rho calculation between original and reconstructed frequency-domain CSI (as original fragments)
n1 = np.sqrt(np.sum(np.conj(X_test_flat) * X_test_flat, axis=1)).astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat_flat) * X_hat_flat, axis=1)).astype('float64')
aa = np.abs(np.sum(np.conj(X_test_flat) * X_hat_flat, axis=1))
rho = np.mean(aa / (n1 * n2), axis=1)  # per-sample correlation -> mean along axis=1 (kept same form)
# Note: original printed average correlation as np.mean(rho) after this step.

# Save rho to CSV (same naming)
rho_file = "result/rho_%s.csv" % file
np.savetxt(rho_file, rho, delimiter=",")

# Print results (matching original printouts)
print("In {} environment".format(envir))
print("When dimension is", encoded_dim)
print("NMSE is ", nmse_db)
print("Correlation is ", np.mean(rho))


# ──────────────────────── Save model architecture & weights ─────────────────────── #
model_json = autoencoder.to_json()
with open("result/model_%s.json" % file, "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("result/model_%s.h5" % file)


# ──────────────────────── Visualization (plot first 10 original & reconstructed absolute-value images) ─────────────────────── #
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # original absolute value
    ax = plt.subplot(2, n, i + 1)
    x_testplo = np.abs(x_test[i, 0, :, :] - 0.5 + 1j * (x_test[i, 1, :, :] - 0.5))
    plt.imshow(np.max(np.max(x_testplo)) - x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

    # reconstructed absolute value
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = np.abs(x_hat[i, 0, :, :] - 0.5 + 1j * (x_hat[i, 1, :, :] - 0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo)) - decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

plt.show()
