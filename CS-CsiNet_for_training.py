
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback


try:
    tf.reset_default_graph()
except AttributeError:

    pass

# ---------------- Configuration ----------------
envir = 'indoor'  # 'indoor' or 'outdoor'
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

residual_num = 2
encoded_dim = 512  # e.g., 512,128,64,32

file = 'CS-CsiNet_%s_dim%d%s' % (envir, encoded_dim, time.strftime('_%m_%d'))
path = 'result/TensorBoard_%s' % file

# ---------------- Utility / Layers ----------------
def add_common_layers(y):
    """Add BatchNorm + LeakyReLU (與原始片段相同)"""
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y

def residual_block_decoded(y):
    """Residual block for decoder: 3x3 Conv2D stack with shortcut"""
    shortcut = y
    y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    # Residual fusion: shortcut + conv stack output 
    y = BatchNormalization()(y)
    y = add([shortcut, y])
    y = LeakyReLU()(y)
    return y

def residual_network(encoded, residual_num, encoded_dim):
    """Decoder pipeline: Dense -> Reshape -> residual blocks -> final Conv2D(sigmoid)"""
    decoded = Dense(img_total, activation='linear')(encoded)
    decoded = Reshape((img_channels, img_height, img_width))(decoded)
    for i in range(residual_num):
        decoded = residual_block_decoded(decoded)
    decoded = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(decoded)
    return decoded

# ---------------- Custom Callback ----------------
class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses_val.append(logs.get('val_loss'))

# ---------------- Data Loading ----------------
# Load MATLAB-formatted CSI datasets (train/val/test)
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT']
elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htrainout.mat')
    x_train = mat['HT']
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT']
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT']
else:
    raise ValueError("Unknown environment: %s" % envir)

# Reshape to channels_first [batch, channels, height, width]
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

# Convert to float32
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# ---------------- Fixed Random Projection Encoder (A matrix) ----------------
matA = sio.loadmat('data/A%d.mat' % (encoded_dim))
A = matA['A']  # fixed projection matrix
# Flatten input per sample for dot product with A 
x_train_flat = np.reshape(x_train, (len(x_train), -1))
x_val_flat = np.reshape(x_val, (len(x_val), -1))
x_test_flat = np.reshape(x_test, (len(x_test), -1))

y_train = np.dot(x_train_flat, A.T)
y_val = np.dot(x_val_flat, A.T)
y_test = np.dot(x_test_flat, A.T)

# ---------------- Build Decoder Model ----------------
image_tensor = Input(shape=(encoded_dim,))
network_output = residual_network(image_tensor, residual_num, encoded_dim)
decoder = Model(inputs=[image_tensor], outputs=[network_output])
decoder.compile(optimizer='adam', loss='mse')
print(decoder.summary())

# ---------------- Train ----------------
history = LossHistory()
tensorboard_cb = TensorBoard(log_dir=path)
decoder.fit(y_train, x_train,
            epochs=1000,
            batch_size=200,
            shuffle=True,
            validation_data=(y_val, x_val),
            callbacks=[history, tensorboard_cb])

# Save training/validation loss to CSV
trainloss_file = 'result/trainloss_%s.csv' % file
valloss_file = 'result/valloss_%s.csv' % file
np.savetxt(trainloss_file, np.array(history.losses_train), delimiter=",")
np.savetxt(valloss_file, np.array(history.losses_val), delimiter=",")

# Save model architecture & weights
model_json = decoder.to_json()
with open("result/model_%s.json" % file, "w") as json_file:
    json_file.write(model_json)
decoder.save_weights("result/model_%s.h5" % file)

# ---------------- Inference on Test Data ----------------
tStart = time.time()
x_hat = decoder.predict(y_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

# Save reconstructed flattened
np.savetxt("result/decoded_%s.csv" % file, np.reshape(x_hat, (len(x_hat), -1)), delimiter=",")

# ---------------- Convert tensors [0,1] -> complex domain (-0.5~0.5) ----------------
# Original test complex (time-domain) from x_test
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)

# Reconstructed complex (time-domain)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)

# ---------------- NMSE (dB) ----------------
power = np.sum(np.abs(x_test_C) ** 2, axis=1)
mse = np.sum(np.abs(x_test_C - x_hat_C) ** 2, axis=1)
nmse_db = 10 * math.log10(np.mean(mse / power))
print("In %s environment" % envir)
print("When dimension is", encoded_dim)
print("NMSE is ", nmse_db)

# ---------------- Frequency-domain correlation ----------------
# Load original frequency-domain CSI for correlation calc
if envir == 'indoor':
    matf = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = matf['HF_all']  # shape: [N, height, 125] or similar
elif envir == 'outdoor':
    matf = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = matf['HF_all']

# Convert reconstructed time-domain CSI to frequency-domain:
# 1) reshape to [N, height, width]
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
# 2) zero-pad to 257 bins, FFT along frequency axis, then truncate to first 125 bins
pad_len = 257 - img_width
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, pad_len))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

# Reshape X_hat and X_test to [N, -1] flatten per sample
X_hat_flat = np.reshape(X_hat, (len(X_hat), -1))
X_test_flat = np.reshape(X_test, (len(X_test), -1))

# Compute per-sample correlation coefficient (magnitude of normalized inner product)
n1 = np.sqrt(np.sum(np.conj(X_test_flat) * X_test_flat, axis=1)).astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat_flat) * X_hat_flat, axis=1)).astype('float64')
aa = np.abs(np.sum(np.conj(X_test_flat) * X_hat_flat, axis=1))
rho_samples = aa / (n1 * n2)
rho_mean = np.mean(rho_samples)

print("Correlation is ", rho_mean)

# Save rho to CSV
np.savetxt("result/rho_%s.csv" % file, rho_samples, delimiter=",")

# ---------------- Visualization (first 10 samples) ----------------
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original amplitude
    ax = plt.subplot(2, n, i + 1)
    x_testplo = np.abs(x_test[i, 0, :, :] - 0.5 + 1j * (x_test[i, 1, :, :] - 0.5))
    plt.imshow(np.max(np.max(x_testplo)) - x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

    # Reconstructed amplitude
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = np.abs(x_hat[i, 0, :, :] - 0.5 + 1j * (x_hat[i, 1, :, :] - 0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo)) - decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

plt.show()
