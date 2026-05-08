"""
Rewrite of CsiNet_onlytest.py with identical functionality:
- Load model architecture (JSON) and weights (.h5)
- Load test data (indoor / outdoor)
- Predict (autoencoder.predict), measure average inference time per sample
- Convert tensors to complex CSI, compute time-domain NMSE (dB)
- Convert reconstructed CSI to frequency domain (FFT with zero-padding) and compute correlation coefficient with original frequency CSI
- Visualize first n samples (original vs reconstructed amplitude)
"""
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Keras / TensorFlow imports compatible with TF1/TF2
import tensorflow as tf
from keras.models import model_from_json

# Try to reset graph for TF1 compatibility; use compat for TF2
try:
    if hasattr(tf, 'reset_default_graph'):
        tf.reset_default_graph()
    else:
        tf.compat.v1.reset_default_graph()
except Exception:
    pass

# -------------------- Configuration --------------------
envir = 'indoor'   # 'indoor' or 'outdoor'
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

# Compression dimension used in file naming (must match saved model)
encoded_dim = 512  # e.g. 512,128,64,32 depending on compression rate
file = 'CsiNet_%s_dim%d' % (envir, encoded_dim)

# Residual blocks count (used to match architecture if needed)
residual_num = 2

# Visualization settings
visualize_n = 10  # number of samples to visualize

# -------------------- Load model architecture & weights --------------------
json_path = "saved_model/model_%s.json" % file
h5_path = "saved_model/model_%s.h5" % file

with open(json_path, 'r') as jf:
    loaded_model_json = jf.read()

autoencoder = model_from_json(loaded_model_json)
autoencoder.load_weights(h5_path)

print("Loaded model from:", json_path, "and", h5_path)

# -------------------- Load test data --------------------
# The original script uses MAT files:
# - time-domain test CSI: DATA_Htestin.mat / DATA_Htestout.mat (variable 'HT')
# - frequency-domain full CSI for correlation: DATA_HtestFin_all.mat / DATA_HtestFout_all.mat (variable 'HF_all')
if envir == 'indoor':
    mat_td = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat_td['HT']  # time-domain test CSI array
    mat_fd = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = mat_fd['HF_all']  # frequency-domain CSI for correlation
else:
    mat_td = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat_td['HT']
    mat_fd = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = mat_fd['HF_all']

# -------------------- Preprocess for model input --------------------
# x_test expected shape in original script: (N, 2, 32, 32) with values in [0,1]
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

# -------------------- Inference (prediction) --------------------
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
avg_time_per_sample = (tEnd - tStart) / x_test.shape[0]
print("It cost %f sec per sample (average)" % avg_time_per_sample)

# -------------------- Convert predicted tensors to complex domain --------------------
# original / predicted format: channels_first [N, 2, H, W], real channel idx 0, imag idx 1,
# and values are in [0,1], so complex value = real-0.5 + 1j*(imag-0.5)
def tensor_to_complex(tensor):
    # tensor shape: (N, 2, H, W)
    real = np.reshape(tensor[:, 0, :, :], (len(tensor), -1))
    imag = np.reshape(tensor[:, 1, :, :], (len(tensor), -1))
    return (real - 0.5) + 1j * (imag - 0.5)

x_test_C = tensor_to_complex(x_test)
x_hat_C = tensor_to_complex(x_hat)

# For time-domain NMSE calculation we want shapes (N, H*W)
# x_test_C and x_hat_C already have shape (N, H*W) from tensor_to_complex

# -------------------- Compute NMSE (time-domain) --------------------
power = np.sum(np.abs(x_test_C) ** 2, axis=1)  # per-sample power
mse = np.sum(np.abs(x_test_C - x_hat_C) ** 2, axis=1)  # per-sample MSE
# avoid divide by zero
eps = 1e-12
nmse_linear = mse / (power + eps)
nmse_db = 10 * np.log10(np.mean(nmse_linear))
print("In %s environment" % envir)
print("When dimension is", encoded_dim)
print("NMSE is (dB):", nmse_db)

# -------------------- Frequency-domain correlation --------------------
# Steps:
# 1) reshape x_hat_C to [N, H, W] (time-domain complex matrix)
# 2) zero-pad along frequency dimension to length 257, perform FFT, take first 125 bins (original frequency bins)
# 3) flatten to (N, -1) and compute per-sample normalized inner product with X_test

# 1) reshape to (N, H, W)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))

# zero-pad to 257 along width, FFT, then take first 125 bins
pad_len = 257
freq_bins_keep = 125
pad_width = pad_len - img_width
if pad_width < 0:
    raise ValueError("Expected img_width <= 257")

zeros_pad = np.zeros((len(x_hat_F), img_height, pad_width), dtype=complex)
x_hat_padded = np.concatenate((x_hat_F, zeros_pad), axis=2)
X_hat = np.fft.fft(x_hat_padded, axis=2)
X_hat = X_hat[:, :, 0:freq_bins_keep]  # (N, H, 125)

# reshape X_hat and X_test to (N, -1)
X_hat_flat = np.reshape(X_hat, (len(X_hat), -1))
X_test_flat = np.reshape(X_test, (len(X_test), -1))

# compute per-sample correlation coefficient (magnitude of normalized inner product)
n1 = np.linalg.norm(X_test_flat, axis=1)  # ||X_test||
n2 = np.linalg.norm(X_hat_flat, axis=1)   # ||X_hat||
# avoid zero norms
n1 = np.maximum(n1, eps)
n2 = np.maximum(n2, eps)
inner = np.abs(np.sum(np.conj(X_test_flat) * X_hat_flat, axis=1))
rho_samples = inner / (n1 * n2)  # per-sample correlation
rho_mean = np.mean(rho_samples)
print("Correlation is (average over test set):", rho_mean)

# -------------------- Visualization --------------------
# Plot absolute amplitude of original and reconstructed complex CSI for first n samples
n = min(visualize_n, len(x_test))
plt.figure(figsize=(20, 4))
for i in range(n):
    # original amplitude (reconstruct from x_test channels)
    orig_complex = (x_test[i, 0, :, :] - 0.5) + 1j * (x_test[i, 1, :, :] - 0.5)
    orig_amp = np.abs(orig_complex)
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.max(orig_amp) - orig_amp.T, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

    # reconstructed amplitude
    rec_complex = (x_hat[i, 0, :, :] - 0.5) + 1j * (x_hat[i, 1, :, :] - 0.5)
    rec_amp = np.abs(rec_complex)
    ax2 = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.max(rec_amp) - rec_amp.T, cmap='gray')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()

plt.show()
