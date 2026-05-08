"""
CS-CsiNet Inference Only for CSI Compression and Reconstruction
- 僅做推論：不做訓練
- 使用固定隨機投影矩陣 A 進行 CS 壓縮
- 載入預訓練解碼器進行重建
- 評估 NMSE(dB) 與頻域相關係數
- 視覺化原始/重建 CSI 振幅（前 10 筆）
"""

import os
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# TensorFlow 1.x 與 Keras
import tensorflow as tf
from keras.models import model_from_json


try:
    tf.reset_default_graph()
except Exception:
    pass  

# ========================= 使用者可調參數 =========================
envir = 'indoor'   # 'indoor' 或 'outdoor'
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

# 與訓練時一致的壓縮維度（範例：1/4→512, 1/16→128, 1/32→64, 1/64→32）
encoded_dim = 512

# 模型檔名
model_tag = f'CS-CsiNet_{envir}_dim{encoded_dim}'
model_json_path = os.path.join('saved_model', f'model_{model_tag}.json')
model_weights_path = os.path.join('saved_model', f'model_{model_tag}.h5')

# 隨機投影矩陣 A 的 mat 檔
A_mat_path = os.path.join('data', f'A{encoded_dim}.mat')

# 測試資料 mat 檔（時域與頻域）
if envir == 'indoor':
    test_time_mat_path = os.path.join('data', 'DATA_Htestin.mat')        # x_test: 'HT'
    test_freq_mat_path = os.path.join('data', 'DATA_HtestFin_all.mat')   # X_test: 'HF_all'
else:
    test_time_mat_path = os.path.join('data', 'DATA_Htestout.mat')       # x_test: 'HT'
    test_freq_mat_path = os.path.join('data', 'DATA_HtestFout_all.mat')  # X_test: 'HF_all'

# 視覺化數量
viz_n = 10

# ========================= 工具函式 =========================
def load_decoder(json_path, weights_path):
    with open(json_path, 'r') as f:
        loaded_model_json = f.read()
    decoder = model_from_json(loaded_model_json)
    decoder.load_weights(weights_path)
    return decoder

def load_test_data(time_mat_path, freq_mat_path):
    # 讀取時域測試資料（扁平化）
    mat_t = sio.loadmat(time_mat_path)
    x_test = mat_t['HT']  # shape: [N, img_total]，與訓練時一致
    # 讀取頻域測試資料（相關係數使用）
    mat_f = sio.loadmat(freq_mat_path)
    X_test = mat_f['HF_all']  # shape: [N, img_height, 125]（依資料而定）
    return x_test, X_test

def compress_with_A(x_test_flat, A_path):
    mat = sio.loadmat(A_path)
    A = mat['A']  # shape: [encoded_dim, img_total]
    # y = x A^T
    y = np.dot(x_test_flat, A.T)
    return y

def to_channels_first(x_flat, h, w, c):
    # 轉為 [N, C, H, W]
    return np.reshape(x_flat, (len(x_flat), c, h, w))

def to_complex_from_tensor(tensor_cx):
    """
    將 [0,1] 的兩通道張量轉為複數：
    real = channel0 - 0.5
    imag = channel1 - 0.5
    回傳 shape: [N, H*W] 或保留 2D/3D 視需要
    """
    # tensor_cx: [N, 2, H, W]
    real = np.reshape(tensor_cx[:, 0, :, :], (len(tensor_cx), -1))
    imag = np.reshape(tensor_cx[:, 1, :, :], (len(tensor_cx), -1))
    return (real - 0.5) + 1j * (imag - 0.5)

def compute_nmse_db(x_true_C, x_hat_C):
    # x_true_C, x_hat_C: shape [N, H*W] 的複數
    power = np.sum(np.abs(x_true_C) ** 2, axis=1)
    mse = np.sum(np.abs(x_true_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    nmse_db = 10 * math.log10(nmse)
    return nmse_db

def reconstruct_freq_from_time(x_hat_C, h, w, out_bins=125, pad_width=257):
    """
    將重建的時域 CSI（複數, shape [N, H*W]）還原成 [N, H, W]，
    在頻率維度做 zero-padding 至 pad_width，對該軸做 FFT，
    然後截取前 out_bins 個頻點。
    """
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), h, w))  # [N,H,W]
    pad_len = pad_width - w
    if pad_len < 0:
        raise ValueError("pad_width 必須 >= W")
    # zero-padding on frequency axis
    x_hat_padded = np.concatenate((x_hat_F, np.zeros((len(x_hat_F), h, pad_len))), axis=2)
    X_hat = np.fft.fft(x_hat_padded, axis=2)
    X_hat = X_hat[:, :, 0:out_bins]
    return X_hat

def compute_correlation(X_true, X_hat):
    """
    計算每筆樣本的頻域相關係數 rho：
    rho_i = |<X_true_i, X_hat_i>| / (||X_true_i|| * ||X_hat_i||)
    回傳 (rho_per_sample, rho_mean)
    """
    # 展平成向量
    X_true_f = np.reshape(X_true, (len(X_true), -1))
    X_hat_f = np.reshape(X_hat, (len(X_hat), -1))

    # 內積與 L2 範數
    aa = np.abs(np.sum(np.conj(X_true_f) * X_hat_f, axis=1))
    n1 = np.sqrt(np.sum(np.conj(X_true_f) * X_true_f, axis=1)).astype('float64')
    n2 = np.sqrt(np.sum(np.conj(X_hat_f) * X_hat_f, axis=1)).astype('float64')

    # 每筆 rho
    rho = aa / (n1 * n2 + 1e-12)  # 防止 0 除
    rho_mean = np.mean(rho)
    return rho, rho_mean

def visualize_abs_amplitude(x_true, x_pred, n=10):
    """
    視覺化 |x| 的圖（前 n 筆）
    x_true, x_pred shape: [N, 2, H, W] 的 [0,1] 範圍張量
    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原圖
        ax = plt.subplot(2, n, i + 1)
        x_true_abs = np.abs(x_true[i, 0, :, :] - 0.5 + 1j * (x_true[i, 1, :, :] - 0.5))
        plt.imshow(np.max(np.max(x_true_abs)) - x_true_abs.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()

        # 重建圖
        ax = plt.subplot(2, n, i + 1 + n)
        x_pred_abs = np.abs(x_pred[i, 0, :, :] - 0.5 + 1j * (x_pred[i, 1, :, :] - 0.5))
        plt.imshow(np.max(np.max(x_pred_abs)) - x_pred_abs.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
    plt.show()

# ========================= 主流程 =========================
def main():
    # 1) 載入測試資料（時域 & 頻域）
    x_test_flat, X_test_freq = load_test_data(test_time_mat_path, test_freq_mat_path)
    # 與訓練一致：float32
    x_test_flat = x_test_flat.astype('float32')  # [N, img_total]

    # 2) CS 壓縮（固定隨機投影 A）
    y_test = compress_with_A(x_test_flat, A_mat_path)  # [N, encoded_dim]

    # 3) 載入解碼器並推論
    decoder = load_decoder(model_json_path, model_weights_path)

    t_start = time.time()
    x_hat = decoder.predict(y_test)  # 預期輸出為 [N, 2, H, W]
    t_end = time.time()
    infer_time_per_sample = (t_end - t_start) / x_test_flat.shape[0]
    print("It cost %f sec" % infer_time_per_sample)

    # 4) 準備原始時域張量以評估/視覺化
    x_test_cf = to_channels_first(x_test_flat, img_height, img_width, img_channels)  # [N,2,H,W]

    # 5) 轉為複數表示，計算 NMSE(dB)
    x_test_C = to_complex_from_tensor(x_test_cf)   # [N, H*W] 的複數
    x_hat_C = to_complex_from_tensor(x_hat)        # [N, H*W] 的複數
    nmse_db = compute_nmse_db(x_test_C, x_hat_C)

    # 6) 頻域相關係數（載入原始頻域，對重建時域做 FFT）
    #    原始頻域資料重塑為 [N, H, 125]
    X_test_freq = np.reshape(X_test_freq, (len(X_test_freq), img_height, 125))
    X_hat_freq = reconstruct_freq_from_time(x_hat_C, img_height, img_width, out_bins=125, pad_width=257)
    rho_per_sample, rho_mean = compute_correlation(X_test_freq, X_hat_freq)

    # 7) 印出指標
    print("In %s environment" % envir)
    print("When dimension is", encoded_dim)
    print("NMSE is ", nmse_db)
    print("Correlation is ", rho_mean)

    # 8) 視覺化原始與重建 CSI 振幅（前 n 筆）
    visualize_abs_amplitude(x_test_cf, x_hat, n=viz_n)

if __name__ == '__main__':
    main()
