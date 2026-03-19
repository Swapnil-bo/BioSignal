import numpy as np
import pywt
from scipy.signal import welch

def bandpower(epoch, sf=160., band=(8, 30)):
    freqs, psd = welch(epoch, sf, nperseg=sf*2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx], axis=1)  # shape: (n_channels,)

def wavelet_features(epoch, wavelet='db4', level=4):
    coeffs = pywt.wavedec(epoch, wavelet, level=level, axis=1)
    features = []
    for c in coeffs:
        features.append(np.mean(np.abs(c), axis=1))   # mean abs per channel
        features.append(np.std(c, axis=1))             # std per channel
    return np.concatenate(features)  # shape: (n_channels * levels * 2,)

def extract_features(data):
    n_epochs = data.shape[0]
    features = []
    for i in range(n_epochs):
        epoch = data[i]  # shape: (64, 481)
        bp    = bandpower(epoch)
        wt    = wavelet_features(epoch)
        feat  = np.concatenate([bp, wt])
        features.append(feat)
    return np.array(features)

if __name__ == '__main__':
    data   = np.load('data/processed/epochs.npy')
    labels = np.load('data/processed/labels.npy')

    X = extract_features(data)
    print(f"Feature matrix shape : {X.shape}")
    print(f"Labels shape         : {labels.shape}")

    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', labels)
    print("Saved X.npy and y.npy to data/processed/")
