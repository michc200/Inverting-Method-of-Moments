import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

# should I use the biased or unbiased version of the autocorrelation? - currently using biased
def get_autocorrelation(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    return np.convolve(signal, signal[::-1], mode='full')[::-1] / n

def get_third_moment(signal: np.ndarray) -> np.ndarray:
    
    n = len(signal) # max lag
    third_moment = np.zeros((2*n - 1, 2*n - 1))
    padded_signal = np.pad(signal, (n//2 + 1, 0), 'constant') # unsure about padding length
    
    for lag_1 in range(2*n - 1):
        for lag_2 in range(2*n - 1):
            third_moment[lag_1, lag_2] = np.sum(padded_signal*np.roll(padded_signal, lag_1)*np.roll(padded_signal, lag_2))
    
    return third_moment / n

def add_gaussian_noise(signal: np.ndarray, sigma: int = 1) -> np.ndarray:
    noise = np.random.normal(0, sigma, np.shape(signal))
    return signal + noise
    

# Example usage
if __name__ == "__main__":
    signal = np.array(np.arange(3))
    mean = np.mean(signal)
    print(mean)

    acor = get_autocorrelation(signal)
    print(acor)

    third_moment = get_third_moment(signal)
    print(third_moment)
    print('yes')