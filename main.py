import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

def get_autocorrelation(signal: np.ndarray, length: int) -> np.ndarray:
    """
    This function calculates the autocorrelation of a signal for the values in (-length, length).
    """
    n = len(signal)
    padded_signal = np.pad(signal, (n, n), 'constant')
    acor = np.zeros(2*length - 1)
    for lag in range(-length + 1, length):
        acor[lag + length - 1] = np.sum(np.roll(padded_signal, lag) * padded_signal)
    return acor / n

def get_third_moment(signal: np.ndarray, length: int) -> np.ndarray:

    n = length # max lag

    third_moment = np.zeros((2*n - 1, 2*n - 1))
    padded_signal = np.pad(signal, (n, n), 'constant')
    
    for lag_1 in range(-n + 1, n):
        for lag_2 in range(-n + 1, n):
            third_moment[lag_1 + n - 1, lag_2 + n - 1] = np.sum(padded_signal*np.roll(padded_signal, lag_1)*np.roll(padded_signal, lag_2))
    
    return third_moment / len(signal)

def add_gaussian_noise(signal: np.ndarray, sigma: int = 1) -> np.ndarray:
    noise = np.random.normal(0, sigma, np.shape(signal))
    return signal + noise

def main():
    # Generate a gaussian signal:
    base_signal = add_gaussian_noise(np.zeros(1000), 1)

    # Add noise
    Noisy_signal = add_gaussian_noise(base_signal, 0.1)

    # calculate moments:
    mean = np.mean(Noisy_signal)
    acor = get_autocorrelation(Noisy_signal, length=100)
    start = time.time()
    third_moment = get_third_moment(Noisy_signal, length=100)
    end = time.time()
    print(end - start)
    print(f"The mean of the signal is: {mean}")
    print(f"the auto correlation of the signal is: {acor}")
    print(f"the third moment of the signal is {third_moment}")
    print('tat')



# Example usage
if __name__ == "__main__":
    main()
    print('yay')