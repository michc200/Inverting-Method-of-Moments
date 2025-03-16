import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms


def get_autocorrelation(signal):
    n = len(signal)
    return np.convolve(signal, signal[::-1], mode='full') / n

def get_third_moment(signal):
    third_moment = []
    n = len(signal)
    for i in range(len(signal)):
        for j in range(len(signal)):
            for k in range(len(signal)):
                third_moment.append(signal[i]*signal[j]*signal[k])
    return np.array(third_moment) / n

def add_noise(signal, sigma):
    noise = np.random.normal(0, sigma, np.shape(signal))
    return signal + noise
    

# Example usage
# if __name__ == "__main__":
#     signal = np.array([1, 2, 3, 4, 5])