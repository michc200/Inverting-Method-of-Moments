import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os

BASE_PATH = os.path.dirname(__file__)

def save_data(data: torch.Tensor, path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    file_index = 0
    while os.path.exists(os.path.join(path, f"data_{file_index}.pt")):
        file_index += 1
    torch.save(data, os.path.join(path, f"data_{file_index}.pt"))

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

def generate_data_set(iterations: int, observation_length: int, signal_length: int, sigma: int, seed: int = 0, K: int = 0, file_path: str = os.path.dirname(__file__)) -> None:
    """
        Generates a dataset by creating a Gaussian signal, adding noise, and calculating statistical moments.
    Parameters:
        iterations (int): The number of iterations to generate noisy signals.
        observation_length (int): The length of the noisy signal observations.
        signal_length (int): The length of the base Gaussian signal.
        sigma (int): The standard deviation of the Gaussian noise to be added.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        K (int, optional): The starting position of the signal in the noise. Defaults to 0.
        file_path (str, optional): The path to save the dataset. Defaults to os.path.dirname(__file__).
    Returns:
        None: The function does not return any value but performs calculations for each iteration.
    """
    # Note: At the moment there is only the ability to generate a dataset with a single instance of a signal in it.
    # If there's a need to generate a dataset with multiple signals in it, the code will need to be modified - TBD.
    np.random.seed(seed)
    
    
    means = torch.zeros(iterations)  
    acors = torch.zeros(iterations, 2*signal_length - 1)  
    third_moments = torch.zeros(iterations, 2*signal_length - 1, 2*signal_length - 1)  

    # means = []
    # acors = []
    # third_moments = []

    for iteration in range(iterations):
        # Generate a gaussian signal N(0, 1):
        if iteration % 10 == 0:
            print(f"Generating iteration {iteration}")
        base_signal = np.random.normal(0, 1, signal_length)
        
        # Add noise
        noise = np.random.normal(0, sigma, observation_length)
        padded_base = np.pad(base_signal, (K, observation_length - K - signal_length), 'constant')
        noisy_signal = padded_base + noise

        # calculate moments:
        mean = np.mean(noisy_signal)
        means[iteration] = mean
        acor = get_autocorrelation(noisy_signal, length=signal_length)
        acors[iteration] = torch.tensor(acor)
        third_moment = get_third_moment(noisy_signal, length=signal_length)
        third_moments[iteration] = torch.tensor(third_moment)


    # Save data to the specified path without overwriting existing files
    save_data({"scalars": means, "arrays": acors, "matrices": third_moments}, file_path)


# Example usage
if __name__ == "__main__":
    generate_data_set(iterations=10, observation_length=200, signal_length=100, sigma=1, seed=0, K=0)