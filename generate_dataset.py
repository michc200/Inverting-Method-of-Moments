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
    This function calculates the autocorrelation of a signal for the values in (0, length).
    """
    n = len(signal)
    assert n >= length, "maximum lag should be less than the length of the signal"

    padded_signal = np.pad(signal, (0, length), 'constant')
    acor = np.zeros(length)
    for lag in range(length):
        acor[lag] = np.sum(padded_signal[0:n] * padded_signal[lag:n+lag])
    return acor / n

def get_third_moment(signal: np.ndarray, length: int) -> np.ndarray:
    """
    Computes the third-order moment matrix of a signal.
    Parameters:
        signal (np.ndarray): Input signal.
        length (int): Maximum lag in each direction for the third-order moment.
    Returns:
        np.ndarray: 2D matrix of shape (length, length).
    """
    assert len(signal) >= length, "maximum lag should be less than the length of the signal"
    n = len(signal)
    third_moment = np.zeros((length, length))
    padded_signal = np.pad(signal, (0, length), 'constant')
    
    for lag_1 in range(0, length):
        for lag_2 in range(0, length):
            third_moment[lag_1, lag_2] = np.sum(
                padded_signal[0:n] * 
                padded_signal[0 + lag_1 : n + lag_1] * 
                padded_signal[0 + lag_2 : n + lag_2])
    
    return third_moment / n

def add_gaussian_noise(signal: np.ndarray, sigma: int = 1) -> np.ndarray:
    noise = np.random.normal(0, sigma, np.shape(signal))
    return signal + noise

def create_clean_observation(base_signal: np.ndarray, observation_length: int, gamma: float = 0.2) -> np.ndarray:
    """
    Generates a clean observation signal by embedding multiple instances of a base signal 
    into a zero-padded array while maintaining a separation condition.
    Parameters:
    -----------
    base_signal : np.ndarray
        The base signal to be embedded into the observation.
    observation_length : int
        The total length of the observation signal.
    gamma : float, optional
        A parameter controlling the density of the base signal instances in the observation.
        Must be less than or equal to 0.5. Default is 0.2.
    Returns:
    --------
    np.ndarray
        The generated observation signal containing multiple instances of the base signal 
        separated by zeros.
    """
    # TODO: Change implementation to fix non-uniformity in the density 
    L = len(base_signal)
    instance_number = int((observation_length * gamma) / L)
    assert gamma <= 0.5, "gamma should be less than 0.5 for the separation condition to hold"
    assert instance_number * (L*2-1) <= observation_length, f"For seperation condition, (L*2-1) * instance_number = {(2*L-1) * instance_number} should be <= N={observation_length}"
    assert instance_number > 0, "At least one instance of the base signal should be embedded in the observation"

    observation = np.zeros(observation_length)
    available_zeros = observation_length - instance_number * (L*2)
    padded_base = np.pad(base_signal, (0, L-1), 'constant')

    start = 0    
    for i in range(instance_number):
        shift = np.random.choice(np.arange(available_zeros))
        observation[start + shift : start + shift + 2*L - 1] += padded_base
        available_zeros -= shift
        start += 2*L + shift

    return observation

def generate_data_set(iterations: int, observation_length: int, signal_length: int, sigma: int, seed: int = 0, gamma: float = 0.2, file_path: str = os.path.dirname(__file__)) -> None:
    """
    Generates a dataset by creating a Gaussian signal, adding noise, and calculating statistical moments.
    Parameters:
        iterations (int): The number of iterations to generate noisy signals.
        observation_length (int): The length of the noisy signal observations.
        signal_length (int): The length of the base Gaussian signal.
        sigma (int): The standard deviation of the Gaussian noise to be added.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        gamma (float, optional): A parameter controlling the density of the base signal instances in the observation. Must be less than or equal to 0.5. Default is 0.2.
        file_path (str, optional): The path to save the dataset. Defaults to os.path.dirname(__file__).
    Returns:
        None: The function does not return any value but performs calculations for each iteration.
    """
    np.random.seed(seed)
    
    
    means = torch.zeros(iterations)  
    acors = torch.zeros(iterations, signal_length)  
    third_moments = torch.zeros(iterations, signal_length, signal_length)
    base_signal = np.random.normal(0, 1, signal_length)
 
    for iteration in range(iterations):
        if (iteration + 1) % 10 == 0:
            print(f"Generating iteration {iteration+1}")

        # Create an observation of base signal with density gamma:
        clean_observation = create_clean_observation(base_signal=base_signal, observation_length=observation_length, gamma=gamma)

        # Add noise
        noise = np.random.normal(0, sigma, observation_length)
        noisy_signal = clean_observation + noise

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
    # TODO: use the functions in utils and experiment_recovery to invert the moments, using my functions to calculate the moments
    generate_data_set(iterations=1, observation_length=10**5, signal_length=21, sigma=0.5, seed=312, gamma=0.2, file_path=os.path.join(BASE_PATH, "long_observation"))