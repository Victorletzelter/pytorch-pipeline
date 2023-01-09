# data_augmentation.py

import numpy as np
import random
import librosa

def add_noise(data: np.ndarray, noise_factor: float) -> np.ndarray:
    """
    Add white noise to the audio data.
    
    Parameters:
        data: An array of audio data.
        noise_factor: A factor controlling the amount of noise to add. 
                      0.0 means no noise, 1.0 means maximum noise.
    
    Returns:
        The audio data with noise added.
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def shift_pitch(data: np.ndarray, sample_rate: int, pitch_factor: float) -> np.ndarray:
    """
    Shift the pitch of the audio data up or down.
    
    Parameters:
        data: An array of audio data.
        sample_rate: The sample rate of the audio data.
        pitch_factor: A factor controlling the amount of pitch shift.
                      Values greater than 1.0 shift the pitch up, 
                      values less than 1.0 shift the pitch down.
    
    Returns:
        The audio data with the pitch shifted.
    """
    return librosa.effects.pitch_shift(data, sample_rate, pitch_factor)

def stretch_time(data: np.ndarray, sample_rate: int, stretch_factor: float) -> np.ndarray:
    """
    Stretch or compress the time of the audio data.
    
    Parameters:
        data: An array of audio data.
        sample_rate: The sample rate of the audio data.
        stretch_factor: A factor controlling the amount of time stretch.
                        Values greater than 1.0 stretch the time, 
                        values less than 1.0 compress the time.
    
    Returns:
        The audio data with the time stretched.
    """
    return librosa.effects.time_stretch(data, stretch_factor)

def random_augmentation(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply a random combination of noise, pitch shift, and time stretch to the audio data.
    
    Parameters:
        data: An array of audio data.
        sample_rate: The sample rate of the audio data.
    
    Returns:
        The augmented audio data.
    """
    # Choose a random augmentation function and apply it to the data
    augment_fn = random.choice([add_noise, shift_pitch, stretch_time])
    return augment_fn(data, sample_rate, random.uniform(0.8, 1.2))
