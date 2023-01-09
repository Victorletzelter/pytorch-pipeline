# data_storage.py

import pickle
import numpy as np

def save_to_file(data: np.ndarray, filepath: str) -> None:
    """
    Save the audio data to a file.
    
    Parameters:
        data: An array of audio data.
        filepath: The filepath to save the data to.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_from_file(filepath: str) -> np.ndarray:
    """
    Load audio data from a file.
    
    Parameters:
        filepath: The filepath to load the data from.
    
    Returns:
        The audio data loaded from the file.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
