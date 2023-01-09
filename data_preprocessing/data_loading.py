# data_loading.py

import librosa
import requests

def load_from_file(filepath: str) -> np.ndarray:
    """
    Load audio data from a local file.
    
    Parameters:
        filepath: The filepath to load the data from.
    
    Returns:
        The audio data loaded from the file.
    """
    return librosa.load(filepath, sr=None)

def load_from_url(url: str) -> np.ndarray:
    """
    Load audio data from a URL.
    
    Parameters:
        url: The URL to load the data from.
    
    Returns:
        The audio data loaded from the URL.
    """
    response = requests.get(url)
    return librosa.load(response.content, sr=None)
