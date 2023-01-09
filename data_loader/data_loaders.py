from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import torch
from torch.utils.data import Dataset, DataLoader, BaseDataLoader
import scipy
from scipy.io.wavefile import read

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class AudioDataLoader(BaseDataLoader):
    """
    Audio data loader class that loads audio data from a directory and
    returns a tuple of audio data and labels.
    """
    
    def __init__(self, data_dir, batch_size, shuffle=True, transform=None):
        """
        Initialize the AudioDataLoader.
        
        Parameters
        ----------
        data_dir : str, Directory containing the audio data.
        batch_size : int, Number of audio data points to include in each batch.
        shuffle : bool, optional. Whether to shuffle the audio data. Defaults to True.
        transform : callable, optional. Optional transform to apply to the audio data.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load audio data and labels
        self.audio_data, self.labels = self._load_data()
        
        # Initialize the data loader using the audio data and labels
        super(AudioDataLoader, self).__init__(
            dataset=list(zip(self.audio_data, self.labels)),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
    def _load_data(self):
        """
        Load audio data and labels from the data directory.
        """
        audio_data = []
        labels = []
        
        # Iterate over files in the data directory
        for filename in os.listdir(self.data_dir):
            # Load audio data and label from file
            audio, label = self._load_audio_file(os.path.join(self.data_dir, filename))

            if self.transform:
                audio = self.transform(audio)

            audio_data.append(audio)
            labels.append(label)
        
        return audio_data, labels
    
    def _load_audio_file(self, filepath):
        # Load audio data and sample rate from WAV file
        audio, sr = read(filepath)
        # Extract label from file name
        label = os.path.splitext(os.path.basename(filepath))[0]
        return audio, label
