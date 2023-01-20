import torch
import numpy as np 
import torchaudio
from torch.utils.data import Dataset
import new_datasets
import pandas as pd

def test_audio_preprocess_dataset_stft():
    # Create a sample config dictionary
    config = {'sample_rate': 44100, 'n_fft': 2048, 'hop_length': 512, 'itd': False, 'icld': False}
    # Create a sample annotation dataframe
    annotation_data = {'filename': ['sample1.wav', 'sample2.wav'], 'label': [0, 1]}
    annotation_df = pd.DataFrame(annotation_data)
    # Create a sample audio loader function
    def audio_loader(filename):
        return torch.randn(1, 1, 44100)
    # Create an instance of the Audio_preprocess_dataset class
    dataset = new_datasets.Audio_preprocess_dataset(config, annotation_df, audio_loader, 'cpu')
    # Create a dataloader to get the data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Get a sample from the dataset
    for data in dataloader:
        signal, label = data
        break
    # Compute the STFT using the torchaudio library
    stft = torchaudio.transforms.STFT(n_fft=2048, hop_length=512)
    stft_output = stft(signal)
    # Get the magnitude of the output
    stft_output = stft_output[:,:,:,0]
    # Compute the expected output by manually performing the STFT on the signal
    signal = signal.squeeze(0)
    signal = signal.numpy()
    signal = signal.T
    expected_output = np.abs(np.fft.fft(signal, n=2048))
    expected_output = torch.from_numpy(expected_output)
    # Assert that the output of the dataset is equal to the expected output
    assert torch.allclose(stft_output, expected_output, atol=1e-5)

def main():
    test_audio_preprocess_dataset_stft()

if __name__ == '__main__':
    main()
