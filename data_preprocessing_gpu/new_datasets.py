import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import yaml

class Audio_preprocess_dataset(Dataset):
    # Class for performing audio preprocessing, possibly using a GPU. 
    def __init__(self, config, annotation_loader, audio_loader, device):
        """
        Initializes the Audio_preprocess_dataset class. It loads the configuration settings from the yaml file config, 
        the annotations data using annotation_loader, sets the audio data directory 
        and device, assigns the audio_loader and config as instance variable. 
        It also creates a directory to save the preprocessed data if it doesn't already exist.
        """
        with open(config, 'r') as f:
            config = yaml.safe_load(f) #The configuration of the preprocessing is loaded
        self.annotations = annotation_loader(config['annotations_file'])
        self.audio_dir = config['audio_dir']
        self.device = device 
        self.audio_loader = audio_loader
        self.config = config

        if 'preprocessed_data_dir' in config :
            self.processed_data_dir = config['preprocessed_data_dir']        
        else : 
            self.processed_data_dir = os.path.join(self.audio_dir, 'preprocessed_data') 

        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

    def __len__(self):
        """
        Returns the number of audio samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Returns the preprocessed audio signal and label of the sample at index index from the dataset.
        """
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = self.audio_loader(audio_path = audio_sample_path, config = self.config)
        signal = signal.to(self.device)
        return signal, label

    def _get_audio_sample_path(self, index):
        """
         Returns the file path of the audio sample at index index.
        """
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        """
        Returns the label of the audio sample at index index.
        """
        return self.annotations.iloc[index, 6]

    def preprocess_and_save(self, batch_size=1,num_workers=0,save_as_tensor=True):
        """
        Creates a PyTorch DataLoader object with the dataset, which can be used to iterate over the dataset in batches. 
        It applies preprocessing to the audio signals using the function load_audio and saves the preprocessed data as tensors 
        if save_as_tensor is True, otherwise it saves the data in audio format.
        """
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        for i, data in enumerate(dataloader):
            signals, labels = data
            for j in range(batch_size):
                signal = signals[j]
                label = labels[j]
                filename = os.path.basename(self._get_audio_sample_path(i*batch_size+j))
                processed_path = os.path.join(self.processed_data_dir, filename)

                if save_as_tensor: 
                    state = {'signal': signal, 'label': label}
                    torch.save(state, processed_path)

                else : 
                    torchaudio.save(processed_path, signal.to("cpu"), sr=self.config['sample_rate'])

def load_audio(audio_path, config):
    """
    Loads an audio file from the given audio_path using torchaudio.load() and applies various preprocessing steps like resampling, 
    cutting, padding, and spectrogram transformation to the signal based on the settings in config.
    """
    signal, sr = torchaudio.load(audio_path)
    signal = _resample_if_necessary(signal, sr, config['sample_rate'])
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal, config['num_samples'])
    signal = _right_pad_if_necessary(signal, config['num_samples'])

    for transform in config['transforms']:
        if transform['type'] == 'STFT':
            signal = torch.stft(signal, n_fft=transform['n_fft'], hop_length=transform['hop_length'])
        elif transform['type'] == 'PowerSpectrogram':
            signal = torchaudio.transforms.PowerSpectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'])(signal)
        elif transform['type'] == 'MelSpectrogram':
            signal = torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'], n_mels=transform['n_mels'])(signal)
        elif transform['type'] == 'Scaling':
            signal = torchaudio.transforms.Scale(mean=transform['mean'], std=transform['std'])(signal)
        elif transform['type'] == 'MFCC':
            signal = torchaudio.transforms.MFCC(sample_rate=config['sample_rate'], n_mfcc=transform['n_mfcc'])(signal)
        elif transform['type'] == 'MinMaxScaler':
            signal = torchaudio.transforms.MinMaxScaling(min=transform['min'], max=transform['max'])(signal)
        elif transform['type'] == 'AmplitudeToDB':
            signal = torchaudio.transforms.AmplitudeToDB()(signal)
        elif transform['type'] == 'TimeStretch':
            signal = torchaudio.transforms.TimeStretch(rate=transform['rate'])(signal)
        elif transform['type'] == 'FrequencyMasking':
            signal = torchaudio.transforms.FrequencyMasking(freq_mask_param=transform['freq_mask_param'])(signal)
        elif transform['type'] == 'TimeMasking':
            signal = torchaudio.transforms.TimeMasking(time_mask_param=transform['time_mask_param'])(signal)
    return signal, sr

def load_annotations(annotations_file):
    return pd.read_csv(annotations_file)

def _cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal

def _right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def _resample_if_necessary(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def main():
    # set the device to use for preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load the configuration file
    CONFIG_FILE = "preprocessing-config.yml"

    # create the dataset
    dataset = Audio_preprocess_dataset(config=CONFIG_FILE, annotation_loader=load_annotations, audio_loader=load_audio, device=device)

    # preprocess and save the audio data
    batch_size = 16
    num_workers = 0
    save_as_tensor = True
    dataset.preprocess_and_save(batch_size, num_workers, save_as_tensor)

if __name__ == '__main__':
    main()
