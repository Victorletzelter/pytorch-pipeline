import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import yaml
import numpy as np

class SALSA(torch.nn.Module):
    """
    A PyTorch implementation of the SALSA (Spatial Cue-Augmented Log-Spectrogram Features) transform.
    """
    def __init__(self, n_fft, hop_length, itd=True, icld=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.itd = itd
        self.icld = icld

    def calculate_itd(self, signal):
        """
        Calculates the inter-channel time difference of arrival (ITD) for each channel in the signal.
        """
        # Example implementation:
        # Compute the cross-correlation between each channel and the reference channel
        reference_channel = 0
        itd = np.zeros((signal.shape[0], signal.shape[1]))
        for channel in range(signal.shape[0]):
            if channel != reference_channel:
                cross_correlation = np.correlate(signal[reference_channel], signal[channel], mode='full')
                lag = np.argmax(cross_correlation) - signal.shape[1]
                itd[channel] = lag / float(self.sample_rate)
        return itd

    def calculate_icld(self, signal):
        """
        Calculates the inter-channel level difference (ICLD) for each channel in the signal.
        """
        icld = np.zeros((signal.shape[0], signal.shape[1]))
        for channel in range(signal.shape[0]):
            for ref_channel in range(signal.shape[0]):
                if channel != ref_channel:
                    icld[channel] = 20*np.log10(np.mean(signal[channel])/np.mean(signal[ref_channel]))
        return icld

    def forward(self, signal):
        signal = signal.squeeze(0)  # remove the batch dimension
        signal = signal.numpy()  # convert the tensor to a numpy array
        signal = signal.T  # transpose the signal to have shape (n_channels, n_samples)

        # Compute the STFT
        stft_matrix = np.abs(np.fft.fft(signal, n=self.n_fft))

        # Compute the log-spectrogram
        log_spectrogram = np.log10(stft_matrix ** 2)

        # Extract the inter-channel time difference of arrival (ITD) and inter-channel level difference (ICLD)
        itd = self.calculate_itd(signal) if self.itd else None
        icld = self.calculate_icld(signal) if self.icld else None

        # Concatenate the log-spectrogram and the spatial cues to form the SALSA features
        if self.itd and self.icld:
            salsa_features = np.concatenate((log_spectrogram, itd, icld), axis=0)
        elif self.itd:
            salsa_features = np.concatenate((log_spectrogram, itd), axis=0)
        elif self.icld:
            salsa_features = np.concatenate((log_spectrogram, icld), axis=0)
        else:
            salsa_features = log_spectrogram
        # Convert the features to a tensor and return
        salsa_features = torch.from_numpy(salsa_features)
        return salsa_features

class Audio_preprocess_dataset(Dataset):
    # Class for performing audio preprocessing, possibly using a GPU. 
    def __init__(self, config, annotation_loader_with_indexes, audio_loader, device):
        """
        Initializes the Audio_preprocess_dataset class. It loads the configuration settings from the yaml file config, 
        the annotations data using annotation_loader, sets the audio data directory 
        and device, assigns the audio_loader and config as instance variable. 
        It also creates a directory to save the preprocessed data if it doesn't already exist.
        """
        with open(config, 'r') as f:
            config = yaml.safe_load(f) #The configuration of the preprocessing is loaded
        self.annotations = annotation_loader_with_indexes(config['annotations_dir'])
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
        # label = self._get_audio_sample_label(index)
        signal, sr = self.audio_loader(audio_path = audio_sample_path, config = self.config)
        signal = signal.to(self.device)
        return signal

    def _get_audio_sample_path(self, index):
        """
         Returns the file path of the audio sample at index index supposing the audio is in the WAV format.
        """
        filenames = {}
        for key in self.annotations:
            number = key.split("_")[-1]
            filenames[number] = key.split('_idx_'+number)[-2]
        
        filename_from_index = filenames[str(index)]+'.wav'
        path = os.path.join(self.audio_dir, filename_from_index)

        return path

    def _get_audio_sample_label(self, index):
        """
        Returns the label of the audio sample at index index.
        """
        filenames = {}
        for key in self.annotations:
            number = key.split("_")[-1]
            filenames[number] = key.split('_idx_'+number)[-2]
        
        filename_from_index = filenames[str(index)]

        return self.annotations[filename_from_index+'_idx_'+str(index)]

    def preprocess_and_save(self, batch_size=1,num_workers=0,save_as_tensor=True):
        """
        Creates a PyTorch DataLoader object with the dataset, which can be used to iterate over the dataset in batches. 
        It applies preprocessing to the audio signals using the function load_audio and saves the preprocessed data as tensors 
        if save_as_tensor is True, otherwise it saves the data in audio format.
        """
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        for i, data in enumerate(dataloader):
            print(i)
            signals = data
            for j in range(batch_size):
                signal = signals[j]
                filename = os.path.basename(self._get_audio_sample_path(i*batch_size+j))
                processed_path = os.path.join(self.processed_data_dir, 'processed_'+filename.split('.wav')[-2])

                if save_as_tensor: 
                    torch.save(signal, processed_path)

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
            signal = torch.stft(signal, n_fft=transform['n_fft'], hop_length=transform['hop_length'],window=transform['window'],return_complex=True)
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
        elif transform['type'] == 'SALSA':
            salsa = SALSA(n_fft=transform['n_fft'], hop_length=transform['hop_length'], itd=transform['itd'], icld=transform['icld'])
            signal = salsa(signal)
        elif transform['type'] == 'SALSA_LITE':
            signal = torch.stft(signal, n_fft=transform['n_fft'], hop_length=transform['hop_length'])
            magnitude_spectrogram = torch.abs(signal)
            log_magnitude_spectrogram = torch.log(magnitude_spectrogram + transform['log_epsilon'])
            if 'icld' in transform and transform['icld']:
                icld = torch.sum(magnitude_spectrogram[:,0,:] * magnitude_spectrogram[:,1,:], dim=1)
                icld = torch.div(icld, torch.sum(magnitude_spectrogram[:,0,:]**2, dim=1) + torch.sum(magnitude_spectrogram[:,1,:]**2, dim=1))
                icld = torch.log(icld + transform['log_epsilon'])
                signal = torch.cat((log_magnitude_spectrogram, icld.view(-1,1)),dim=1)
            else:
                signal = log_magnitude_spectrogram
        elif transform['type'] == 'GCC_PHAT':
            epsilon = 1e-8
            if 'STFT' not in [i['type'] for i in config['transforms']]:
                signal = torch.stft(signal, n_fft=transform['n_fft'], hop_length=transform['hop_length'])
            # reshaping the tensors 
            magnitude_spectrogram1 = signal[0,:,:].view(-1,1)
            magnitude_spectrogram2 = signal[1,:,:].view(-1,1)
            numerator = torch.sum(magnitude_spectrogram1*torch.conj(magnitude_spectrogram2), dim=0)
            denominator = torch.sqrt(torch.sum(torch.abs(magnitude_spectrogram1)**2, dim=0)*torch.sum(torch.abs(magnitude_spectrogram2)**2, dim=0))
            denominator += epsilon
            gcc_phat = torch.div(numerator, denominator)
            signal = torch.angle(gcc_phat)
        elif transform['type'] == 'PCEN':
            signal = pcen_audio(signal,alpha=transform['alpha'],delta=transform['delta'],r=transform['r'],s=transform['s'],epsilon=transform['epsilon'])

    return signal, sr

def pcen_audio(signal, sr, alpha=0.98, delta=2, r=0.5, s=0.025, epsilon=1e-8, n_mels=128, n_fft=2048, hop_length=None):
    """
    Perform PCEN transform on raw audio signal
    Parameters:
    signal: 1D tensor representing the raw audio signal
    sr: Sample rate of the raw audio signal
    alpha: The alpha parameter in the PCEN formula.
    delta: The delta parameter in the PCEN formula.
    r: The r parameter in the PCEN formula.
    s: The s parameter in the PCEN formula.
    epsilon: The epsilon parameter in the PCEN formula.
    n_mels: Number of mel filters to use
    n_fft: FFT window size
    hop_length: The hop length of the STFT
    return: 3D tensor representing the PCEN transformed mel frequency spectrogram
    """
    # Compute mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(signal)
    E_ = torch.cat([mel_spectrogram[:, :1], mel_spectrogram[:, :-1]], dim=1)
    M = torch.nn.functional.conv1d(E_, torch.tensor([1-s, s], device=mel_spectrogram.device).view(1,2,1), stride=1)
    M = M + epsilon
    return ((mel_spectrogram / (M)**alpha) + delta)**r - delta**r

def load_annotations_with_indexes(annotation_dir) :
    """
    Function that loads annotations files supposed to be in CSV format in a given directory. 
    This also associates indexes with each file to be consistent with the torch.utils.Dataset parent class. 
    The loaded annotation for each file are stored in a dictionnary.
    """
    index = 0
    annotations = {}
    for file in os.listdir(annotation_dir) :
        annotations['{}'.format(file.split('.csv')[-2])+'_idx_{}'.format(str(index))] = torch.tensor(pd.read_csv(os.path.join(annotation_dir,file)).values)
        index += 1
    print('OK')
    return annotations

# def load_annotations(annotations_file):
#     return pd.read_csv(annotations_file)

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

# def calculate_gcc_phat(magnitude_spectrogram1, magnitude_spectrogram2):
#     """
#     This function takes two magnitude spectrograms and calculates the GCC-PHAT values.
#     It first calculates the numerator of the GCC-PHAT values by taking the element-wise product of the two magnitude spectrograms and summing the result over the frequency axis.
#     Then it calculates the denominator of the GCC-PHAT values by taking the sum of the squares of the magnitudes of each spectrogram and taking the square root of the product of the two sums.
#     Finally, it calculates the GCC-PHAT values by taking the angle of the element-wise division of the numerator by the denominator.
#     """
#     numerator = np.sum(magnitude_spectrogram1*np.conj(magnitude_spectrogram2), axis=0)
#     denominator = np.sqrt(np.sum(np.abs(magnitude_spectrogram1)**2, axis=0)*np.sum(np.abs(magnitude_spectrogram2)**2, axis=0))
#     gcc_phat = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
#     return np.angle(gcc_phat)


# def calculate_gcc_phat_all_pairs(magnitude_spectrograms):
#     """
#     This function takes a list of magnitude spectrograms, one per channel, and calculates the GCC-PHAT values between all pairs of channels.
#     It creates an empty array, gcc_phat_all_pairs, with dimensions (num_channels,num_channels,magnitude_spectrograms[0].shape[0],magnitude_spectrograms[0].shape[1]) to store the GCC-PHAT values.
#     It iterates over all pairs of channels, and for each pair, it calls the calculate_gcc_phat() function with the two magnitude spectrograms to get the GCC-PHAT values.
#     Then it stores the GCC-PHAT values in the gcc_phat_all_pairs array.
#     """
#     num_channels = len(magnitude_spectrograms)
#     gcc_phat_all_pairs = np.zeros((num_channels,num_channels,magnitude_spectrograms[0].shape[0],magnitude_spectrograms[0].shape[1]))
#     for i in range(num_channels):
#         for j in range(i,num_channels):
#             gcc_phat_all_pairs[i,j,:,:] = calculate_gcc_phat(magnitude_spectrograms[i], magnitude_spectrograms[j])
#             gcc_phat_all_pairs[j,i,:,:] = calculate_gcc_phat(magnitude_spectrograms[j], magnitude_spectrograms[i])
#     return gcc_phat_all_pairs

def main():
    # set the device to use for preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load the configuration file
    CONFIG_FILE = "preprocessing-config.yml"

    # create the dataset
    dataset = Audio_preprocess_dataset(config=CONFIG_FILE, annotation_loader_with_indexes=load_annotations_with_indexes, audio_loader=load_audio, device=device)

    # # preprocess and save the audio data
    batch_size = 1
    num_workers = 0
    save_as_tensor = True
    dataset.preprocess_and_save(batch_size, num_workers, save_as_tensor)

if __name__ == '__main__':
    main()
