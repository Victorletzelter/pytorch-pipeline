import os
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa
import yaml
import numpy as np
import torch.multiprocessing
import multiprocessing
from framing import *

def stacked_covmat_eigh(arr):
    """
    Given an array of shape hape ``(freqbins, t, ch)``, first computes
    an array of shape ``(freqbins, ch, ch)``, where for each freqbin
    we compute the ``(ch, ch)`` spatial covariance matrix, averaged
    among all given ``t``. Then, it computes the eigendecomposition
    of the covariance matrices.
    :returns: The pair ``(ews, evs)`` of shapes ``(freqbins, ch)``
      and ``(freqbins, ch, ch)``, containing the per-freqbin eigenvalues
      and eigenvectors, respectively.
    This function makes use of two main speedup strategies: first, all
    freqbins are computed in parallel. Second, since the covmat is
    Hermitian, we only need one of its triangles to compute the
    decomposition, via ``np.linalg.eigh(covmats, UPLO='U')``.
    """
    f, _, ch = arr.shape
    covmats = np.zeros((f, ch, ch), dtype=arr.dtype)
    for i in range(ch):
        for j in range(i, ch):
            ch_i, ch_j = arr[:, :, i], arr[:, :, j]
            covmats[:, i, j] = (ch_i * ch_j.conj()).sum(axis=-1)
    ews, evs = np.linalg.eigh(covmats, UPLO="U")
    #
    return ews, evs


class SalsaNoiseFloorTracker:
    """
    Heuristic noise floor tracker as proposed and implemented in the SALSA
    paper, and refactored here into a class.
    """
    def __init__(self, initial_floor, steps=3,
                 up_ratio_initial=1.02, up_ratio_many=1.002, down_ratio=0.98,
                 epsilon=1e-6):
        """
        This noise floor tracker operates on each frequency bin independently,
        through time. Depending on the values, the floor will rise or sink.
        :param initial_floor: Array of shape ``(freq,)``, representing the
          initial state of the noise floor tracker.
        :param int steps: Number of consecutive steps to be considered in
          order to decide how to update the noise floor.
        :param up_ratio_initial: Ratio to raise noise floor within the initial
          ``steps``.
        :param float up_ratio_many: Slower ratio to raise noise floor when the
          number of consecutive ``steps`` has been surpassed.
        :param float down_ratio: Ratio to lower noise floor.
        :param float epsilon: Lower bound for the floor values, will be clipped
          to this.
        """
        self.steps = steps
        self.epsilon = epsilon
        self.up_init = up_ratio_initial
        self.up_many = up_ratio_many
        self.down = down_ratio
        #
        self.floor = initial_floor.copy()
        if (self.floor < epsilon).any():
            print(f"WARNING: modifying floor values to be >= {epsilon}.")
            self.floor[self.floor < epsilon] = epsilon
        #
        self.tracker = np.zeros(self.floor.shape, dtype=np.int64)

    def __call__(self, frame, floor_mask_ratio=1.5):
        """
        Call this method with a new frame of frequency bins to update the
        current noise floor and retrieve a mask with the bins that are
        considered above noise floor.
        :param frame: Array of same shape as ``initial_floor``.
        :param floor_mask_ratio: All bins with values above the updated
          noise floor times this ratio will be True in the returned mask.
        :returns: Mask of same shape as given frame, with True whenever
          the entry is above noise floor times ``floor_mask_ratio``.
        """
        # check "above noise floor" entries and update tracker
        above_floor = frame > self.floor
        not_above_floor = ~above_floor
        self.tracker += above_floor
        few_consecutive = above_floor & (self.tracker <= self.steps)
        many_consecutive = above_floor & (self.tracker > self.steps)
        # floor rises more for the first consecutive samples above noise. After
        # several consecutive samples, floor rises more slowly
        self.floor[few_consecutive] *= self.up_init
        self.floor[many_consecutive] *= self.up_many
        # floor sinks whenever signal is not above noise, but stays >=epsilon
        self.floor[not_above_floor] *= self.down
        self.floor[self.floor < self.epsilon] = self.epsilon
        # Reset tracker for any reading that wasn't above noise floor
        self.tracker[not_above_floor] = 0
        # Compute and return above-noise-floor mask
        mask = frame > (floor_mask_ratio * self.floor)
        return mask


class SpatialFeaturesAbstract:
    """
    This class contains all the common functionality among different SALSA
    representations:
    * Calculation of lower, upper and cutoff frequencies
    * Calculation of frequency normalization vector
    * Method to calculate STFTs and log-mel spectrograms
    * Full feature pipeline that computes (and optionally clips) the STFTs and
      log-mels, computes (and optionally clips) the spatial features,
      and finally retrieves the log-mels and spatial features concatenated.
    To use it, extend the ``features(stft, norm_freq, **kwargs):`` method
    with the desired feature and then call the instance with the desired
    kwargs. See SALSA and SALSA-Lite examples below.
    """

    SOUND_SPEED = 343  # m/s
    F_DTYPE = np.float32

    def __init__(self, fs=24000, stft_winsize=512, hop_length=300,
                 fmin_doa=50, fmax_doa=2000, fmax_spec=9000):
        """
        """
        n_bins = stft_winsize // 2 + 1
        # freqs can be cropped between lower and cutoff bin to prevent spatial
        # aliasing. Once cropped, all phase feats above upper can be set to 0
        lower_bin = np.int(np.floor(fmin_doa * stft_winsize / np.float(fs)))  # 512: 1; 256: 0
        lower_bin = np.max((1, lower_bin))
        upper_bin = np.int(np.floor(fmax_doa * stft_winsize / np.float(fs)))  # 9000Hz: 512: 192, 256: 96
        # Cutoff frequency for spectrograms
        cutoff_bin = np.int(np.floor(fmax_spec * stft_winsize / np.float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
        assert upper_bin <= cutoff_bin, "Upper bin for spatial feature is " + \
            "higher than cutoff bin for spectrogram!"
        # Normalization factor
        self.delta = 2 * np.pi * fs / (stft_winsize * self.SOUND_SPEED)
        # feature bins will be divided by this: (freq, 1)
        self.norm_freq = np.arange(n_bins, dtype=self.F_DTYPE)[:, None]
        self.norm_freq[0, 0] = 1  # from salsa lite code
        self.norm_freq *= self.delta
        #
        self.stft_winsize = stft_winsize
        self.hop_length = hop_length
        self.n_bins = n_bins
        #
        self.lobin, self.upbin, self.cutbin = lower_bin, upper_bin, cutoff_bin

    def spectrograms(self, wavchans):
        """
        :param wavchans: Float array of shape ``(channels, samples)``
        :returns: A pair ``(stfts, log_specs)``, each element of shape
          ``(channels, freqbins, time)``.
        """
        n_chans, _ = wavchans.shape  # (n_chans, n_samples)
        # first compute logmel spectrograms for all channels
        log_specs = []
        for ch_i in np.arange(n_chans):
            stft = librosa.stft(
                y=np.asfortranarray(wavchans[ch_i, :]),
                n_fft=self.stft_winsize, hop_length=self.hop_length,
                center=True, window="hann", pad_mode="reflect")
            if ch_i == 0:
                n_frames = stft.shape[1]
                stfts = np.zeros((n_chans, self.n_bins, n_frames),
                                 dtype="complex")
            stfts[ch_i, :, :] = stft
            # Compute log linear power spectrum
            spec = (np.abs(stft) ** 2)
            log_spec = librosa.power_to_db(
                spec, ref=1.0, amin=1e-10, top_db=None)
            log_specs.append(log_spec)
        log_specs = np.stack(log_specs)  # (ch, freqbins, time)
        #
        return stfts, log_specs

    def features(self, stft, norm_freq, **kwargs):
        """
        Extend this method with the desired functionality. It must fulfill
        the following interface:
        * Inputs: ``(stft, norm_freq, **kwargs)``, where stft is a complex
          array of shape ``(ch, freq, t)``, and norm_freq is a float array
          of shape ``(freq, 1)``.
        * Output: Feature array of shape ``(ch, freq, t)``
        See e.g. the ``SalsaFeatures`` and ``SalsaLiteFeatures`` classes.
        """
        raise NotImplementedError("Implement features here!")

    def __call__(self, wavchans, clip_freqs, clip_spatial_alias,
                 **feat_kwargs):
        """
        :param wavchans: Float array of shape ``(channels, samples)``
        :param bool clip_freqs: Whether to remove undesired frequency bins
        :param bool clip_spatial_alias: Whether to zero-out potentially
          aliasing freqbins from the spatial features
        :returns: Array of shape ``(n_feats, freqbins, time)``, where the
          number of features equals ``channels + spatial_feats``, because it
          is a concatenation of the logmel spectrograms and the result of the
          ``features`` method.
        """
        _, _ = wavchans.shape  # (n_chans, n_samples) test if rank 2
        assert wavchans.dtype == self.F_DTYPE, f"{self.F_DTYPE} expected!"
        stfts, log_specs = self.spectrograms(wavchans)  # (ch, f, t)

        if clip_freqs:
            stfts = stfts[:, self.lobin:self.cutbin]
            log_specs = log_specs[:, self.lobin:self.cutbin]
            norm_freq = self.norm_freq[self.lobin:self.cutbin, :]
        else:
            norm_freq = self.norm_freq

        spatial_feats = self.features(stfts, norm_freq, **feat_kwargs)

        if clip_spatial_alias:
            spatial_feats[:, self.upbin:] = 0
        result = np.concatenate([log_specs, spatial_feats])
        return result


# #############################################################################
# # SALSA
# #############################################################################
class SalsaFeatures(SpatialFeaturesAbstract):
    """
    On-the-fly, parallelized CPU implementation of the SALSA features from
    the original paper.
    Usage example::
      sf = SalsaFeatures(fs=sr, stft_winsize=STFT_WINSIZE,
                         hop_length=STFT_HOP,
                         fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
      s = sf(wav, clip_freqs=True, clip_spatial_alias=False,
             ew_thresh=5.0, covmat_avg_neighbours=3,
             is_tracking=True, floor_mask_ratio=1.5)
    """

    def features(self, stfts,
                 norm_freq,
                 ew_thresh: float = 5.0,
                 covmat_avg_neighbours: int = 3,
                 is_tracking: bool = True,
                 floor_mask_ratio: float = 1.5):
        """
        This is a parallelized version of extract_normalized_eigenvector as
        originally implemented. This version has been tested to be correct up
        to sign flip in eigenvectors (since eigendecomposition is invariant to
        eigenvector sign). See class docstring for usage example.
        :param stfts: complex STFT of shape ``(n_chans, n_bins, n_frames)``,
          clipped between lower_bin and upper_bin.
        :param norm_freq: Array of shape ``(n_bins, 1)``, used to normalize
          features by frequency as explained in the paper.
        :param float ew_thresh: Required ratio between largest and 2nd-largest
          eigenvalue, used in the coherence test: all timefreq bins with
          covmats below this ratio will be ignored.
        :param int covmat_avg_neighbours: At each timepoint, the function will
          include this many points to the left and right to calculate the avg
          spatial covariance matrix. E.g. if 3 is given, 7 neighbouring
          matrices in total will be averaged.
        :param is_tracking: If True, use a heuristic noise-floor tracker to
          ignore noisy freqbins.
        :param float floor_mask_ratio: Any timefreq bins with intensity below
          noise level times this float will be considered noisy and ignored.
        :returns: Array of shape ``(n_chans-1, n_bins, n_frames)`` containing
          the SALSA features.
        """
        stfts = stfts.transpose(1, 2, 0)  # (freq, t, ch)
        n_bins, n_frames, n_chans = stfts.shape
        result = np.zeros((n_chans - 1, n_bins, n_frames))

        # padding stfts for avg covmat computation
        stft_pad = np.pad(
            stfts, ((0, 0), (covmat_avg_neighbours, covmat_avg_neighbours),
                    (0, 0)), "wrap")

        # amplitude spectrogram
        signal_magspec = np.abs(  # (freqs, T)
            stft_pad[:, covmat_avg_neighbours:covmat_avg_neighbours + n_frames,
                     0])

        # Initial noisefloor assuming first few frames are noise
        noise_floor = 0.5 * np.mean(signal_magspec[:, 0:5], axis=1)  # (freqs,)
        noise_tracker = SalsaNoiseFloorTracker(
            initial_floor=noise_floor, steps=3,
            up_ratio_initial=1.02, up_ratio_many=1.002,
            down_ratio=0.98, epsilon=1e-6)

        # Default mask is always all ones, so define it just once
        if not is_tracking:
            allpass_mask = np.ones(signal_magspec.shape[0], dtype=np.bool)

        for iframe, magspec_col in enumerate(signal_magspec.T,
                                             covmat_avg_neighbours):
            # Optionally, use noise tracker to mask out noisy bins
            if is_tracking:
                mask = noise_tracker(magspec_col,
                                     floor_mask_ratio=floor_mask_ratio)
            else:
                mask = allpass_mask

            # Compute spatial covmat eigendecomposition for all non-noisy bins
            readings = stft_pad[mask, iframe - covmat_avg_neighbours:
                                iframe + covmat_avg_neighbours + 1, :]
            ews, evs = stacked_covmat_eigh(readings)

            # Further remove from mask any bins with bad coherence
            good_coherence_mask = ews[:, -1] > (ews[:, -2] * ew_thresh)
            mask[mask] = good_coherence_mask

            # compute SALSA features for any non-masked bins
            evs = evs[good_coherence_mask]
            max_evs = evs[:, :, -1]  # all "last columns"
            norm_evs = np.angle(max_evs[:, 0:1].conj() * max_evs[:, 1:])
            norm_evs /= norm_freq[mask]
            # update result
            result[:, mask, iframe - covmat_avg_neighbours] = norm_evs.T
        #
        return result  # (ch, freq, t)

# #############################################################################
# # SALSA LITE
# #############################################################################
class SalsaLiteFeatures(SpatialFeaturesAbstract):
    """
    On-the-fly, parallelized CPU implementation of the SALSA-Lite features from
    the original paper.
    Usage example::
    slf = SalsaLiteFeatures(fs=sr, stft_winsize=STFT_WINSIZE,
                            hop_length=STFT_HOP,
                            fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
    sl = slf(wav, clip_freqs=True, clip_spatial_alias=False)
    """

    def features(self, stfts, norm_freq):
        """
        This is a parallelized version of extract_normalized_eigenvector as
        originally implemented. This version has been tested to be correct up
        to sign flip in eigenvectors (since eigendecomposition is invariant to
        eigenvector sign). See class docstring for usage example.
        :param stfts: complex STFT of shape ``(n_chans, n_bins, n_frames)``,
          clipped between lower_bin and upper_bin.
        :param norm_freq: Array of shape ``(n_bins, 1)``, used to normalize
          features by frequency as explained in the paper.
        """
        result = np.angle(stfts[None, 0].conj() * stfts[1:])
        result /= norm_freq
        return result  # (ch, freq, t)

class Audio_preprocess_dataset(Dataset):
    # Class for performing audio preprocessing, possibly using a GPU. 
    def __init__(self, config, annotation_loader_with_indexes, audio_loader, device):
        """
        Initializes the Audio_preprocess_dataset class. It loads the configuration settings from the yaml file config, 
        the annotations data using annotation_loader, sets the audio data directory 
        and device, assigns the audio_loader and config as instance variable. 
        It also creates a directory to save the preprocessed data if it doesn't already exist.
        """
        if type(config) is str : #If the config is given as a path and not as a dictionnary
            with open(config, 'r') as f:
                config = yaml.safe_load(f) #The configuration of the preprocessing is loaded
                
        self.annotations = annotation_loader_with_indexes(config['annotations_dir'])
        self.annotations = convert_polar_to_cartesian(self.annotations) #Supposing that the output data is in the polar format, the output 
        # is converted into the cartesian format.
        self.audio_dir = config['audio_dir']
        self.device = device 
        self.audio_loader = audio_loader
        self.config = config
        self.is_framed = False #Bolean indicating whether the framing operation has been performed. 
        self.frame_generator = AudioFrameGenerator(sample_rate = config['sample_rate'], frame_size = config['num_samples'], hop_size = config['hop_framing_length'],
                                                   annotation_resolution=config['annotation_resolution'])

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
        if self.is_framed is False : 
            audio_sample_path = self._get_audio_sample_path(index,wav_mode=True)
            unformatted_labels = self._get_audio_sample_label(index)
            signal = self.audio_loader(audio_path = audio_sample_path, config = self.config, is_framed=False)
            labels = load_labels(unformatted_labels, self.config)
        else : 
            audio_sample_path = self._get_audio_sample_path(index,wav_mode=False)
            labels = self._get_audio_sample_label(index)
            signal = self.audio_loader(audio_path = audio_sample_path, config = self.config, is_framed=True)
        signal = signal.to(self.device)
        labels = labels.to(self.device)
        return signal,labels

    def _get_audio_sample_path(self, index, wav_mode=True):
        """
        Returns the file path of the audio sample at index index supposing the audio is in the WAV format.
        """
        filenames = {}
        for key in self.annotations:
            number = key.split("_")[-1]
            filenames[number] = key.split('_idx_'+number)[-2]
        
        if wav_mode is True :
            filename_from_index = filenames[str(index)]+'.wav'
            path = os.path.join(self.audio_dir, filename_from_index)

        else : 
            filename_from_index = filenames[str(index)]      
            path = os.path.join(self.framed_dir, filename_from_index)

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
    
    def frame_all(self) :
        """
        Split the audio data into frames of equal length using the AudioFrameGenerator. 
        """
        self.framed_dir = os.path.join(self.processed_data_dir,'framed_signals_dir')
        self.framed_labels_dir = os.path.join(self.processed_data_dir,'framed_labels_dir')
            
        if not os.path.exists(self.framed_dir):
            os.makedirs(self.framed_dir)
            os.makedirs(self.framed_labels_dir)
            
        for index in range(len(self.annotations)) :
            
            audio_sample_path = self._get_audio_sample_path(index)
            file_name = os.path.basename(audio_sample_path)
            unformatted_labels = self._get_audio_sample_label(index)
            labels = load_labels(unformatted_labels, self.config)
            
            audio, _ = torchaudio.load(audio_sample_path)
            
            frames = self.frame_generator.frame(audio, labels) 
            
            frame_number = 0
            
            for frame in frames :
                signal, label = frame
                if self.frame_generator.is_framed :
                    torch.save(signal, os.path.join(self.framed_dir,'frame_{}_'.format(frame_number)+file_name.split('.wav')[-2]))
                    torch.save(label, os.path.join(self.framed_labels_dir,'labels_'+'frame_{}_'.format(frame_number)+file_name.split('.wav')[-2]))
                frame_number+=1
                
        #Reindexing of all the frames
        self.is_framed = True
        self.annotations = {}
        index = 0
        for file in os.listdir(self.framed_labels_dir) :
            self.annotations['{}'.format(file.split('labels_')[-2]+file.split('labels_')[-1]+'_idx_{}'.format(str(index)))] = torch.load(os.path.join(self.framed_labels_dir,file))
            index += 1      

    def preprocess_and_save(self, batch_size=1,num_workers=0,save_as_tensor=True):
        """
        Creates a PyTorch DataLoader object with the dataset, which can be used to iterate over the dataset in batches. 
        It applies preprocessing to the audio signals using the function load_audio and saves the preprocessed data as tensors 
        if save_as_tensor is True, otherwise it saves the data in audio format.
        """
        dataloader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, num_workers=num_workers, shuffle=False,pin_memory=False)
        for i, data in enumerate(dataloader):
            print(i)
            signals, labels = data
            for j in range(batch_size):
                signal = signals[j]
                corresponding_labels = labels[j]
                filename = os.path.basename(self._get_audio_sample_path(i*batch_size+j))
                processed_signals_path = os.path.join(self.processed_data_dir, 'processed_'+filename.split('.wav')[-2])
                processed_labels_path = os.path.join(self.processed_data_dir, 'processed_labels_'+filename.split('.wav')[-2])

                if save_as_tensor: 
                    torch.save(signal, processed_signals_path)

                else : 
                    torchaudio.save(processed_signals_path, signal.to("cpu"), sr=self.config['sample_rate'])
                
                torch.save(corresponding_labels, processed_labels_path)
                
                # If a "spectrogram type" transform is used among the transforms, we save the associated time values.
                for transform in self.config['transforms']:
                    if transform['type'] in ['STFT','PowerSpectrogram','MelSpectrogram','SALSA','SALSA_LITE'] : 
                        times_values = coord_time(n=signal.shape[2], sr=self.config["sample_rate"], hop_length=transform["hop_length"],n_fft = transform["n_fft"])
                        np.save(file = os.path.join(self.processed_data_dir,'time_values_prosseced_'+filename.split('.wav')[-2]),arr = times_values)
                        break

def load_audio(audio_path, config, is_framed=True):
    """
    Loads a tensor corresponding to a framed audio file and applies various preprocessing steps like resampling, 
    cutting, padding, and spectrogram transformation to the signal based on the settings in config.
    """
    if is_framed is True :
        signal = torch.load(audio_path)
    
    else : 
        signal, sr = torchaudio.load(audio_path)
    
    # if "sample_rate" in config :
    #     signal = _resample_if_necessary(signal, sr, config['sample_rate']) #For re sampling the signal
    
    # if config['mix_down'] is True: 
    #     signal = _mix_down_if_necessary(signal) #For collapsing the channels into one single channel by averaging
    
    # if "num_samples" in config :
    #     signal = _cut_if_necessary(signal, config['num_samples']) #For cropping the signal in accordance with config['num_samples']
    #     signal = _right_pad_if_necessary(signal, config['num_samples']) #For padding the signal in accordance with config['num_samples']

    for transform in config['transforms']:

        if 'window' in transform : #The processing is for now only adapted with Hann window.
            if transform['window']=='hann' :
                window_fn=torch.hann_window

        if transform['type'] == 'STFT':
            signal = torchaudio.transforms.Spectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn)(signal)
        elif transform['type'] == 'PowerSpectrogram':
            signal = torchaudio.transforms.Spectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn,power=2)(signal)
        elif transform['type'] == 'MelSpectrogram':
            signal = torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'], n_mels=transform['n_mels'],n_fft=transform['n_fft'], 
            hop_length=transform['hop_length'],window_fn=window_fn)(signal)
        elif transform['type'] == 'Scale':
            signal = transform['mean']+transform['std']*(signal-signal.mean())/signal.std()
        elif transform['type'] == 'MFCC':
            signal = torchaudio.transforms.MFCC(sample_rate=config['sample_rate'], n_mfcc=transform['n_mfcc'])(signal)
        elif transform['type'] == 'MinMaxScaler':
            signal = (signal-signal.min())/(signal.max()-signal.min())
        elif transform['type'] == 'AmplitudeToDB':
            signal = torchaudio.transforms.AmplitudeToDB()(signal)
        elif transform['type'] == 'TimeStretch':
            if 'STFT' not in [i['type'] for i in config['transforms']]:
                signal = torchaudio.transforms.Spectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn)(signal)
            signal = torchaudio.transforms.TimeStretch(fixed_rate=transform['fixed_rate'])(signal)
        elif transform['type'] == 'FrequencyMasking':
            if 'STFT' not in [i['type'] for i in config['transforms']]:
                signal = torchaudio.transforms.Spectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn)(signal)
            signal = torchaudio.transforms.FrequencyMasking(freq_mask_param=transform['freq_mask_param'])(signal)
        elif transform['type'] == 'TimeMasking':
            if 'STFT' not in [i['type'] for i in config['transforms']]:
                signal = torchaudio.transforms.Spectrogram(n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn)(signal)
            signal = torchaudio.transforms.TimeMasking(time_mask_param=transform['time_mask_param'])(signal)
        elif transform['type'] == 'SALSA':
            sf = SalsaFeatures(fs=config['sampling_rate'], stft_winsize=transform['n_fft'],hop_length=transform['hop_length'],fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
            signal = sf(audio_path, clip_freqs=True, clip_spatial_alias=False,ew_thresh=5.0, covmat_avg_neighbours=3,is_tracking=True, floor_mask_ratio=1.5)
        elif transform['type'] == 'SALSA_LITE':
            slf = SalsaLiteFeatures(fs=config['sampling_rate'], stft_winsize=transform['n_fft'],hop_length=transform['hop_length'],fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
            signal = slf(audio_path, clip_freqs=True, clip_spatial_alias=False)
        elif transform['type'] == 'MelSpecGCC':
            # Code based on the SALSA-Lite description of the features by Nguyen et al. : https://arxiv.org/pdf/2111.08192.pdf
            epsilon = 1e-8
            
            if 'MelSpectrogram' not in [i['type'] for i in config['transforms']]:
                mel_signal = torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'], n_mels=transform['n_mels'],
                                                              n_fft=transform['n_fft'], hop_length=transform['hop_length'],window_fn=window_fn)(signal)
            
            num_channels = mel_signal.shape[0]
            signal = torch.clone(mel_signal)
            
            for i in range(num_channels) :
                for j in range(i,num_channels) : 
                    
                    complex_spectrogram_i = mel_signal[i,:,:] #The spectrogram corresponding to the ith channel
                    complex_spectrogram_j = mel_signal[j,:,:] #The spectrogram corresponding to the jth channel
            
                    numerator = complex_spectrogram_i*torch.conj(complex_spectrogram_j)
                    denominator = torch.abs(complex_spectrogram_i*torch.conj(complex_spectrogram_j))
                    denominator += epsilon
                    quotient = torch.div(numerator,denominator)
                    gcc_phat = torch.fft.ifft(quotient, dim=1)
                    
                    signal = torch.cat((signal,gcc_phat[None,:]),dim=0) #TODOO : Check for the lagging question

        elif transform['type'] == 'PCEN':
            signal = pcen_audio(signal,alpha=transform['alpha'],delta=transform['delta'],r=transform['r'],s=transform['s'],epsilon=transform['epsilon'])

    return signal

def convert_polar_to_cartesian(dict_polar_tensors) :
    """
    Function that converts a dictionnary of 2D tensors whose last two columns correspond to positions in polar coordinates (i.e azimut and
    elevation in that order), into another dictionnary of 2D tensors in which those last two columns are converted in the cartesian format 
    considering the sources a located in the unit sphere (i.e radius = 1).
    """
    dict_cartesian_tensors = {}
    
    for key in dict_polar_tensors : 
    
        azi_rad = dict_polar_tensors[key][:,-2]*np.pi/180
        ele_rad = dict_polar_tensors[key][:,-1]*np.pi/180
        tmp_label = torch.cos(ele_rad)
        x = torch.cos(azi_rad)*tmp_label
        y = torch.sin(azi_rad)*tmp_label
        z = torch.sin(ele_rad)
        dict_cartesian_tensors[key] = torch.cat((dict_polar_tensors[key][:,0:3],x[:,None],y[:,None],z[:,None]),dim=1)
    
    return dict_cartesian_tensors 

def load_labels(cartesian_tensor, config):
        """
        Loads a 2D tensor of shape (..., 5) and converts it into a (n_label_frames, n_classes, 4) tensor.
        Credits: https://github.com/sharathadavanne/seld-dcase2022/blob/c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986/cls_feature_class.py
        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """
        n_labels_steps = cartesian_tensor[-1,0].int()
        
        SED_label = torch.zeros((n_labels_steps, config['data']['nb_unique_classes']))
        x_label = torch.zeros((n_labels_steps, config['data']['nb_unique_classes']))
        y_label = torch.zeros((n_labels_steps, config['data']['nb_unique_classes']))
        z_label = torch.zeros((n_labels_steps, config['data']['nb_unique_classes']))
        
        for line in range(cartesian_tensor.shape[0]) :
            if cartesian_tensor[line,0]<n_labels_steps :
              SED_label[cartesian_tensor[line,0].int(),cartesian_tensor[line,1].int()]=1
              x_label[cartesian_tensor[line,0].int(),cartesian_tensor[line,1].int()]=cartesian_tensor[line,2]
              y_label[cartesian_tensor[line,0].int(),cartesian_tensor[line,1].int()]=cartesian_tensor[line,3]
              z_label[cartesian_tensor[line,0].int(),cartesian_tensor[line,1].int()]=cartesian_tensor[line,4]
            
        doa_mat = torch.stack((x_label,y_label,z_label),dim=-1)
        label_mat = torch.cat((SED_label[:,:,None],doa_mat),dim=-1)
        
        return label_mat

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
    return annotations

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

def samples_to_time(samples, sr) :
    """
    Credits : Librosa https://librosa.org/doc/main/_modules/librosa/core/convert.html
    """
    return np.asanyarray(samples) / float(sr)

def frames_to_samples(frames,hop_length,n_fft) :
    """
    Credits : Librosa https://librosa.org/doc/main/_modules/librosa/core/convert.html
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)

def frames_to_time(frames,sr,hop_length,n_fft) :
    """
    Credits : Librosa https://librosa.org/doc/main/_modules/librosa/core/convert.html
    """
    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)

    return samples_to_time(samples, sr=sr)

def coord_time(n: int, sr: float = 22050, hop_length: int = 512,n_fft = 1024) :
    """Get time coordinates from frames. 
    Credits : Librosa https://librosa.org/doc/0.9.0/_modules/librosa/display.html"""
    times: np.ndarray = frames_to_time(np.arange(n), sr=sr, hop_length=hop_length,n_fft=n_fft)
    return times

def main():
    # set the device to use for preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load the configuration file
    CONFIG_FILE = sys.argv[1]

    # create the dataset
    dataset = Audio_preprocess_dataset(config=CONFIG_FILE, annotation_loader_with_indexes=load_annotations_with_indexes, audio_loader=load_audio, device=device)

    dataset.frame_all()
    
    # preprocess and save the audio data
    batch_size = 2
    num_workers = 0
    save_as_tensor = True
    dataset.preprocess_and_save(batch_size, num_workers, save_as_tensor)

if __name__ == '__main__':
    main()

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
