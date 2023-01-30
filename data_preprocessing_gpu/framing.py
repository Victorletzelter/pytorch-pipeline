### Class for turning signal tensor files into frames 

import torch

class AudioFrameGenerator:
    def __init__(self, sample_rate, frame_size, hop_size, annotation_resolution):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.annotation_resolution = annotation_resolution
        self.is_framed = False
        
    def frame(self,audio_signal,labels):
        self.num_channels,self.num_samples = audio_signal.shape
        print('NUM SAMPLES : {}'.format(self.num_samples))
        start_signal = 0
        start_labels = 0
        
        while start_signal + self.frame_size <= self.num_samples:
            self.is_framed = True
            # print('FRAMED AUDIO SIGNAL SHAPE : {}'.format(audio_signal[:,start_signal:start_signal + self.frame_size].shape)) 
            # print('FRAMED LABELS SHAPE : {}'.format(labels[start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:].shape))
            # print('START_SIGNAL : {}'.format(start_signal)) 
            
            # Deal with missing labels in the last frames
            if labels[start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:].shape[0]<self.frame_size//self.annotation_resolution :
                # Calculate padding on the first dimension
                pad_front = 0
                pad_back = self.frame_size//self.annotation_resolution - labels[start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:].shape[0]

                #Reshape label tensor
                labels_tensor=labels[None,start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:]
                
                # Use torch.nn.ConstantPad3d to perform padding
                labels_tensor = torch.nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, pad_back), value=0)(labels_tensor)
                
                #Go back to the initial dimension
                labels_tensor = labels_tensor[0,:,:,:]
            
            else : 
                labels_tensor = labels[start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:]
               
            # yield audio_signal[:,start_signal:start_signal + self.frame_size], labels[start_labels:start_labels + self.frame_size//self.annotation_resolution,:,:]
            yield audio_signal[:,start_signal:start_signal + self.frame_size], labels_tensor
            start_signal += self.hop_size
            start_labels += self.hop_size//self.annotation_resolution
            
        if start_signal + self.frame_size > self.num_samples:
            self.is_framed = False

    # def __iter__(self,audio_signal):
        
    #     self.num_samples, self.num_channels = audio_signal.shape
    #     start = 0
    #     while start + self.frame_size <= self.num_samples:
    #         yield audio_signal[start:start + self.frame_size, :]
    #         start += self.hop_size

    def sample_frames(self, num_frames, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        frame_starts = torch.sort(torch.unique(torch.randint(
            low=0, high=self.num_samples - self.frame_size, size=(num_frames,)
        )))[0]
        return [self.audio_signal[start:start + self.frame_size, :] for start in frame_starts]
