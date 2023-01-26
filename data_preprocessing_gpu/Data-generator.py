#
# Data generator for training the SELDnet
# Credits : https://github.com/Victorletzelter/seld-net/blob/51efb7b6d8968a1eec231126de12c9c8e1833ab8/cls_data_generator.py
#

import os
import numpy as np
import Feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, datagen_mode='train', dataset='ansim', ov=1, split=1, db=30, batch_size=32, seq_len=64,
            shuffle=True, nfft=512, classifier_mode='regr', weakness=0, cnn3d=False, xyz_def_zero=False, extra_name='',
            azi_only=False
    ):
        """This function is the constructor of the DataGenerator class. It initializes the class with the given parameters.
        Args:
            datagen_mode (str): The mode of the generator, either 'train' or 'test'
            dataset (str): The name of the dataset to be used
            ov (int): The overlap factor for the dataset
            split (int): The split factor for the dataset
            db (int): The decibel threshold for the dataset
            batch_size (int): The number of samples in each batch
            seq_len (int): The length of the sequence of samples
            shuffle (bool): Whether or not to shuffle the data
            nfft (int): The size of the FFT window
            classifier_mode (str): The type of classifier to be used, either 'regr' or 'clas'
            weakness (int): The weakness factor for the dataset
            cnn3d (bool): Whether or not to use a 3D CNN
            xyz_def_zero (bool): Whether or not to define the xyz axis origin as the zero point
            extra_name (str): Additional name for the dataset
            azi_only (bool): Whether or not to use only azimuth
        """
        self._datagen_mode = datagen_mode
        self._classifier_mode = classifier_mode
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._shuffle = shuffle
        self._feat_cls = FeatureClass.FeatureClass(dataset=dataset, ov=ov, split=split, db=db, nfft=nfft)
        self._label_dir = self._feat_cls.get_label_dir(classifier_mode, weakness, extra_name)
        self._feat_dir = self._feat_cls.get_normalized_feat_dir(extra_name)
        self._thickness = weakness
        self._xyz_def_zero = xyz_def_zero
        self._azi_only = azi_only

        self._filenames_list = list()
        self._nb_frames_file = None     # Assuming number of frames in feat files are the same
        self._feat_len = None
        self._2_nb_ch = 2 * self._feat_cls.get_nb_channels()
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = len(self._class_dict.keys())
        self._default_azi, self._default_ele = self._feat_cls.get_default_azi_ele_regr()
        self._is_cnn3d_model = cnn3d
        self._get_label_filenames_sizes()

        self._batch_seq_len = self._batch_size*self._seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._seq_len * self._batch_size))))

        print(
            'Datagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            'nb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                self._datagen_mode, len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._feat_len, self._2_nb_ch, self._label_len
                )
        )

        print(
            'Dataset: {}, ov: {}, split: {}\n'
            'batch_size: {}, seq_len: {}, shuffle: {}\n'
            'label_dir: {}\n '
            'feat_dir: {}\n'.format(
                dataset, ov, split,
                self._batch_size, self._seq_len, self._shuffle,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        """
        This function returns the number of samples in the training, validation, and test sets.
        """
        feat_shape = (self._batch_size, self._2_nb_ch, self._seq_len, self._feat_len)
        label_shape = [
            (self._batch_size, self._seq_len, self._nb_classes),
            (self._batch_size, self._seq_len, self._nb_classes*(2 if self._azi_only else 3))
        ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        """
        This function returns the number of batches in the entire dataset.
        """
        return self._nb_total_batches

    def _get_label_filenames_sizes(self):
        """
        This function returns the filenames and sizes of the label files.
        """
        for filename in os.listdir(self._label_dir):
            if self._datagen_mode in filename:
                self._filenames_list.append(filename)

        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._feat_len = temp_feat.shape[1] // self._2_nb_ch

        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
        self._label_len = temp_label.shape[-1]
        self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes
        return

    def generate(self):
        """
        This function is a generator that yields batches of data.
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.
            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0

            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)
                        self._circ_buf_label.append(temp_label[row_cnt])
                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._batch_seq_len, self._feat_len * self._2_nb_ch))
                label = np.zeros((self._batch_seq_len, self._label_len))
                for j in range(self._batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                    label[j, :] = self._circ_buf_label.popleft()
                feat = np.reshape(feat, (self._batch_seq_len, self._feat_len, self._2_nb_ch))

                # Split to sequences
                feat = self._split_in_seqs(feat)
                feat = np.transpose(feat, (0, 3, 1, 2))
                label = self._split_in_seqs(label)

                if self._azi_only:
                    # Get Cartesian coordinates from azi/ele
                    azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
                    x = np.cos(azi_rad)
                    y = np.sin(azi_rad)

                    # Set default Cartesian x,y,z coordinates to 0,0,0
                    if self._xyz_def_zero:
                        no_ele_ind = np.where(label[:, :, 2 * self._nb_classes:] == self._default_ele)
                        x[no_ele_ind] = 0
                        y[no_ele_ind] = 0

                    label = [
                        label[:, :, :self._nb_classes],  # SED labels
                        np.concatenate((x, y), -1)       # DOA Cartesian labels
                    ]
                else:
                    # Get Cartesian coordinates from azi/ele
                    azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
                    ele_rad = label[:, :, 2 * self._nb_classes:] * np.pi / 180
                    tmp_label = np.cos(ele_rad)

                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)

                    # Set default Cartesian x,y,z coordinates to 0,0,0
                    if self._xyz_def_zero:
                        no_ele_ind = np.where(label[:, :, 2 * self._nb_classes:] == self._default_ele)
                        x[no_ele_ind] = 0
                        z[no_ele_ind] = 0
                        y[no_ele_ind] = 0

                    label = [
                        label[:, :, :self._nb_classes],  # SED labels
                        np.concatenate((x, y, z), -1)    # DOA Cartesian labels
                         ]

                yield feat, label

    def _split_in_seqs(self, data):
        """This function splits the input data into sequences of a specified length.
        Args :
            data (numpy array): The input data to be split into sequences
        """
        if len(data.shape) == 1:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :, :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        """
        This function splits multi-channel data into separate channels.

        data (numpy array): The input data to be split into separate channels
        num_channels (int): The number of channels to split the data into
        """
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] // num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_list_index(self, azi, ele):
        """
        This function returns the list index of the given azimuth and elevation.
        Args : 
            azi (int): The azimuth value
            ele (int): The elevation value
        """
        return self._feat_cls.get_list_index(azi, ele)

    def get_matrix_index(self, ind):
        """
        This function returns the matrix index of the given list index.
        Args :
            ind (int): The list index
        """
        return np.array(self._feat_cls.get_vector_index(ind))

    def get_nb_classes(self):
        """
        This function returns the number of classes in the dataset.
        """
        return self._nb_classes

    def nb_frames_1s(self):
        """
        This function returns the number of frames in 1s of audio for the dataset.
        """
        return self._feat_cls.nb_frames_1s()
    
    
    