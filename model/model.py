import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net(BaseModel):
    def __init__(self, data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                 rnn_size, fnn_size, classification_mode):
        """
        Initialize the Net model.
        
        Parameters
        ----------
        data_in : tuple, Tuple containing the shape of the input data.
        data_out : tuple, Tuple containing the shape of the output data.
        dropout_rate : float, Float value representing the rate at which to apply dropout.
        nb_cnn2d_filt : int, Integer representing the number of filters to use in the Conv2D layers.
        pool_size : list, List of integers representing the pooling sizes for each Conv2D layer.
        rnn_size : list, List of integers representing the hidden sizes for the GRU layers.
        fnn_size : list, List of integers representing the number of filters for the fully-connected layers.
        classification_mode : str, String representing the classification mode (either "event" or "event_mixture").
        """
        super(Net, self).__init__()

        self.data_in = data_in
        self.data_out = data_out
        self.dropout_rate = dropout_rate
        self.nb_cnn2d_filt = nb_cnn2d_filt
        self.pool_size = pool_size
        self.rnn_size = rnn_size
        self.fnn_size = fnn_size
        self.classification_mode = classification_mode

        self.conv2d_layers = {} 
        self.batchnorm_layers = {}
        self.pooling_layers = {}
        self.dropout_layers = {}

        # Create Conv2D layers
        for i, convCnt in enumerate(self.pool_size):
            conv2d = nn.Conv2d(in_channels=self.data_in[-3], out_channels=self.nb_cnn2d_filt, kernel_size=(3, 3), padding=1)
            self.conv2d_layers['conv_layer_{}'.format(i)] = conv2d
            self.batchnorm_layers['conv_layer_{}'.format(i)] = nn.BatchNorm2d(self.nb_cnn2d_filt)
            self.pooling_layers['conv_layer_{}'.format(i)] = nn.MaxPool2d(kernel_size=(1, self.pool_size[i]))
            self.dropout_layers['conv_layer_{}'.format(i)] = nn.Dropout(p=self.dropout_rate)

        # Create GRU layers
        self.gru_layers = {}
        for i, nb_rnn_filt in enumerate(self.rnn_size):
            gru = nn.GRU(input_size=data_in[-2], hidden_size=nb_rnn_filt, batch_first=True, bidirectional=True)
            self.gru_layers['rnn_layer_{}'.format(i)] = gru

        # Create fully-connected layers
        self.fnn_layers = {}
        for i, nb_fnn_filt in enumerate(self.fnn_size):
            fnn = nn.Linear(in_features=nb_fnn_filt, out_features=nb_fnn_filt)
            self.fnn_layers['fcc_layer_{}'.format(i)] = fnn

        self.fnn_layers['fcc_final_sed'] = nn.Linear(in_features=nb_fnn_filt, out_features=self.data_out[0][-1])  # for SED output
        self.fnn_layers['fcc_final_doa'] = nn.Linear(in_features=nb_fnn_filt, out_features=self.data_out[1][-1])  # for DOA output

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def load_weights(self, weights):
        """
        Load weights into the model.
        
        Parameters
        ----------
        weights : tensor, Tensor containing weights to load into the model.
        """
        # Iterate over Conv2D layers and set their weights
        for layer in self.conv2d_layers:
            self.conv2d_layers[layer].weight = weights[layer]
        # Iterate over GRU layers and set their weights
        for layer in self.gru_layers:
            self.gru_layers[layer].weight = weights[layer]
        # Iterate over fully-connected layers and set their weights
        for layer in self.fnn_layers:
            self.fnn_layers[layer].weight = weights[layer]

    def forward(self, x):
        """
        Perform a forward pass on the model. 
        
        Parameters
        ----------
        x : tensor, Tensor containing a data point of shape data_in
        """
        # Initialize list to hold intermediate outputs
        intermediate_outputs = []

        # Pass input through Conv2D layers
        for i, (conv2d, batchnorm, pooling, dropout) in enumerate(zip(self.conv2d_layers, self.batchnorm_layers, self.pooling_layers, self.dropout_layers)):
            x = self.conv2d_layers[conv2d](x)
            x = self.batchnorm_layers[batchnorm](x)
            x = F.relu(x)
            x = self.pooling_layers[pooling](x)
            x = self.dropout_layers[dropout](x)

            # Add intermediate output to list
            intermediate_outputs.append(x)

        # Flatten intermediate outputs for use in GRU layers
        x = torch.cat(intermediate_outputs, dim=1)
        x = x.view(x.size(0), x.size(1), -1)

        # Pass input through GRU layers
        for gru in self.gru_layers:
            x, _ = self.gru_layers[gru](x)

        # Flatten output for use in fully-connected layers
        x = x.view(x.size(0), -1)

        # Pass input through fully-connected layers
        for fnn in self.fnn_layers:
            x = self.fnn_layers[fnn](x)

        # Return output for SED and DOA
        return x[:, :self.data_out[0][-1]], x[:, self.data_out[0][-1]:]




        




