# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from Conv1d import Conv1d
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        '''
        stride = 4
        "Since the network has 4 neurons in the final layer and scans with a stride of 4, it produces one 4-channel output every 4 time instants. Since there are 128 time instants in the inputs and no zero-padding is done, the network produces 31 outputs in all, one every 4 time instants. When flattened, this output will have 124 (4 × 31) values."
        
        Checking:
        input length = 128
        1st layer: ((128-8)/4)+1 = 31 steps, 8 as kernal size
        31*4strides = 124 vals
        '''
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4) # C_in=24, C_out=8, k=8, stride=4
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1) # C_in=8, C_out=16, k=1,stride=1 (cuz no scanning)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1) # C_in=16,C_out=4, k=1,stride=1 (cuz no scanning)
        
        # architecture :[Flatten(), Linear(8 * 24, 8), ReLU(), Linear(8, 16), ReLU(), Linear(16, 4)]
        # no need wriet flatten
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] # Add the layers in the correct order

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        # For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)

        w1_t=w1.T
        w1_reshaped = w1_t.reshape(8, 8, 24)
        w1_conv = np.transpose(w1_reshaped, (0, 2, 1)) # change (out_channels, kernel_size, in_channels) to (out_channels, in_channels, kernel_size)
        self.conv1.conv1d_stride1.W = w1_conv
        
        w2_t = w2.T
        w2_reshaped = w2_t.reshape(16, 1, 8)
        w2_conv = np.transpose(w2_reshaped, (0, 2, 1))
        self.conv2.conv1d_stride1.W = w2_conv
        
        w3_t = w3.T
        w3_reshaped = w3_t.reshape(4, 1, 16)
        w3_conv = np.transpose(w3_reshaped, (0, 2, 1))
        self.conv3.conv1d_stride1.W = w3_conv

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
            
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # First layer: 4 distinct filters (4 colors in first hidden layer)
        # kernel_size should match receptive field size
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        
        # Second layer: 8 distinct filters (8 colors in second hidden layer)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        
        # Third layer: 2 distinct filters (2 colors in output layer)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)
        #   4 : Slice the weight matrix and reduce to only the shared weights
        #   (hint: be careful, steps 1-3 are similar, but not exactly like in the simple scanning MLP)
        
        # slice weights
        w1=w1[:48,:2]
        w2=w2[:4,:8]
        w3=w3[:16,:4]
        
        w1_t=w1.T
        # reshape to (out_channel,kernel_size,in_channel)
        k_size_1=self.conv1.kernel_size
        w1_reshaped=w1_t.reshape(2,k_size_1,24)
        self.conv1.conv1d_stride1.W=np.transpose(w1_reshaped,(0,2,1))

        w2_t=w2.T
        k_size_2=self.conv2.kernel_size
        w2_reshaped=w2_t.reshape(8,k_size_2,2)
        self.conv2.conv1d_stride1.W=np.transpose(w2_reshaped,(0,2,1))

        w3_t=w3.T
        k_size_3=self.conv3.kernel_size
        w3_reshaped=w3_t.reshape(4,k_size_3,8)
        self.conv3.conv1d_stride1.W=np.transpose(w3_reshaped,(0,2,1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
