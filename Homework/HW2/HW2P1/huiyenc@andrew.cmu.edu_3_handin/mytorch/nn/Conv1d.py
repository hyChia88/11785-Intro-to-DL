import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A  # store A for backward pass

        N, C_in, W_in = self.A.shape
        C_out, C_in, k = self.W.shape
        
        # Get correct output width
        W_out = W_in - k + 1

        # Initialize output
        Z = np.zeros((N, C_out, W_out))
        
        # Vectorized convolution still using a loop over positions
        for i in range(W_out):
            for c_out in range(C_out):
                # Extract window patches for this position - shape: [N, C_in, k]
                window = A[:, :, i:i+k]
                
                # Multiply window with weights and sum - shape: [N]
                Z[:, c_out, i] = np.sum(window * self.W[c_out, :, :], axis=(1, 2))
                
        # Add bias (broadcasting)
        Z += self.b.reshape(1, C_out, 1)
            
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        N, C_in, W_in = self.A.shape
        C_out, C_in, k = self.W.shape
        _, _, W_out = dLdZ.shape
        
        # Pad dLdZ for full convolution
        pad = k - 1
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        
        # Flip weights for the convolution transpose operation
        W_flip = np.flip(self.W, axis=2)
        
        # Initialize gradients
        dLdA = np.zeros_like(self.A)
        self.dLdW = np.zeros_like(self.W)
        
        # Compute dLdA using vectorized operations
        for i in range(W_in):
            for c_in in range(C_in):
                for c_out in range(C_out):
                    # For each input position and channel, compute gradient contribution
                    dLdA[:, c_in, i] += np.sum(
                        dLdZ_pad[:, c_out, i:i+k] * W_flip[c_out, c_in, :], 
                        axis=1
                    )
        
        # Compute dLdW - more reliable implementation
        for c_out in range(C_out):
            for c_in in range(C_in):
                for i in range(k):
                    # For each weight, compute gradient contribution
                    self.dLdW[c_out, c_in, i] = np.sum(
                        self.A[:, c_in, i:i+W_out] * dLdZ[:, c_out, :]
                    )
        
        # Compute dLdb by summing over batch and width dimensions
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize Conv1d() and Downsample1d() instance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input
        A_pad = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A_pad)

        # Downsample
        Z_downsampled = self.downsample1d.forward(Z)

        return Z_downsampled

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward to upsample
        dLdZ_upsampled = self.downsample1d.backward(dLdZ) 

        # Call Conv1d_stride1 backward
        dLdA_pad = self.conv1d_stride1.backward(dLdZ_upsampled)

        # Remove padding if needed
        if self.pad > 0:
            return dLdA_pad[:, :, self.pad:-self.pad]
        else:
            return dLdA_pad