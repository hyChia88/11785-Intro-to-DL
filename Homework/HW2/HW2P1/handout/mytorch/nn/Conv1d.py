# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

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
        # print("dLdW",self.W.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        Func:
        A*W + b
        sliding on A by kernel size and *W, then + b
        
        """
        self.A = A # store A for backward pass

        N, C_in, W_in =  self.A.shape
        C_out, C_in, k = self.W.shape
        
        # Get correct W_out shape
        W_out = W_in - k + 1

        Z = np.zeros((N,C_out, W_out))
        
        # 2. Perform convolution
        # self.b need to reshape to self.b to (1, out_channels, 1).
        # for n in range(N):
        #     for c_out in range(C_out):
        #         for i in range(W_out):
        #             # calc covn at pos (n,c_out,i)
        #             Z[n, c_out, i] = np.sum(A[n, :, i:i+k] * self.W[c_out]) + self.b[c_out]
        
        # Optimized convolution using np.einsum
        for i in range(W_out):
            # Extract window of size k starting at position i
            window = A[:, :, i:i+k]
            # Use einsum for efficient computation across batch, channels and kernel positions
            Z[:, :, i] = np.einsum('nik,oik->no', window, self.W) + self.b[np.newaxis, :]
                    
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        N, C_in, W_in =  self.A.shape
        C_out, C_in, k = self.W.shape
        _, _, W_out = dLdZ.shape
        
        # 1. Pad front & back each k-1
        pad = k - 1
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        
        W_flip= np.flip(self.W, axis=-1)
        
        # 2. Find dLdA
        # broadcast dLdA
        dLdA = np.zeros_like(self.A)
        self.dLdW = np.zeros_like(self.W)
        
        # for i in range(W_in): # length of A
        #     # dLdA[:,:i] = dLdZ_pad[:,:,i:i+k] * W_flip
        #     window = dLdZ_pad[:, :, i:i+k]
        #     dLdA[:, :, i] = np.sum(window[:, :, np.newaxis, :] * W_flip[np.newaxis, :, :, :], axis=(1, 3))
        
        # for i in range(k-W_out+1): # length of A - length of dLdZ + 1
        #     self.dLdW[:,:i] = self.A[:,:,i:i+W_out] * dLdZ
        #     # self.dLdW[:, :, i] = np.sum(self.A[:, :, i:i+W_out] * dLdZ, axis=(0, 2))

        for i in range(W_in):
            for j in range(k):
                if i + j < dLdZ_pad.shape[2]:
                    # For each position in input and kernel
                    # Use matrix multiplication to compute contribution to gradient
                    dLdA[:, :, i] += np.einsum('no,oi->ni', 
                                              dLdZ_pad[:, :, i+j], 
                                              W_flip[:, :, j])
                    
        # Compute dLdW for each position in the kernel
        for c_out in range(C_out):
            for c_in in range(C_in):
                for i in range(k):
                    # Compute contribution to kernel gradient using efficient dot product
                    self.dLdW[c_out, c_in, i] = np.sum(
                        self.A[:, c_in, i:i+W_out] * dLdZ[:, c_out, :]
                    )
        
        # Compute dLdb by summing gradients across batch and spatial dimensions
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        return dLdA
    
        # for c_out in range(C_out):
        #     for c_in in range(C_in):
        #         for i in range(k):
        #             self.dLdW[c_out, c_in, i] = np.sum(self.A[:, c_in, i:i+W_out] * dLdZ[:, c_out, :])
                
                
        # self.dLdb = np.sum(dLdZ, axis=(0,2))
        # # print(self.dLdW)

        # return dLdA

# for any stride > 1
class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride) # get stride to downsample

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A # store A for backward pass
        # 1. Pad the input appropriately using np.pad() function
        A_pad = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant', constant_values=0)

        # 2. Call Conv1d_stride1, calc in stride = 1
        Z = self.conv1d_stride1.forward(A_pad)

        # 3. downsample
        Z_downsampled = self.downsample1d.forward(Z)

        return Z_downsampled

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # 1. Call downsample1d backward to upsample
        dLdZ_upsampled = self.downsample1d.backward(dLdZ) 

        # 2. Call Conv1d_stride1 backward
        dLdA_pad = self.conv1d_stride1.backward(dLdZ_upsampled)

        if self.pad > 0:
            return dLdA_pad[:,:,self.pad:-self.pad]
        else:
            return dLdA_pad