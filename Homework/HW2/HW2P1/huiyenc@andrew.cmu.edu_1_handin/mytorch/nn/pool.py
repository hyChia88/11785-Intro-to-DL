import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
            
        numpy.unravel_index(indices, shape, order='C')
        > Converts a flat index or array of flat indices into a tuple of coordinate arrays.
        """
        self.A = A
        N, C_out, H_in, W_in = A.shape
        k = self.kernel
        
        H_out = H_in - k + 1
        W_out = W_in - k + 1
        
        # init Z and indicies
        Z = np.zeros((N,C_out,H_out, W_out))
        self.indices = np.zeros((N,C_out, H_out, W_out,2), dtype=int)
        
        for n in range(N):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        win = A[n,c,h:h+k,w:w+k]
                        # get max of the window
                        Z[n,c,h,w] = np.max(win)
                        # put the grad to where max at
                        max_idx = np.unravel_index(np.argmax(win), (k,k))
                        self.indices[n,c,h,w]=max_idx
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, H_out, W_out = dLdZ.shape
        _, _, H_in, W_in = self.A.shape
        k = self.kernel
        
        # init
        dLdA = np.zeros((N, C, H_in, W_in))
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        i, j = self.indices[n, c, h, w]
                        dLdA[n, c, h+i, w+j] += dLdZ[n, c, h, w]
        
        return dLdA

class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel
        self.A = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, C, H_in, W_in = A.shape
        k = self.kernel
        
        H_out = H_in - k + 1
        W_out = W_in - k + 1
        
        Z = np.zeros((N, C, H_out, W_out))
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        # get mean of the window
                        window = A[n, c, h:h+k, w:w+k]
                        Z[n, c, h, w] = np.mean(window)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        N, C, H_out, W_out = dLdZ.shape
        _, _, H_in, W_in = self.A.shape
        k = self.kernel
        
        dLdA = np.zeros((N, C, H_in, W_in))
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        dLdA[n, c, h:h+k, w:w+k] += dLdZ[n, c, h, w] / (k * k)
        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_upsampled)
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 =MeanPool2d_stride1(kernel)
        self.downsample2d =Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_upsampled)
        return dLdA
