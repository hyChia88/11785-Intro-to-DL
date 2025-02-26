import numpy as np
from resampling import *

# work for stide == 1
class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A # save for backwards
        C_out, C_in, k, k = self.W.shape
        N, C_in, H_in, W_in = A.shape
        
        W_out = W_in - k + 1
        H_out = H_in - k + 1
        
        # init Z
        Z = np.zeros((N,C_out,H_out, W_out))

        # # Convolution: W as the filter, b as bias, output Z should be N,C_out,H_out,W_out
        # for n in range(N): # for all batch
        #     for c in range(C_out):
        #         for i in range(H_out):
        #             for j in range(W_out):
        #                 # window
        #                 win = A[n,:,i:i+k,j:j+k]
        #                 Z[n, c, i, j] = np.sum(win * self.W[c]) + self.b[c] # win * self.W[c] not matrix multi @

        # Opt
        for h in range(H_out):
            for w in range(W_out):
                Z[:, :, h, w] = np.einsum('ncij,ocij->no', A[:, :, h:h+k, w:w+k], self.W)
        
        Z += self.b.reshape(1, C_out, 1, 1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C_out, H_out, W_out = dLdZ.shape
        _, C_in, H_in, W_in = self.A.shape
        _, _, k, k = self.W.shape
        
        # 1. find dLdb. dLdb as the sum of dLdZ **need to do  axis=(0, 2, 3) cuz b.shape is (C_out,)
        self.dLdb = np.sum(dLdZ, axis=(0,2,3))
        
        # 2. find dLdW. basic rule: A * dLdZ = dLdW
        # init dLdW
        self.dLdW = np.zeros_like(self.W)
        
        # Conv
        # for n in range(N):
        #     for c in range(C_out):
        #         for h in range(H_out):
        #             for w in range(W_out):
        #                 # get gradient
        #                 grad = dLdZ[n,c,h,w]
        #                 win = self.A[n,:, h:h+k, w:w+k]
        #                 self.dLdW[c] += grad * win # matrix multiplication, output dLdW shouldbe [C_in, C_out, K, K]
        
        for h in range(H_out):
            for w in range(W_out):
                win = self.A[:, :, h:h+k, w:w+k] 
                grads = dLdZ[:, :, h, w] 
                 # 'no,ncij->ocij': for each output channel (o) and input channel (c),
                self.dLdW += np.einsum('no,ncij->ocij', grads, win)

        # 3. find dLdA
        # 3.1 pad dLdZ[N,C_out,H_out,W_out] to k-1 at each edge
        dLdZ_pad = np.pad(dLdZ, ((0, 0), (0, 0), (k-1, k-1), (k-1, k-1)), mode='constant')

        # 3.2 flip
        W_flip = np.flip(self.W, (2, 3))
        
        # 3.3 get dLdA, dLdA = dLdZ_pad * W_flip
        # init dLdA
        dLdA = np.zeros_like(self.A)
        # for n in range(N):
        #     for c_in in range(C_in):
        #         for c_out in range(C_out):
        #             for h in range(H_in):
        #                 for w in range(W_in):
        #                     h_start = h
        #                     w_start = w
        #                     window = dLdZ_pad[n, c_out, h_start:h_start+k, w_start:w_start+k]
        #                     # accumulate grad (+= but not =)
        #                     dLdA[n, c_in, h, w] += np.sum(window * W_flip[c_out, c_in])
        
        for h in range(H_in):
            for w in range(W_in):
                win = dLdZ_pad[:, :, h:h+k, w:w+k] 
                dLdA[:, :, h, w] = np.einsum('noij,ocij->nc', win, W_flip)

        return dLdA

# stride != 1
class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        '''
        Specifically, a conv2d stride1 layer followed by a downsample2d layer with factor k=2 is equivalent to a conv2d stride2. More generally, a conv2d stride1 layer followed by a downsample2d layer with factor k is equivalent to a conv1d stride with stride k.
        '''
        
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, 
                                       weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_pad = np.pad(A, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)),mode='constant',constant_values=0)

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A_pad)

        # downsample
        Z_downsampled = self.downsample2d.forward(Z)

        return Z_downsampled

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_pad = self.conv2d_stride1.backward(dLdZ_upsampled)

        # Unpad the gradient
        if self.pad > 0:
            return dLdA_pad[:,:,self.pad:-self.pad,self.pad:-self.pad]
        else:
            return dLdA_pad