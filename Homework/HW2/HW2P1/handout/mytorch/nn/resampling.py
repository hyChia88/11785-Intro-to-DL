import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
            
        """
        self.A = A
        # Q: Create a new array Z with the correct shape
        # 1. Get W_in from A, A_shape = N x C x W_in
        N, C, W_in = A.shape
        k = self.upsampling_factor
        
        # 2. Get W_out from W_in (scalar)
        W_out = k * (W_in -1)+1
        
        # 3. Initialize Z, Z_shape = N x C x W_out
        Z = np.zeros((N,C,W_out), dtype=A.dtype)

        # 4. Use sclicing to fill Z, assign value of A to every kth position
        Z[:, :, ::k] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, W_in = self.A.shape
        k = self.upsampling_factor
        
        # 1. Initialize Z, Z_shape = N x C x W_in
        dLdA = np.zeros((N,C,W_in), dtype=dLdZ.dtype)
        
        # 2. Extract every k-th value from dLdZ
        dLdA[...] = dLdZ[..., ::k]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)
        self.A = A.copy() # store A for backward pass
        k = self.downsampling_factor
        
        # 1. slicing A by k
        Z = A[..., ::k]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.downsampling_factor
        N,C,W_in = self.A.shape
        
        # Create a new array dLdA with the correct shape
        dLdA = np.zeros((N,C,W_in), dtype=dLdZ.dtype)

        # Fill in the values of dLdA with values of A as needed
        dLdA[..., ::k] = dLdZ
        
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.A = A
        k = self.upsampling_factor
        N,C,H_in,W_in = A.shape
        
        H_out=k*(H_in-1)+1
        W_out=k*(W_in-1)+1

        # 1. Create a new array Z with the correct shape
        Z = np.zeros((N,C,H_out,W_out), dtype=A.dtype)

        # 2. Fill in the values of Z by upsampling A, assign values at every k-th steps
        Z[..., ::k, ::k] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        k = self.upsampling_factor
        N,C,H_out,W_out=dLdZ.shape
        
        # 1. Slice dLdZ by the upsampling factor to get dLdA, extract k-th element in both h & w
        dLdA = dLdZ[..., ::k, ::k]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        k = self.downsampling_factor
        N,C,H_in,W_in = A.shape

        # 1. Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)
        self.A = A
        # slice A in 2d by k
        Z = A[..., ::k, ::k]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        k = self.downsampling_factor
        N, C, H_in, W_in = self.A.shape # direct use the size form self.A (input)
        
        # Create a new array dLdA with the correct shape. error accur here why?
        dLdA = np.zeros((N, C, H_in, W_in), dtype=dLdZ.dtype)

        # Fill in the values of dLdA with values of dLdZ as needed
        dLdA[..., ::k, ::k] = dLdZ
        
        return dLdA
