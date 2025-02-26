import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        # reshape Z from multiple to 1d, -1 as the num of size
        Z = A.reshape(A.shape[0],-1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        # reshape back to input shape
        dLdA = dLdZ.reshape(self.input_shape)

        return dLdA
