import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug

        # ???: what is the format of in & out feature
        '''
        in features Cin scalar - number of input features
        out features Cout scalar - number of output features
        
        W W matrix Cout × Cin weight parameters
        b b matrix Cout × 1 bias parameters
        '''

        self.in_features = in_features
        self.out_features = out_features
        
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
                - (N × Cin) where N is batch size, Cin is in_features
        
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """

        self.A = A
        # A matrix N × Cin batch of N inputs each represented by Cin features

        self.N = A.shape[0]

        self.ones = np.ones((self.N, 1))

        Z = A @ self.W.T + self.ones @ self.b.T
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        
        - dLdA: (N, Cout)
        - dLdW: (Cout, Cin)
        - dLdb: (Cout, 1)
        """
        self.ones = np.ones((self.N, 1))
        
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.ones

        if self.debug:
            self.dLdA = dLdA

        return dLdA
