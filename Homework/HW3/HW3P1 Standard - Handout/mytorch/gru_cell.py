import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        # Save input
        self.x = x
        self.hidden = h_prev_t #save prev hidden state

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        
        # reset gate:
        # reset gate determines how much of the previous hidden state to forget
        # r_t = σ(W_rx * x_t + b_rx + W_rh * h_{t-1} + b_rh)
        self.r = self.r_act.forward(x @ self.Wrx.T + self.brx + h_prev_t @ self.Wrh.T + self.brh)
        
        # update gate z 
        # update gate determines how much of the new state to keep vs old state
        # z_t = σ(W_zx * x_t + b_zx + W_zh * h_{t-1} + b_zh)
        self.z = self.z_act.forward( x @ self.Wzx.T + self.bzx + h_prev_t @ self.Wzh.T + self.bzh)

        # candidate hidden state n
        # candidate hidden state is the new information to be added
        # n_t = tanh(W_nx * x_t + b_nx + r_t ⊙ (W_nh * h_{t-1} + b_nh))
        self.n = self.h_act.forward(x @ self.Wnx.T + self.bnx + self.r * (h_prev_t @ self.Wnh.T + self.bnh))
        
        # final hiddent state: 
        # between previous hidden state and new candidate state based on update gate
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        # 1) Reshape vectors for outer product calculations
        x_col = self.x.reshape(-1, 1)
        h_prev_col = self.hidden.reshape(-1, 1)
        
        # 2) Transpose all calculated dWs...
        dz = delta * (self.hidden - self.n)
        dn = delta * (1 - self.z)
        dh_prev_partial = delta * self.z
        
        # 3)
        # Update gate gradient
        dz_new = dz * self.z * (1 - self.z)

        # Candidate state gradient
        dn_new = dn * (1 - self.n ** 2)

        # Reset gate gradient
        dr = dn_new * (self.hidden @ self.Wnh.T + self.bnh)
        dr_new = dr * self.r * (1 - self.r)

        # Accumulate gradients for reset gate weights and biases
        # result1 = np.outer(a, b)
        # result2 = a.reshape(-1, 1) @ b.reshape(1, -1)
        
        self.dWrx = dr_new.reshape(-1, 1) @ x_col.T
        self.dWrh = dr_new.reshape(-1, 1) @ h_prev_col.T
        self.dbrx = dr_new
        self.dbrh = dr_new
        
        # Accumulate gradients for update gate weights and biases
        self.dWzx = dz_new.reshape(-1, 1) @ x_col.T
        self.dWzh = dz_new.reshape(-1, 1) @ h_prev_col.T
        self.dbzx = dz_new
        self.dbzh = dz_new
        
        # Accumulate gradients for candidate state weights and biases
        self.dWnx = dn_new .reshape(-1, 1) @ x_col.T
        self.dWnh = (dn_new  * self.r).reshape(-1, 1) @ h_prev_col.T
        self.dbnx = dn_new 
        self.dbnh = dn_new * self.r
        
        # Compute
        dx = self.Wrx.T @ dr_new + self.Wzx.T @ dz_new + self.Wnx.T @ dn_new

        dh_prev = dh_prev_partial
        dh_prev += self.Wrh.T @ dr_new
        dh_prev += self.Wzh.T @ dz_new
        dh_prev += self.Wnh.T @ (dn_new * self.r)
        
        return dx, dh_prev