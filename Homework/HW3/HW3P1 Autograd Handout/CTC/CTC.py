import numpy as np
import sys

sys.path.append("../")
from mytorch.autograd_engine import *
from mytorch.functional import *

"""
MYNOTE:
CTC (Connectionist Temporal Classification) used to is a type of neural network output and associated scoring
function, for training recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems
where the timing is variable.
"""

class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        skip_connect = []
        for i in range(len(extended_symbols)):
            # if current symbol != BLANK and not = as prev, skip connect
            if i >= 2 and extended_symbols[i] != self.BLANK and extended_symbols[i] != extended_symbols[i-2]:
                skip_connect.append(True)
            else:
                skip_connect.append(False)
        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        alpha[0][0] = logits[0][extended_symbols[0]]
        
        # TODO: Intialize alpha[0][1]
        if S>1:
            alpha[0][1] = logits[0][extended_symbols[1]]

        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # MYNOTE: dynamic filling alpha
        for t in range(1,T):
            for s in range(S):
                symbol = extended_symbols[s]
                alpha[t][s] += alpha[t-1][s] * logits[t][symbol]
                
                # If allow
                if s >= 1:
                    alpha[t][s] += alpha[t-1][s-1] * logits[t][symbol]

                if s >= 2 and skip_connect[s]:
                    alpha[t][s] += alpha[t-1][s-2] * logits[t][symbol]
        # <---------------------------------------------
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # MYNOTE: 后向算法计算β(t,r), 即从时间t和扩展符号位置r到序列末尾的所有路径的总概率
        T = logits.shape[0]
        S = len(extended_symbols)
        
        beta = np.zeros((T, S))
        if S > 1:
            # Init the last as 1.0
            beta[T-1][S-2] = 1.0
        beta[T-1][S-1] = 1.0
            
        # MYNOTE: dynamic filling beta block, from back to front
        for t in range(T-2, -1, -1):
            for s in range(S):
                # 当前位置的符号
                symbol = extended_symbols[s]
                
                # 转移到下一个位置
                if s < S-1:
                    next_symbol = extended_symbols[s+1]
                    beta[t, s] += beta[t+1, s+1] * logits[t+1, next_symbol]
                
                # 保持在当前位置
                beta[t, s] += beta[t+1, s] * logits[t+1, symbol]
                
                # 跳过连接（如果允许）
                if s < S-2 and skip_connect[s+2]:
                    next_next_symbol = extended_symbols[s+2]
                    beta[t, s] += beta[t+1, s+2] * logits[t+1, next_next_symbol]

        # <--------------------------------------------
        # print(beta.shape)
        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        T, S = alpha.shape
        
        # 初始化gamma矩阵
        gamma = np.zeros((T, S))
        
        # 对每个时间步t计算后验概率
        for t in range(T):
            # 计算归一化因子
            normalization = np.sum(alpha[t, :] * beta[t, :])
            
            # 计算每个位置的后验概率
            for s in range(S):
                if normalization > 0:
                    gamma[t, s] = (alpha[t, s] * beta[t, s]) / normalization
                else:
                    gamma[t, s] = 0
        # <---------------------------------------------

        return gamma


class CTCLoss(object):

    def __init__(self, autograd_engine, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()
        self.autograd_engine = autograd_engine

        self.BLANK = BLANK
        self.ctc = CTC()

        # NOTE: Toggle using ctc_loss_backward version
        # or a version using more primitive operations
        self.USE_PRIMITIVE = True
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward
                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        self.gammas=[]
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # 1. 获取当前批次的目标序列和长度
            curr_target = target[b, :target_lengths[b]]
            curr_input_length = input_lengths[b]
            
            # 2. 获取当前批次的logits
            curr_logits = logits[:curr_input_length, b, :]
            
            # 3. 扩展目标序列，添加空白符号
            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(curr_target)
            self.extended_symbols.append(ext_symbols)
            
            # 4. 计算前向概率
            alpha = self.ctc.get_forward_probs(curr_logits, ext_symbols, skip_connect)
            
            # 5. 计算后向概率
            beta = self.ctc.get_backward_probs(curr_logits, ext_symbols, skip_connect)
            
            # 6. 计算后验概率 input using alpha x beta, get gamma, normalized along column
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            for r in range(gamma.shape[1]):
                total_loss[b] -= np.sum(gamma[0:, r] * np.log(curr_logits[:, ext_symbols[r]]))

            self.gammas.append(gamma)
            
            # # 7. 计算当前批次的损失
            # # CTC损失本质上是负对数似然
            # # 对于每个时间步t，计算目标标签的概率之和
            # T, S = alpha.shape
            
            # # 计算所有路径的概率总和（通常是alpha的最后一行最后两个元素之和）
            # # 这代表了所有可能产生目标序列的路径的总概率
            # prob = alpha[T-1, S-1] + (S > 1 and alpha[T-1, S-2] or 0)
            
            # # Negative Log-Likelihood
            # if prob > 0:
            #     total_loss[batch_itr] = -np.log(prob)
            # else:
            #     # model complete impossible to get correct ans, loss to be impossible huge
            #     total_loss[batch_itr] = float('inf')
                    
            # <---------------------------------------------

            """
            MYNOTE:
            The equation:
            L_CTC(x, y) = -log Σ_{π∈B⁻¹(y)} P(π|x)
            
            so in 
            # Negative Log-Likelihood
            if prob > 0:
                total_loss[batch_itr] = -np.log(prob)
            else:
                # model complete impossible to get correct ans, loss to be impossible huge
                total_loss[batch_itr] = float('inf')
                        
            """
        # total_loss = np.sum(total_loss) / B
        total_loss = np.mean(total_loss)
        
        # Convert lists to appropriate numpy format before passing to add_operation
        # This is crucial to avoid the 'list' object has no attribute '__array_interface__' error
        np_gammas = np.array(self.gammas, dtype=object)
        np_ext_symbols = np.array(self.extended_symbols, dtype=object)
        
        # TODO: You must implement ctc_loss_backward
        self.autograd_engine.add_operation(
            inputs=[self.logits, self.input_lengths, self.gammas, self.extended_symbols],
            output=total_loss,
            gradients_to_update=[None, None, None, None],
            backward_operation=ctc_loss_backward,
        )
        return total_loss
