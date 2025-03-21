import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        blank = 0
        path_prob = 1.0
        
        # y probs, an array of shape (len(SymbolSets) + 1, seq length, batch size)
        num_symbols = y_probs.shape[0]
        seq_length = y_probs.shape[1]
        batch_size = y_probs.shape[2]
        
        batch_idx = 0
        
        # Raw path with indices (before compression)
        raw_path = []
        
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0]) ?? but sequence length - len(y_probs[0] will be 0
        for i in range(seq_length):
            # 2. Iterate over symbol probabilities
            symbol_probs = y_probs[:, i, batch_idx]
            # 3. update path probability, by multiplying with the current max probability
            # Find the most probable symbol, and select it as max_idx
            max_idx = np.argmax(symbol_probs)
            curr_max_prob = symbol_probs[max_idx]
            
            path_prob *= curr_max_prob
                
            # 4. Select most probable symbol and append to decoded_path
            raw_path.append(max_idx)
        # 5. Compress sequence (Inside or outside the loop)
        decoded_path = []
        prev_idx = -1  # Initialize with a value that won't match any symbol index
        
        # ??
        for idx in raw_path:
            # Skip blanks
            if idx == blank:
                continue
            
            # Skip repeats (only add if different from previous)
            if idx != prev_idx:
                # Convert index to symbol (subtract 1 because index 0 is blank)
                symbol = self.symbol_set[idx - 1]
                decoded_path.append(symbol)
            
            # Update previous index
            prev_idx = idx
        
        # Join the symbols into a string
        decoded_str = ''.join(decoded_path)
        
        return decoded_str, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables

        Argument(s)
        -----------
        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)
        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion
        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        Perform beam search decoding

        Input
        -----
        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                batch size for part 1 will remain 1

        Returns
        -------
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)
        merged_path_scores [dict]:
            all the final merged paths with their scores
        """
        # Create BeamSearch class instance and run it
        bs = BeamSearchClass(self.symbol_set, y_probs, self.beam_width)
        return bs()


class BeamSearchClass:
    def __init__(self, SymbolSets, y_probs, BeamWidth):
        self.symbols = SymbolSets
        self.y_probs = y_probs
        self.k = BeamWidth

        # Initialize with empty path (blank)
        self.paths_blank = ['']
        self.paths_blank_score = {'': y_probs[0, 0, 0]}

        # Initialize with single symbol paths
        self.paths_symbol = [c for c in self.symbols]
        self.paths_symbol_score = {}
        for i, c in enumerate(SymbolSets):
            self.paths_symbol_score[c] = y_probs[i + 1, 0, 0]

    def __call__(self):
        # Process each time step
        for t in range(1, self.y_probs.shape[1]):
            # Prune to beam width
            self.prune()
            
            # Extend with symbols and blanks
            updated_paths_symbol, updated_paths_symbol_score = self.extend_with_symbol(t)
            updated_paths_blank, updated_paths_blank_score = self.extend_with_blank(t)
            
            # Update paths for next iteration
            self.paths_blank = updated_paths_blank
            self.paths_symbol = updated_paths_symbol
            self.paths_blank_score = updated_paths_blank_score
            self.paths_symbol_score = updated_paths_symbol_score

        # Merge final paths and return best path
        return self.merge()

    def extend_with_symbol(self, t):
        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        # Extend paths ending with blank by adding a symbol
        for path in self.paths_blank:
            for i, c in enumerate(self.symbols):
                new_path = path + c
                updated_paths_symbol.append(new_path)
                updated_paths_symbol_score[new_path] = self.paths_blank_score[path] * self.y_probs[i + 1, t, 0]

        # Extend paths ending with symbol
        for path in self.paths_symbol:
            for i, c in enumerate(self.symbols):
                # Handle CTC merge rule for same symbol
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_symbol_score:
                    updated_paths_symbol_score[new_path] += self.paths_symbol_score[path] * self.y_probs[i + 1, t, 0]
                else:
                    updated_paths_symbol_score[new_path] = self.paths_symbol_score[path] * self.y_probs[i + 1, t, 0]
                    updated_paths_symbol.append(new_path)

        return updated_paths_symbol, updated_paths_symbol_score

    def extend_with_blank(self, t):
        updated_paths_blank = []
        updated_paths_blank_score = {}

        # Extend paths ending with blank by adding another blank
        for path in self.paths_blank:
            updated_paths_blank.append(path)
            updated_paths_blank_score[path] = self.paths_blank_score[path] * self.y_probs[0, t, 0]

        # Extend paths ending with symbol by adding a blank
        for path in self.paths_symbol:
            if path in updated_paths_blank:
                updated_paths_blank_score[path] += self.paths_symbol_score[path] * self.y_probs[0, t, 0]
            else:
                updated_paths_blank_score[path] = self.paths_symbol_score[path] * self.y_probs[0, t, 0]
                updated_paths_blank.append(path)

        return updated_paths_blank, updated_paths_blank_score

    def prune(self):
        updated_paths_blank = []
        updated_paths_blank_score = {}
        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        # Collect all scores for pruning
        scores = []
        for score in self.paths_blank_score.values():
            scores.append(score)
        for score in self.paths_symbol_score.values():
            scores.append(score)

        # Sort scores and determine cutoff
        scores.sort(reverse=True)
        cutoff = scores[min(self.k - 1, len(scores) - 1)] if scores else 0

        # Keep only paths with scores >= cutoff
        for path in self.paths_blank:
            if self.paths_blank_score[path] >= cutoff:
                updated_paths_blank.append(path)
                updated_paths_blank_score[path] = self.paths_blank_score[path]

        for path in self.paths_symbol:
            if self.paths_symbol_score[path] >= cutoff:
                updated_paths_symbol.append(path)
                updated_paths_symbol_score[path] = self.paths_symbol_score[path]

        # Update paths and scores
        self.paths_symbol_score = updated_paths_symbol_score
        self.paths_symbol = updated_paths_symbol
        self.paths_blank_score = updated_paths_blank_score
        self.paths_blank = updated_paths_blank

    def merge(self):
        paths = self.paths_blank.copy()
        scores = dict(self.paths_blank_score)

        # Merge paths ending with symbol
        for path in self.paths_symbol:
            if path in paths:
                scores[path] += self.paths_symbol_score[path]
            else:
                paths.append(path)
                scores[path] = self.paths_symbol_score[path]

        # Find path with highest score
        max_path = ""
        max_score = float('-inf')
        for path in scores:
            if scores[path] > max_score:
                max_path = path
                max_score = scores[path]

        return max_path, scores