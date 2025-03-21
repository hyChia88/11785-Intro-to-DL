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
        path_prob = 1

        '''
        1. Iterate over sequence length - len(y_probs[0])
        2. Iterate over symbol probabilities
        3. update path probability, by multiplying with the current max probability
        4. Select most probable symbol and append to decoded_path
        5. Compress sequence (Inside or outside the loop)
        '''
        
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
        # Initialize with empty path (blank) and single symbol paths
        paths_blank = ['']
        paths_blank_score = {'': y_probs[0, 0, 0]}
        
        paths_symbol = [c for c in self.symbol_set]
        paths_symbol_score = {}
        for i, c in enumerate(self.symbol_set):
            paths_symbol_score[c] = y_probs[i + 1, 0, 0]

        # Process each time step
        for t in range(1, y_probs.shape[1]):
            # Prune to beam width
            paths_blank, paths_symbol, paths_blank_score, paths_symbol_score = self._prune(
                paths_blank, paths_symbol, paths_blank_score, paths_symbol_score
            )
            
            # Extend paths
            updated_paths_symbol, updated_paths_symbol_score = self._extend_with_symbol(
                paths_blank, paths_symbol, paths_blank_score, paths_symbol_score, y_probs[:, t, 0]
            )
            
            updated_paths_blank, updated_paths_blank_score = self._extend_with_blank(
                paths_blank, paths_symbol, paths_blank_score, paths_symbol_score, y_probs[:, t, 0]
            )
            
            # Update paths for next iteration
            paths_blank, paths_symbol = updated_paths_blank, updated_paths_symbol
            paths_blank_score, paths_symbol_score = updated_paths_blank_score, updated_paths_symbol_score

        # Merge final paths
        merged_paths, merged_scores = self._merge(paths_blank, paths_symbol, paths_blank_score, paths_symbol_score)
        
        # Find best path
        best_path = ""
        best_score = float('-inf')
        for path, score in merged_scores.items():
            if score > best_score:
                best_path = path
                best_score = score
                
        return best_path, merged_scores

    def _extend_with_symbol(self, paths_blank, paths_symbol, paths_blank_score, paths_symbol_score, y_prob_t):
        """Extend paths by adding a symbol"""
        updated_paths_symbol = []
        updated_paths_symbol_score = {}

        # Extend paths ending with blank by adding a symbol
        for path in paths_blank:
            for i, c in enumerate(self.symbol_set):
                new_path = path + c
                updated_paths_symbol.append(new_path)
                updated_paths_symbol_score[new_path] = paths_blank_score[path] * y_prob_t[i + 1]

        # Extend paths ending with symbol
        for path in paths_symbol:
            for i, c in enumerate(self.symbol_set):
                # Handle CTC merge rule for same symbol
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_symbol_score:
                    updated_paths_symbol_score[new_path] += paths_symbol_score[path] * y_prob_t[i + 1]
                else:
                    updated_paths_symbol_score[new_path] = paths_symbol_score[path] * y_prob_t[i + 1]
                    updated_paths_symbol.append(new_path)

        return updated_paths_symbol, updated_paths_symbol_score

    def _extend_with_blank(self, paths_blank, paths_symbol, paths_blank_score, paths_symbol_score, y_prob_t):
        """Extend paths by adding a blank"""
        updated_paths_blank = []
        updated_paths_blank_score = {}

        # Extend paths ending with blank by adding another blank
        for path in paths_blank:
            updated_paths_blank.append(path)
            updated_paths_blank_score[path] = paths_blank_score[path] * y_prob_t[0]

        # Extend paths ending with symbol by adding a blank
        for path in paths_symbol:
            if path in updated_paths_blank:
                updated_paths_blank_score[path] += paths_symbol_score[path] * y_prob_t[0]
            else:
                updated_paths_blank_score[path] = paths_symbol_score[path] * y_prob_t[0]
                updated_paths_blank.append(path)

        return updated_paths_blank, updated_paths_blank_score

    def _prune(self, paths_blank, paths_symbol, paths_blank_score, paths_symbol_score):
        """Prune paths to keep only beam_width highest probability paths"""
        # Collect all scores
        scores = []
        for score in paths_blank_score.values():
            scores.append(score)
        for score in paths_symbol_score.values():
            scores.append(score)

        # Sort scores and determine cutoff
        scores.sort(reverse=True)
        cutoff = scores[min(self.beam_width - 1, len(scores) - 1)] if scores else 0

        # Keep only paths with scores >= cutoff
        pruned_paths_blank = []
        pruned_paths_blank_score = {}
        for path in paths_blank:
            if paths_blank_score[path] >= cutoff:
                pruned_paths_blank.append(path)
                pruned_paths_blank_score[path] = paths_blank_score[path]

        pruned_paths_symbol = []
        pruned_paths_symbol_score = {}
        for path in paths_symbol:
            if paths_symbol_score[path] >= cutoff:
                pruned_paths_symbol.append(path)
                pruned_paths_symbol_score[path] = paths_symbol_score[path]

        return pruned_paths_blank, pruned_paths_symbol, pruned_paths_blank_score, pruned_paths_symbol_score

    def _merge(self, paths_blank, paths_symbol, paths_blank_score, paths_symbol_score):
        """Merge identical paths that differ only by their final blank"""
        merged_paths = []
        merged_scores = {}

        # Start with paths ending with blank
        for path in paths_blank:
            merged_paths.append(path)
            merged_scores[path] = paths_blank_score[path]

        # Merge with paths ending with symbol
        for path in paths_symbol:
            if path in merged_paths:
                merged_scores[path] += paths_symbol_score[path]
            else:
                merged_paths.append(path)
                merged_scores[path] = paths_symbol_score[path]

        return merged_paths, merged_scores