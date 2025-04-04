# 11785 HW3P2 Submission
huiyenc, Chia Hui Yen, March 10, 2025

## Links
- Weights & Biases Project: https://wandb.ai/hychia2024-carnegie-mellon-university/hw3p2?nw=nwuserhychia2024

---

## Development Log

### Attempt 1: Basic Implementation
- Completed a basic ASR model architecture using CTC encoder-decoder framework with pBLSTM
- Implemented `pad_sequence` to handle variable-length sequences
- Stored original sequence length information in `collate_fn` for proper packing/unpacking

### Attempt 2: Architecture Improvements
- **Configuration**:
  ```yaml
  subset: 1.0
  learning_rate: 0.01
  epochs: 10
  train_beam_width: 5
  test_beam_width: 10
  mfcc_features: 28
  embed_size: 256
  batch_size: 256
  encoder_dropout: 0.5
  lstm_dropout: 0.5
  decoder_dropout: 0.5
  ```

- **Data Augmentation**:
  - Added TimeMasking and FrequencyMasking to improve robustness:
    ```python
    time_mask = TimeMasking(time_mask_param=10)
    freq_mask = FrequencyMasking(freq_mask_param=5)
    ```

- **Model Enhancements**:
  - Enhanced the encoder with ResNet structures, CNNs, and multiple pBLSTM layers (1-3 layers)
  - Increased hidden size dimensions for better representation capacity
  - Implemented LockedDropout instead of standard dropout for improved RNN training stability
  - Modified handling of odd-length time steps (experimental with padding instead of truncation to preserve information)

### Attempt 3: Architecture Refinement
- Fixed parameter passing errors between encoder and decoder components
- Corrected model architecture issues to ensure proper information flow

### Attempt 4: Bugfixes and Optimization
- Fixed critical bug in Levenshtein distance calculation:
  ```python
  # Incorrect:
  label_indices = label[i][:label_lens[i]].cpu().numpy()
  
  # Corrected:
  label_indices = label[i, :label_lens[i]].cpu().numpy()
  ```
- Resolved tokenization issues during training
- Results: Significantly reduced noise in training process and stabilized model convergence