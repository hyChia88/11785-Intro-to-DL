import numpy as np

# Example logits
logits = np.array([3, 10, 9])

# Softmax
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

probs = softmax(logits)
print("Probabilities:", probs)

# Cross-entropy with one-hot true labels
true_class = np.array([0, 0, 1])  # Class 3 is true
cross_entropy = -np.sum(true_class * np.log(probs))
print("Cross-entropy loss:", cross_entropy)

# Conditional probability example
# P(class=3 | input) = probs[2]
print("P(class=3 | input) =", probs[2])