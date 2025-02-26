import math

def f(x, y, i):
    if i >= 100:
        return f"x: {round(x,1)}, y: {round(y,1)}"
    
    # Calculate partial derivatives for z = 20 + x^2 + y^2 - 10cos(2πx) - 10cos(2πy)
    dx = 2*x + 20*math.pi*math.sin(2*math.pi*x)  
    dy = 2*y + 20*math.pi*math.sin(2*math.pi*y)
    
    # Move in direction of steepest ascent (positive gradient)
    x_new = x + 0.001 * dx
    y_new = y + 0.001 * dy
    
    return f(x_new, y_new, i+1)

print(f(1.7, -1.2, 0))

# import numpy as np
# import matplotlib.pyplot as plt

# # Input embedding
# X = np.array([1, 2, 3, 4])

# # Weight matrix
# W = np.array([
#     [1, 0, 1],
#     [1, 1, 1],
#     [0, 1, 1],
#     [0, 1, 1]
# ])

# # Calculate logits (Z)
# Z = np.dot(X, W)
# print("Logits Z:", Z)

# # Apply softmax
# exp_Z = np.exp(Z)
# softmax = exp_Z / np.sum(exp_Z)

# for i, v in enumerate(softmax):
#     print(i, v + 0.01, f'{v:.2f}')

