import numpy as np

# Input
X = np.array([1.0, 2.0])

# Weights Input -> Hidden
W1 = np.array([
    [0.5, 0.2],
    [0.3, 0.8]
])

# Bias Hidden
B1 = np.array([0.1, 0.1])

# Weights Hidden -> Output
W2 = np.array([
    [0.7],
    [0.6]
])

# Bias Output
B2 = np.array([0.05])

# Activation Function
def relu(x):
    return np.maximum(0, x)

# Forward Pass
hidden_input = np.dot(X, W1) + B1
hidden_output = relu(hidden_input)

final_input = np.dot(hidden_output, W2) + B2
final_output = final_input

print("Hidden Layer Output:", hidden_output)
print("Final Output:", final_output)
