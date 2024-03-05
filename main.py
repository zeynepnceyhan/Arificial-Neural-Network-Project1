#Code1

import pandas as pd

df = pd.read_excel(r"C:\Users\Zeynep\Desktop\ANN\Doviz_Satislari_20050101_20231205_Training_Set.xlsx")
expl = df.loc[0:7]
name_col = df.loc[11].tolist()
df.columns = name_col
df = df.loc[12:]
df.set_index("No", inplace=True)

inp = int(input("Enter i.th record"))
b= expl.iloc[:,1].tolist()
a = df.loc[inp]
a = pd.DataFrame(a)
b.insert(0,"Tarih")
a = a.T
a.columns = b
print(a)
df.drop('Tarih', axis=1, inplace=True)


#Code2

import numpy as np

# Cost/Loss Function
def cost_function(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)
    return J

# Derivative of Cost/Loss Function
def cost_derivative(X, y, theta):
    m = len(y)
    error = X @ theta - y
    derivative = (1 / m) * (X.T @ error)
    return derivative

# Optimizer using Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    J_history = []

    for _ in range(num_iterations):
        theta = theta - alpha * cost_derivative(X, y, theta)
        J_history.append(cost_function(X, y, theta))

    return theta, J_history

# Model Code
def train_model(X, y, alpha, num_iterations):
    # Add a column of ones to X for the bias term
    X_with_bias = np.c_[np.ones(X.shape[0]), X]

    # Initialize parameters
    theta = np.zeros((X_with_bias.shape[1], 1))

    # Train the model using gradient descent
    theta, J_history = gradient_descent(X_with_bias, y, theta, alpha, num_iterations)

    return theta, J_history

# Example Usage:
# X, y: Input features and output target
# alpha: Learning rate
# num_iterations: Number of iterations for gradient descent

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # Assuming 3 input features
theta_true = np.array([[5], [3], [2]])  # True coefficients for demonstration
y = X @ theta_true + np.random.randn(100, 1)  # Simulated output with noise

# Train the model
theta_learned, cost_history = train_model(X, y, alpha=0.01, num_iterations=1000)

# Print the learned parameters
print("Learned Parameters:")
print(theta_learned)

# Plot the cost history
import matplotlib.pyplot as plt
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

#Code3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.iloc[:, :7]
y = df.iloc[:, 7]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import numpy as np

import numpy as np
import math

def gradient_descent_with_regularization(X, y, theta, learning_rate, num_iterations, alpha_l1, alpha_l2):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        predictions = X @ theta
        errors = predictions - y.values.reshape(-1, 1)  # Convert y to a 2D NumPy array

        # Calculate gradients with regularization
        gradients = (1 / m) * X.T @ errors + (alpha_l2 / m) * theta
        gradients[0] -= alpha_l2 / m * theta[0]  # Exclude regularization for bias term

        # Check for NaN values in gradients using math.isnan
        if any(math.isnan(val) for val in gradients):
            print("NaN values detected in gradients. Stopping gradient descent.")
            break

        # Update parameters
        theta = theta - learning_rate * gradients

        # Calculate cost with regularization
        cost = (1 / (2 * m)) * np.sum(errors**2) + (alpha_l2 / (2 * m)) * np.sum(theta[1:]**2) + (alpha_l1 / m) * np.sum(np.abs(theta[1:]))
        cost_history.append(cost)

    return theta, cost_history



X_train_with_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]

# Initialize parameters
theta_initial = np.zeros((X_train_with_bias.shape[1], 1))  # +1 for bias term

# Train the model with L1 and L2 regularization
theta_learned, cost_history = gradient_descent_with_regularization(X_train_with_bias, y_train, theta_initial, learning_rate=0.01, num_iterations=1000, alpha_l1=0.01, alpha_l2=0.01)

# Print the learned parameters
print("Learned Parameters:")
print(theta_learned)

# Plot the cost history
import matplotlib.pyplot as plt
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


# Code TEST Phase

# Add a column of ones to X_test_scaled for the bias term
X_test_with_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Use the trained model to make predictions
predictions = X_test_with_bias @ theta_learned

# Print the predicted values
print("Predicted Values:")
print(predictions)

# You can compare these predictions with the actual y_test values for evaluation
# For example, you can calculate the mean squared error
mse = np.mean((predictions - y_test.values.reshape(-1, 1))**2)
print("Mean Squared Error:", mse)
