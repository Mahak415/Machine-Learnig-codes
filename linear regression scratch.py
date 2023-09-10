import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Predictions using current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.rand(100, 1)

    # Split the data into training and test sets (80% training, 20% test)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize and fit the linear regression model
    lr = LinearRegression(learning_rate=0.01, num_iterations=1000)
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate and print the Mean Squared Error (MSE) as a measure of model performance
    mse = np.mean((y_pred - y_test) ** 2)
    print("Mean Squared Error:", mse)