import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Linear combination of features and weights
            z = np.dot(X, self.weights) + self.bias

            # Apply sigmoid function to get probabilities
            y_pred = self.sigmoid(z)

            # Calculate gradients
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Predict class labels (0 or 1)
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 3)
    y = (2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] > 0).astype(int)

    # Split the data into training and test sets (80% training, 20% test)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize and fit the logistic regression model
    lr = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate and print the accuracy as a measure of model performance
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)
