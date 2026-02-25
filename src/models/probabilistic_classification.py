import numpy as np

class ScratchLogisticRegression:
    """Multinomial Logistic Regression using Softmax and Gradient Descent."""
    
    def __init__(self, learning_rate=0.1, n_iterations=3000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        
    def _softmax(self, z):
        # Subtracting the max value prevents np.exp() from overflowing
        # mathematically identical, numerically safe
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def _one_hot(self, y):
        one_hot = np.zeros((y.size, self.n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        
        # Initialize weights and biases to zero
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        y_encoded = self._one_hot(y)
        
        for i in range(self.n_iterations):
            # Forward pass: Compute logits and probabilities
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(linear_model)
            
            # Compute Cross-Entropy Loss
            loss = -np.mean(np.sum(y_encoded * np.log(y_pred + 1e-15), axis=1))
            self.losses.append(loss)
            
            # Backward pass: Compute gradients
            # The derivative of Cross-Entropy with Softmax is elegantly: (y_pred - y_true)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1 / n_samples) * np.sum(y_pred - y_encoded, axis=0)
            
            # Gradient descent update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class ScratchGaussianNB:
    """Gaussian Naive Bayes classification."""
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # We will store the mean, variance, and prior probability for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            # Select all rows belonging to class c
            X_c = X[y == c]
            
            # Calculate statistics across the rows (axis=0)
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            
    def _pdf(self, class_idx, x):
        # Gaussian Probability Density Function
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
        
    def _predict_single(self, x):
        posteriors = []
        
        for idx in range(len(self.classes)):
            # Start with the log prior
            prior = np.log(self.priors[idx])
            
            # Sum the log likelihoods of the features given the class
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # The class with the highest posterior probability wins
        return self.classes[np.argmax(posteriors)]
        
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
