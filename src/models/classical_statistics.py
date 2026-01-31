import numpy as np

class ScratchOLS:
    def __init__(self):
        self.coeffs = None

    def fit(self, X, y):
        # 1. Add Bias Term (Intercept)
        # Creates a matrix [1, x_1], [1, x_2] ...
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])

        # 2. The Normal Equation: (X^T * X)^(-1) * X^T * y
        # np.linalg.inv computes the multiplicative inverse of a matrix
        self.coeffs = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
    def predict(self, X):
        # 1. Add Bias Term
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])
        
        # 2. Dot Product: y = X * beta
        return X_b @ self.coeffs

class ScratchPoissonGLM:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # 1. Linear Model (log(mu) = wx + b)
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 2. Link Function (Inverse Log = Exp)
            # y_pred is lambda (the rate/mean)
            y_pred = np.exp(linear_model)

            # 3. Compute Gradient (Derivative of Poisson Log-Likelihood)
            # dL/dw = X * (y_pred - y)
            # Note: This looks identical to OLS gradient but y_pred is exponential!
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 4. Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.exp(linear_model)

class ScratchPLS:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.x_loadings_ = []
        self.x_weights_ = []

    def fit(self, X, y):
        # Center data (Crucial for PLS)
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y)
        X_k = X - self.X_mean
        y_k = y - self.y_mean

        for _ in range(self.n_components):
            # 1. Calculate weights w = X^T * y
            # This finds direction of Max Covariance
            w = np.dot(X_k.T, y_k)
            w /= np.linalg.norm(w) # Normalize
            self.x_weights_.append(w)

            # 2. Calculate scores t = X * w
            t = np.dot(X_k, w)

            # 3. Calculate loadings p = X^T * t / t^T * t
            # How much of X is explained by t?
            div = np.dot(t.T, t)
            if div == 0: div = 1e-10
            p = np.dot(X_k.T, t) / div
            self.x_loadings_.append(p)

            # 4. Deflate X (Remove info explained by this component)
            # X_new = X_old - t * p^T
            # We remove the part of X that projects onto 't'
            X_k -= np.outer(t, p)

    def transform(self, X):
        # To transform new data, we must mimic the deflation process
        X_k = X - self.X_mean
        scores = []
        for w, p in zip(self.x_weights_, self.x_loadings_):
            # Project onto weight
            t = np.dot(X_k, w)
            scores.append(t)
            
            # Deflate X using the loading p
            # (Subtracting the component we just found from the data)
            X_k -= np.outer(t, p)
            
        return np.column_stack(scores)
