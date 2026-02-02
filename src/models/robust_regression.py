import numpy as np

class ScratchRANSAC:
    def __init__(self, n_iterations=100, threshold=5.0, min_samples=2):
        self.n_iters = n_iterations
        self.threshold = threshold # Epsilon: The definition of "Agreeing"
        self.min_samples = min_samples
        self.best_model = None
        self.best_inliers_count = 0
        self.best_inlier_mask = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        
        for _ in range(self.n_iters):
            # 1. Random Sample
            ids = np.random.choice(n_samples, self.min_samples, replace=False)
            X_subset = X_b[ids]
            y_subset = y[ids]
            
            # 2. Fit Model (OLS on subset)
            try:
                # Use psuedoinverse for stability with small samples
                w = np.linalg.pinv(X_subset.T @ X_subset) @ X_subset.T @ y_subset
            except (np.linalg.LinAlgError, ValueError):
                continue
                
            # 3. Test on All Data
            y_pred = X_b @ w
            residuals = np.abs(y - y_pred)
            inliers = residuals < self.threshold
            inlier_count = np.sum(inliers)
            
            # 4. Keep Best
            if inlier_count > self.best_inliers_count:
                self.best_inliers_count = inlier_count
                self.best_model = w
                self.best_inlier_mask = inliers
                
        # Optional: Refit on ALL inliers of the best model for better precision (Polishing step)
        if self.best_inlier_mask is not None:
            X_final = X_b[self.best_inlier_mask]
            y_final = y[self.best_inlier_mask]
            self.best_model = np.linalg.pinv(X_final.T @ X_final) @ X_final.T @ y_final

    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.best_model

class ScratchTheilSen:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        slopes = []
        
        # Brute force all pairs (O(N^2))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if X[j] != X[i]:
                    slope = (y[j] - y[i]) / (X[j] - X[i])
                    slopes.append(slope)
        
        self.slope = np.median(slopes)
        self.intercept = np.median(y - self.slope * X.ravel())

    def predict(self, X):
        return self.slope * X + self.intercept

class ScratchHuber:
    def __init__(self, delta=1.35, learning_rate=0.01, n_iters=1000):
        self.delta = delta
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # 1. Prediction
            y_pred = X @ self.weights + self.bias
            residuals = y - y_pred
            
            # 2. Compute Gradients
            # Detailed Derivative of Huber Loss:
            # If |r| <= delta: grad = -r
            # If |r| > delta:  grad = -delta * sign(r)
            
            mask_small = np.abs(residuals) <= self.delta
            mask_large = ~mask_small
            
            weight_grad = np.zeros(n_features)
            bias_grad = 0
            
            # Case 1: Small errors (MSE derivative)
            # d/dw = -sum(x * r)
            if np.any(mask_small):
                weight_grad += -X[mask_small].T @ residuals[mask_small]
                bias_grad += -np.sum(residuals[mask_small])
                
            # Case 2: Large errors (MAE derivative)
            # d/dw = -sum(x * delta * sign(r))
            if np.any(mask_large):
                weight_grad += -X[mask_large].T @ (self.delta * np.sign(residuals[mask_large]))
                bias_grad += -np.sum(self.delta * np.sign(residuals[mask_large]))
            
            # 3. Update
            self.weights -= self.lr * (weight_grad / n_samples)
            self.bias -= self.lr * (bias_grad / n_samples)

    def predict(self, X):
        return X @ self.weights + self.bias
