
import numpy as np

class ScratchKNN:
    def __init__(self, k=5, weights='uniform'):
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        # Lazy Learning
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        
        for x_query in X_test:
            # 1. Distances
            dists = np.sqrt(np.sum((self.X_train - x_query)**2, axis=1))
            
            # 2. Get Top K
            k_indices = np.argsort(dists)[:self.k]
            k_dists = dists[k_indices]
            k_labels = self.y_train[k_indices]
            
            # 3. Aggregate
            if self.weights == 'uniform':
                pred = np.mean(k_labels)
            elif self.weights == 'distance':
                # Avoid division by zero
                k_dists = np.maximum(k_dists, 1e-10) 
                w = 1 / k_dists
                pred = np.sum(w * k_labels) / np.sum(w)
            
            predictions.append(pred)
            
        return np.array(predictions)

class ScratchNadarayaWatson:
    def __init__(self, h=1.0):
        self.h = h
        self.X_train = None
        self.y_train = None
        
    def kernel(self, d):
        return np.exp(-d**2 / (2 * self.h**2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        preds = []
        for xq in X_test:
            dists = np.sqrt(np.sum((self.X_train - xq)**2, axis=1))
            w = self.kernel(dists)
            if np.sum(w) < 1e-10: preds.append(np.mean(self.y_train))
            else: preds.append(np.sum(w * self.y_train) / np.sum(w))
        return np.array(preds)

class ScratchLocalLinear:
    def __init__(self, h=1.0):
        self.h = h
        self.X_train = None
        self.y_train = None
        
    def kernel(self, d):
        return np.exp(-d**2 / (2 * self.h**2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        preds = []
        # Add bias term to training data for regression
        X_b = np.c_[np.ones(len(self.X_train)), self.X_train]
        
        for xq in X_test:
            # 1. Weights based on distance to query point
            dists = np.sqrt(np.sum((self.X_train - xq)**2, axis=1))
            weights = self.kernel(dists)
            W = np.diag(weights)
            
            # 2. Weighted Linear Regression: (X^T W X)^-1 X^T W y
            # Optimization: We only need to solve it, not invert fully
            # We fit locally, relative to xq to stabilize numerics
            try:
                beta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ self.y_train)
                # Prediction is intercept + slope * query
                pred = beta[0] + beta[1:] @ xq
            except:
                pred = np.mean(self.y_train) # Fallback
            
            preds.append(pred)
        return np.array(preds)
