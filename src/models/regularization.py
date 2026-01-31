import numpy as np

class ScratchRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coeffs = None
        self.intercept = None

    def fit(self, X, y):
        # 1. Center the Target (y)
        # Ridge penalizes SLOPES, not INTERCEPTS. 
        # By centering y, we remove the need to solve for the intercept in the matrix.
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        n_features = X.shape[1]
        
        # 2. Create Identity Matrix (The Penalty)
        I = np.eye(n_features)
        
        # 3. The Ridge Equation
        # (X^T * X + alpha * I)^-1 * X^T * y
        X_T_X = X.T @ X
        Penalty = self.alpha * I  # The Regularization Term
        
        # We invert the "Penalized Covariance Matrix"
        self.coeffs = np.linalg.inv(X_T_X + Penalty) @ X.T @ y_centered
        self.intercept = self.y_mean

    def predict(self, X):
        return (X @ self.coeffs) + self.intercept

    @property
    def coef_(self):
        return self.coeffs

class ScratchLasso:
    def __init__(self, alpha=0.1, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.coeffs = None
        self.intercept = None

    def _soft_threshold(self, rho, lam):
        """
        The heart of Lasso.
        If the correlation (rho) is smaller than the penalty (lam), snap it to 0.
        Otherwise, shrink it by 'lam'.
        """
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0 # <--- This is where Feature Selection happens!

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)
        self.intercept = np.mean(y) # Start with simple mean
        
        # Coordinate Descent Loop
        for _ in range(self.n_iters):
            for j in range(n_features):
                # 1. Calculate the residual (error) assuming this feature j didn't exist
                y_pred_others = (X @ self.coeffs) + self.intercept - (X[:, j] * self.coeffs[j])
                
                # 2. Calculate Correlation (rho) between feature j and the residual
                # "How much does Feature J help fix the error?"
                rho = X[:, j] @ (y - y_pred_others)
                
                # 3. Soft Thresholding
                # We compare the helpfulness (rho) vs the cost (alpha)
                # Note: We scale alpha by n_samples to conform to standard loss definition
                lam = self.alpha * n_samples 
                
                # Normalization factor (usually 1 if standardized, but good to be safe)
                z_j = X[:, j] @ X[:, j]
                
                self.coeffs[j] = self._soft_threshold(rho, lam) / z_j
            
            # Update intercept at the end of each cycle
            self.intercept = np.mean(y - X @ self.coeffs)

    def predict(self, X):
        return (X @ self.coeffs) + self.intercept
    
    @property
    def coef_(self):
        return self.coeffs
