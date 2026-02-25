import numpy as np

class ScratchLinearSVM:
    """Linear Support Vector Machine using Subgradient Descent (Hinge Loss)."""
    
    def __init__(self, learning_rate=0.001, C=1.0, n_iterations=3000):
        self.lr = learning_rate
        self.C = C
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        # Convert labels 0, 1 to -1, 1 for Hinge Loss math
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Subgradient descent
        for _ in range(self.n_iterations):
            # Calculate the decision function (margin distance)
            margins = y_ * (np.dot(X, self.w) - self.b)
            
            # Identify which points violate the margin (margin < 1)
            # These are our "Support Vectors" currently driving the gradient!
            violating_mask = margins < 1
            
            # Gradient of the regularization term (Margin maximizer)
            dw = self.w.copy()
            db = 0
            
            # Add the gradient of the Hinge Loss for violating points
            if np.any(violating_mask):
                dw -= self.C * np.dot(X[violating_mask].T, y_[violating_mask])
                db += self.C * np.sum(y_[violating_mask])
                
            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict(self, X):
        # Predict sign: > 0 is (+1) Benign, < 0 is (-1) Malignant
        approx = np.dot(X, self.w) - self.b
        return np.where(np.sign(approx) == -1, 0, 1)

class ScratchLDA:
    """Linear Discriminant Analysis from scratch using Eigenvectors."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.linear_discriminants = None
        self.class_means = {}
        
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # Calculate the overall mean of the entire dataset
        mean_overall = np.mean(X, axis=0)
        
        # 1. Initialize Scatter Matrices (30x30 matrices for our 30 features)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.class_means[c] = mean_c
            
            # Within-class scatter (Covariance multiplied by number of samples in class)
            # S_W = sum((X_c - mean_c) * (X_c - mean_c)^T)
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
            
            # Between-class scatter
            # S_B = sum(n_c * (mean_c - mean_overall) * (mean_c - mean_overall)^T)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)
            
        # 2. Find the optimal projection directions using Eigenvectors
        # We solve the generalized eigenvalue problem: (S_W^-1 * S_B) * v = lambda * v
        # We add a tiny epsilon to the diagonal of S_W to prevent Singular Matrix errors (division by zero)
        A = np.dot(np.linalg.inv(S_W + 1e-6 * np.eye(n_features)), S_B)
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        # 3. Sort eigenvectors by highest eigenvalue (highest discriminative power)
        # eigh returns them in ascending order, so we reverse them
        eigenvectors = eigenvectors.T[::-1]
        
        # In binary classification, there is only (Classes - 1) = 1 discriminative axis!
        # But we allow keeping more if we ever pass multi-class data
        num_components = self.n_components if self.n_components is not None else len(class_labels) - 1
        self.linear_discriminants = eigenvectors[0:num_components]

    def transform(self, X):
        # Project the original 30-Dimensional data onto our new 1-Dimensional magic axis
        return np.dot(X, self.linear_discriminants.T)
        
    def predict(self, X):
        # A simple classifier: project the point, and assign it to the class with the closest projected mean
        X_projected = self.transform(X)
        
        y_pred = []
        for sample in X_projected:
            distances = {}
            for c, mean_c in self.class_means.items():
                mean_projected = np.dot(mean_c, self.linear_discriminants.T)
                distances[c] = np.linalg.norm(sample - mean_projected)
            
            # Predict the class with the strictly minimum Euclidean distance
            y_pred.append(min(distances, key=distances.get))
            
        return np.array(y_pred)

class ScratchQDA:
    """Quadratic Discriminant Analysis from scratch using Multivariate Gaussians."""
    
    def __init__(self, reg_param=0.1):
        self.reg_param = reg_param
        self.class_priors = {}
        self.class_means = {}
        self.class_cov_invs = {}
        self.class_log_dets = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / n_samples
            self.class_means[c] = np.mean(X_c, axis=0)
            
            # Covariance matrix: (X_c - mean).T @ (X_c - mean) / (N_c - 1)
            # numpy's cov function expects features as rows, so we transpose X_c
            cov_c = np.cov(X_c.T)
            
            # Add Regularization (reg_param) to the diagonal to survive collinearity
            cov_c += self.reg_param * np.eye(n_features)
            
            # Store the mathematical inverse of the Covariance matrix
            self.class_cov_invs[c] = np.linalg.inv(cov_c)
            
            # Safely compute the log of the determinant of the Covariance matrix
            sign, logdet = np.linalg.slogdet(cov_c)
            self.class_log_dets[c] = logdet
            
    def predict(self, X):
        y_pred = []
        for x in X:
            scores = {}
            for c in self.classes:
                # Apply the Quadratic Discriminant function for each class
                diff = x - self.class_means[c]
                
                term1 = -0.5 * self.class_log_dets[c]
                term2 = -0.5 * np.dot(np.dot(diff.T, self.class_cov_invs[c]), diff)
                term3 = np.log(self.class_priors[c])
                
                scores[c] = term1 + term2 + term3
                
            # Predict the class that yielded the highest score
            y_pred.append(max(scores, key=scores.get))
            
        return np.array(y_pred)
