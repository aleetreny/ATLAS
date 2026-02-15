import numpy as np


class DecisionStump:
    """A single-split decision tree (depth=1). The simplest weak learner.
    
    Finds the best feature and threshold to split data into two groups,
    predicting the mean of each group. Despite its simplicity, hundreds
    of stumps combined through boosting can approximate arbitrarily
    complex functions.
    """

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals, n_candidates=50):
        """Find the best single split that minimizes MSE on the residuals.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        residuals : array-like of shape (n_samples,)
            Current residuals (negative gradient of the loss).
        n_candidates : int, default=50
            Number of percentile-based candidate thresholds per feature.
        """
        X = np.array(X)
        best_mse = np.inf
        n_samples, n_features = X.shape

        for feat in range(n_features):
            values = X[:, feat]
            thresholds = np.percentile(values, np.linspace(5, 95, n_candidates))

            for thr in thresholds:
                left_mask = values <= thr
                right_mask = ~left_mask

                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue

                left_mean = residuals[left_mask].mean()
                right_mean = residuals[right_mask].mean()

                preds = np.where(left_mask, left_mean, right_mean)
                mse = np.mean((residuals - preds) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feat
                    self.threshold = thr
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        """Predict using the learned split.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        X = np.array(X)
        return np.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_value,
            self.right_value
        )


class ScratchGradientBoosting:
    """Gradient Boosting Regressor built from scratch using Decision Stumps.
    
    Implements the vanilla gradient boosting algorithm for regression with
    MSE loss. Each iteration fits a DecisionStump to the current residuals
    and adds it to the ensemble with a learning rate shrinkage.
    
    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting rounds (stumps to fit).
    learning_rate : float, default=0.1
        Shrinkage factor applied to each stump's prediction.
        Smaller values require more estimators but generalize better.
    """

    def __init__(self, n_estimators=200, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.initial_prediction = None
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the gradient boosting ensemble.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        y_train : array-like of shape (n_samples,)
            Training targets.
        X_val : array-like of shape (n_val, n_features), optional
            Validation features for tracking overfitting.
        y_val : array-like of shape (n_val,), optional
            Validation targets.
        """
        self.initial_prediction = np.mean(y_train)
        current_preds = np.full(len(y_train), self.initial_prediction)

        for t in range(self.n_estimators):
            # Step 1: Compute residuals (negative gradient of MSE)
            residuals = y_train - current_preds

            # Step 2: Fit a stump to the residuals
            stump = DecisionStump()
            stump.fit(X_train, residuals)
            self.stumps.append(stump)

            # Step 3: Update predictions
            current_preds += self.learning_rate * stump.predict(X_train)

            # Track losses
            train_rmse = np.sqrt(np.mean((y_train - current_preds) ** 2))
            self.train_losses.append(train_rmse)

            if X_val is not None:
                val_preds = self.predict(X_val)
                val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
                self.val_losses.append(val_rmse)

    def predict(self, X):
        """Predict using the boosted ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        preds = np.full(len(X), self.initial_prediction)
        for stump in self.stumps:
            preds += self.learning_rate * stump.predict(X)
        return preds
