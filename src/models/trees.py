import numpy as np

class Node:
    """A single node inside the Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only leaf nodes will have a value
        
    def is_leaf(self):
        return self.value is not None

class ScratchDecisionTree:
    """Decision Tree Classifier built from scratch."""
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        # Calculate probabilities of each class
        p = np.bincount(y) / m
        return 1.0 - np.sum(p ** 2)
        
    def _split(self, X_column, threshold):
        # We find indices where the feature is <= threshold (left) and > threshold (right)
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
        
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        n_samples, n_features = X.shape
        parent_gini = self._gini_impurity(y)
        
        # Iterate over all features
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            # Try every single unique value as a split point!
            for thr in thresholds:
                left_idxs, right_idxs = self._split(X_column, thr)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                    
                # Calculate Information Gain
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
                
                child_gini = (n_l / n) * e_l + (n_r / n) * e_r
                ig = parent_gini - child_gini
                
                # If this gives the biggest drop in Gini so far, save it
                if ig > best_gain:
                    best_gain = ig
                    split_idx = feature_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
        
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping Criteria:
        # 1. We reached max allowed depth
        # 2. We don't have enough samples to justify another split
        # 3. The node is perfectly pure (only 1 class left)
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            # This is a Leaf node. Output the most common class.
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
            
        # Find the absolute best question to ask right now
        best_feat, best_thresh = self._best_split(X, y)
        
        # If we couldn't find a valid split, turn into a leaf
        if best_feat is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
            
        # Recursive construction! Build the left branch, then the right branch
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
            
        # If the passenger's feature is <= the threshold, go left
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

class ScratchRandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            # Bootstrap Aggregating (Bagging): Randomly sample rows with replacement
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            
            # The tree needs to know it should only use a random subset of features per split
            # For simplicity in this scratch implementation, we'll just determine max_features here
            if self.max_features == 'sqrt':
                n_sub_features = int(np.sqrt(n_features))
            else:
                n_sub_features = n_features
                
            # We create a modified ScratchDecisionTree that only considers a random subset of features
            # To avoid heavily modifying the existing ScratchDecisionTree, we'll build a specialized
            # internal tree for the Forest.
            tree = _ForestTree(max_depth=self.max_depth, 
                               min_samples_split=self.min_samples_split, 
                               n_sub_features=n_sub_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        # Predict with all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds is shape (n_estimators, n_samples)
        # We want to take the majority vote for each sample
        # Swap axes to (n_samples, n_estimators)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        y_pred = [np.bincount(preds).argmax() for preds in tree_preds]
        return np.array(y_pred)

class _ForestTree:
    """A specialized decision tree that uses a random subset of features at each split."""
    # (Simplified from ScratchDecisionTree)
    def __init__(self, max_depth=10, min_samples_split=2, n_sub_features=None, extra_trees=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_sub_features = n_sub_features
        self.extra_trees = extra_trees
        self.root = None
        
    def _gini_impurity(self, y):
        m = len(y)
        if m == 0: return 0.0
        p = np.bincount(y) / m
        return 1.0 - np.sum(p ** 2)
        
    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
        
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        n_samples, n_features = X.shape
        parent_gini = self._gini_impurity(y)
        
        # Random subset of features
        feat_idxs = np.random.choice(n_features, self.n_sub_features, replace=False)
        
        for feature_idx in feat_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            if self.extra_trees and len(thresholds) > 1:
                # Extra Trees: pick a COMPLETELY RANDOM threshold between min and max
                thr = np.random.uniform(np.min(thresholds), np.max(thresholds))
                thresholds = [thr]
                
            for thr in thresholds:
                left_idxs, right_idxs = self._split(X_column, thr)
                if len(left_idxs) == 0 or len(right_idxs) == 0: continue
                
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
                
                child_gini = (n_l / n) * e_l + (n_r / n) * e_r
                ig = parent_gini - child_gini
                
                if ig > best_gain:
                    best_gain = ig
                    split_idx = feature_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
        
    def _build_tree(self, X, y, depth=0):
        from trees import Node 
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            return Node(value=np.bincount(y).argmax())
            
        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return Node(value=np.bincount(y).argmax())
            
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

class ScratchExtraTrees(ScratchRandomForest):
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            # Extra Trees typically use the whole dataset, but Bagley can be used. Let's strictly do whole dataset.
            X_sample, y_sample = X, y
            
            if self.max_features == 'sqrt':
                n_sub_features = int(np.sqrt(n_features))
            else:
                n_sub_features = n_features
                
            # extra_trees=True enables random threshold picking
            tree = _ForestTree(max_depth=self.max_depth, 
                               min_samples_split=self.min_samples_split, 
                               n_sub_features=n_sub_features,
                               extra_trees=True)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
