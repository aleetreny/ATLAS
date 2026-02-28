import numpy as np
from sklearn.neighbors import NearestNeighbors


class ScratchSampler:
    """Base sampler with common class-splitting logic."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _split_classes(self, y):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Scratch samplers in this module expect binary targets.")

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        idx_maj = np.where(y == majority_class)[0]
        idx_min = np.where(y == minority_class)[0]
        return idx_maj, idx_min, majority_class, minority_class


class ScratchRandomUndersampler(ScratchSampler):
    """Randomly downsample majority class to minority size."""

    def fit_resample(self, X, y):
        rng = self._rng()
        idx_maj, idx_min, _, _ = self._split_classes(y)

        if len(idx_maj) <= len(idx_min):
            return X.copy(), y.copy()

        undersampled_maj_idx = rng.choice(idx_maj, size=len(idx_min), replace=False)
        final_idx = np.concatenate([undersampled_maj_idx, idx_min])
        rng.shuffle(final_idx)
        return X[final_idx], y[final_idx]


class ScratchRandomOversampler(ScratchSampler):
    """Randomly upsample minority class to majority size."""

    def fit_resample(self, X, y):
        rng = self._rng()
        idx_maj, idx_min, _, _ = self._split_classes(y)

        n_needed = len(idx_maj) - len(idx_min)
        if n_needed <= 0:
            return X.copy(), y.copy()

        oversampled_min_idx = rng.choice(idx_min, size=n_needed, replace=True)
        final_min_idx = np.concatenate([idx_min, oversampled_min_idx])
        final_idx = np.concatenate([idx_maj, final_min_idx])
        rng.shuffle(final_idx)
        return X[final_idx], y[final_idx]


class ScratchSMOTE(ScratchSampler):
    """SMOTE from scratch using nearest-neighbor interpolation."""

    def __init__(self, k_neighbors=5, random_state=42):
        super().__init__(random_state=random_state)
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        rng = self._rng()
        idx_maj, idx_min, majority_class, minority_class = self._split_classes(y)

        X_min = X[idx_min]
        n_majority = len(idx_maj)
        n_minority = len(idx_min)
        n_synthetic_needed = n_majority - n_minority

        if n_synthetic_needed <= 0:
            return X.copy(), y.copy()

        if n_minority < 2:
            raise ValueError("SMOTE requires at least 2 minority samples.")

        effective_k = min(self.k_neighbors, n_minority - 1)
        nn = NearestNeighbors(n_neighbors=effective_k + 1)
        nn.fit(X_min)

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            idx = rng.integers(0, n_minority)
            base_point = X_min[idx]

            _, neighbor_indices = nn.kneighbors(base_point.reshape(1, -1))
            valid_neighbors = neighbor_indices[0][1:]
            chosen_neighbor_idx = rng.choice(valid_neighbors)
            neighbor_point = X_min[chosen_neighbor_idx]

            gap = rng.uniform(0.0, 1.0)
            synthetic_point = base_point + gap * (neighbor_point - base_point)
            synthetic_samples.append(synthetic_point)

        synthetic_samples = np.asarray(synthetic_samples)

        X_resampled = np.vstack([X[idx_maj], X[idx_min], synthetic_samples])
        y_resampled = np.concatenate(
            [
                np.full(n_majority, majority_class, dtype=y.dtype),
                np.full(n_minority + n_synthetic_needed, minority_class, dtype=y.dtype),
            ]
        )

        shuffle_idx = rng.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
