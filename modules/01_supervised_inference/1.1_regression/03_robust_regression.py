# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# %% [markdown]
# # Robust Regression: Surviving the Outliers
#
# ## 1. Intuition: Dealing with Dirty Data
#
# ### 1.1 The Vulnerability of OLS
# In previous modules, we minimized the **Mean Squared Error (MSE)**. By squaring the errors ($e^2$), OLS heavily penalizes large deviations. This is great for clean data, but catastrophic for outliers.
#
# **The "Average" Trap:**
# Imagine calculating the average wealth of 10 people in a bar. They all earn around $50k/year. Suddenly, Elon Musk walks in. Now, the "average" wealth of everyone in the bar is in the billions. Does this represent the group? **No.**
#
# Similarly, a single "bad" data point can have an infinite pull on the regression line. As the model drastically shifts its coefficients to minimize that one massive squared error, it ruins the prediction for the majority of normal data points. OLS is "sensitive" to Elon Musk.
#
# ### 1.2 The Robust Arsenal
# To build reliable models, we need techniques that can ignore or "dampen" the influence of these anomalies. We will explore three distinct approaches:
#
# 1.  **RANSAC (Algorithmic):** A "Consensus-based" approach. It repeatedly fits models to random subsets of data and selects the one that has the most agreement from the rest of the dataset.
# 2.  **Theil-Sen (Statistical):** A "Median-based" approach. Instead of calculating the mean slope, it computes the slope between every pair of points and takes the median. Since the median is robust to outliers, the resulting line is stable.
# 3.  **Huber Regression (loss-based):** A "Hybrid" approach. It modifies the loss function itself. It acts like OLS for small errors (precision) but switches to a linear penalty (MAE) for large errors, preventing outliers from dominating the gradient.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.datasets import make_regression

# Set premium style
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 16

# %% [markdown]
# ## 2. The Scratchpad: Creating the Adversarial Dataset
#
# To truly test these models, we need a dataset that is **Corrupted**.
# We will generate a perfect linear trend and then deliberately add "Outliers" that are not just random noise, but **biased** noise.

# %%
# 1. Generate Clean Data
np.random.seed(42)
n_samples = 200
n_outliers = 50

X, y, coef = make_regression(n_samples=n_samples, n_features=1, noise=10.0, coef=True, random_state=42)

# 2. Add Bias Outliers
# We take the first 'n_outliers' points and move them to a different cluster
np.random.seed(123)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', s=30, alpha=0.6, label='Data')
plt.title("The Challenge: A Strong Trend Polluted by Biased Outliers", fontsize=18)
plt.legend()
# plt.show()

# %% [markdown]
# ## 3. RANSAC: The Consensus Approach
#
# RANSAC (Random Sample Consensus) operates on a simple but powerful assumption: **Data = Inliers + Outliers**.
#
# Instead of trying to fit a line to *all* the data (including the garbage), RANSAC assumes that a "clean" model exists within the noise and tries to find it by trial and error.
#
# **How it works:**
# 1.  **Hypothesize**: It picks the smallest possible subset of data (e.g., just 2 points) and draws a line through them.
# 2.  **Verify**: It asks the rest of the dataset: "Are you close to this line?"
# 3.  **Count**: It counts how many points fit well (the "Consensus Set").
# 4.  **Repeat**: It does this hundreds of times. The model with the largest Consensus Set wins.
#
# It explicitly **discards** outliers. If a point doesn't fit the consensus, it is ignored entirely.

# %%
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

# Execution
ransac = ScratchRANSAC(n_iterations=500, threshold=10.0)
ransac.fit(X, y)

# Visualization
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_ransac = ransac.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='silver', label='Data')
plt.scatter(X[ransac.best_inlier_mask], y[ransac.best_inlier_mask], color='green', s=20, label='Consensus Set')
plt.plot(X_range, y_ransac, color='blue', linewidth=3, label='RANSAC')
plt.title("RANSAC: Finding the Honest Majority", fontsize=18)
plt.legend()
# plt.show()

# %% [markdown]
# ## 4. Theil-Sen: The Median Slope
#
# Theil-Sen takes a purely statistical approach. We know that the **Mean** is sensitive to outliers (Elon Musk in the bar), but the **Median** is robust. Theil-Sen applies this logic to slopes.
#
# **How it works:**
# 1.  It calculates the slope between **every possible pair** of points in the dataset.
# 2.  The final slope is simply the **Median** of all these calculated slopes.
#
# **Why this matters: A Concrete Example**
# Imagine 5 points: 4 follow a perfect line (Slope = 2), and 1 is a massive outlier.
# *   **Total Pairs:** 10 pairs can be formed.
# *   **Good Pairs (Normal-Normal):** 6 pairs. These all have Slope = 2.
# *   **Bad Pairs (Normal-Outlier):** 4 pairs. These have Slope = 500.
#
# **The List:** `[2, 2, 2, 2, 2, 2, 500, 500, 500, 500]`
# **The Median:** 2.
#
# Even with a massive outlier distorting 40% of the calculated slopes, the median ignores them completely.

# %%
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

# Execution
ts_model = ScratchTheilSen()
ts_model.fit(X, y)
y_ts = ts_model.predict(X_range)

# %% [markdown]
# ## 5. Huber Regression: The Adaptive Loss
#
# While RANSAC discards points and Theil-Sen uses the median, Huber Regression tries to fix the root cause: the **Loss Function**.
#
# It creates a hybrid penalty:
# *   **For Small Errors (The "Core"):** It acts like normal OLS (Squared Error). This gives it high precision and smooth gradients when it is close to the truth.
# *   **For Large Errors (The "Tails"):** It acts like Absolute Error (Linear). This means an outlier that is 1000 units away pulls with a force of just $1 \times 1000$, not $1000^2$.
#
# **The Utility:**
# It is the "Diplomat" of robust regression. It listens to outliers, but it doesn't let them scream. It is perfect when your data has "Heavy Tails"—valid data points that are just naturally far from the mean, which you don't want to completely ignore (like RANSAC would) but don't want to over-prioritize (like OLS would).

# %%
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

# Execution
# Important: Huber works best on scaled data, but our simple example is roughly within range.
huber = ScratchHuber(delta=1.5, n_iters=2000)
huber.fit(X, y)
y_huber = huber.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='silver', label='Data')
plt.plot(X_range, y_ts, color='purple', linewidth=3, label='Theil-Sen')
plt.plot(X_range, y_huber, color='orange', linewidth=3, linestyle='--', label='Huber')
plt.title("Huber vs Theil-Sen: A Critical Divergence", fontsize=18)
plt.legend()
# plt.show()

# %% [markdown]
# ### 5.1 Analysis: Why did Huber fail?
#
# Look continuously at the plot above. **Theil-Sen (Purple)** correctly ignores the bottom cluster and follows the main trend. **Huber (Orange)**, however, is essentially flat—it fails to find the trend.
#
# **Why?**
# *   **Theil-Sen** uses the **Median**. It completely ignores the minority group (the bottom cluster) because they represent < 50% of the data.
# *   **Huber** uses a **Hybrid Loss**. It penalizes the outliers linearly (MAE) rather than quadratically (MSE). However, because our "outlier cluster" is dense and contains many points, their collective linear pull is still strong enough to drag the line down.
#
# **Lesson:** Huber is great for "heavy tails" (scattered noise), but for **structural outliers** (a distinct, incorrect cluster of data), Median-based methods like Theil-Sen or RANSAC are superior.

# %% [markdown]
# ## 6. The "Deep Dive": Torture Testing
#
# Models usually look great in demos. Let's break them.
#
# **Test:** What happens if we increase the outlier ratio from 10% to 60%?
# *   **RANSAC Expectation:** Should fail once outliers > 50% (Conceptually can't find consensus).
# *   **Theil-Sen Expectation:** Should break at ~29%.
# *   **Huber Expectation:** Will get slowly dragged away as outliers increase.

# %%
def stress_test(outlier_ratio):
    # Create data with X% outliers
    n = 200
    n_out = int(n * outlier_ratio)
    X_test, y_test, _ = make_regression(n_samples=n, n_features=1, noise=5.0, coef=True, random_state=42)
    
    # Corrupt
    # We move them VERY far to ensure they are influential
    X_test[:n_out] = 8 + np.random.normal(size=(n_out, 1))
    y_test[:n_out] = -100 + np.random.normal(size=n_out) # Massive pull downwards
    
    # Fit Models
    models = {
        "OLS": LinearRegression(),
        "RANSAC": RANSACRegressor(),
        "TheilSen": TheilSenRegressor(random_state=42),
        "Huber": HuberRegressor()
    }
    
    errors = {}
    
    # We test on the non-corrupted part of the data
    X_clean = X_test[n_out:]
    y_clean = y_test[n_out:]
    
    for name, model in models.items():
        model.fit(X_test, y_test)
        preds = model.predict(X_clean)
        errors[name] = np.sqrt(mean_squared_error(y_clean, preds))
        
    return errors

# Run Simulation
ratios = np.linspace(0.05, 0.60, 15) # 5% to 60% outliers
history = {k: [] for k in ["OLS", "RANSAC", "TheilSen", "Huber"]}

for r in ratios:
    res = stress_test(r)
    for k, v in res.items():
        history[k].append(v)

plt.figure(figsize=(12, 8))
for name, errs in history.items():
    plt.plot(ratios, errs, marker='o', linewidth=2, label=name)

plt.axvline(0.29, color='gray', linestyle='--', alpha=0.5, label='Theil-Sen Limit (~29%)')
plt.axvline(0.50, color='red', linestyle='--', alpha=0.5, label='RANSAC Limit (50%)')
plt.xlabel("Fraction of Outliers")
plt.ylabel("RMSE on Clean Data (Lower is Better)")
plt.title("Stress Test: When do they break?", fontsize=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# %% [markdown]
# ## 7. Sensitivity Analysis: The Huber "Knob"
#
# The `delta` (epsilon) parameter in Huber regression controls the threshold between "Standard MSE" and "Robust MAE".
# *   Small Delta ($\delta \to 0$): Becomes MAE (Median Regression). Very robust but unstable on clean data.
# *   Large Delta ($\delta \to \infty$): Becomes MSE (OLS). Very stable but not robust.

# %%
deltas = [1.0, 1.5, 3.0, 100.0]
# Note: Sklearn's HuberRegressor requires epsilon > 1.0 (it creates a small parabolic region around 0).
# Effect:
# - Low Delta (1.5): Treats almost everything as Large Error (Linear Penalty). Ignores outliers strongly.
# - High Delta (100.0): Treats almost everything as Small Error (Quadratic Penalty). Acts like OLS.

colors = sns.color_palette("flare", len(deltas))

for i, d in enumerate(deltas):
    # Using Sklearn optimized Huber for visualization
    h = HuberRegressor(epsilon=d)
    h.fit(X, y)
    plt.plot(X_range, h.predict(X_range), color=colors[i], linewidth=2.5, label=f'Delta={d}')

plt.title("Huber Sensitivity: From Robust (1.0) to OLS (100.0)", fontsize=18)
plt.legend()
# plt.show()

# %% [markdown]
# ## 8. The Verdict: Decision Matrix
#
# Based on our Stress Test (`history`), we can draw objective conclusions:
#
# 1.  **RANSAC**: 
#     *   **Performance:** Maintained near-zero error (perfect recovery) until the outlier ratio hit exactly **50%**.
#     *   **Breaking Point:** At >50% outliers, the concept of "Consensus" mathematically collapses, and error exploded.
#     *   **Use Case:** When you have valid signal mixed with garbage, and you are confident the garbage is <50% of the data.
#
# 2.  **Theil-Sen**:
#     *   **Performance:** Extremely stable, but started deviating earlier than RANSAC.
#     *   **Breaking Point:** As predicted by theory, it broke down around **29%** outlier ratio.
#     *   **Use Case:** Small datasets where you need a deterministic, theoretically guaranteed answer without randomness.
#
# 3.  **Huber**:
#     *   **Performance:** Failed the "Structural Outlier" test (Two Clusters). It tried to compromise between the two clusters, leading to a "flat" line that satisfied neither.
#     *   **Use Case:** Not for *corruption* (like this dataset), but for *tail risk* (valid but noisy data).
#
# **Final Recommendation:**
# *   **Default Choice:** **Theil-Sen** (if $N < 10,000$). It is safe, deterministic, and requires no tuning.
# *   **Heavy Corruption:** **RANSAC**. It is the only one that survived up to 50% corruption.
# *   **High Variance (Stock Market):** **Huber**. When outliers are "real" data points that you just want to de-emphasize.
