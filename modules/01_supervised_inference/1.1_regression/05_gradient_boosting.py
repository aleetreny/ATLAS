# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Gradient Boosting: The Industry Standard for Tabular Data
#
# ## 1. Theoretical Foundation
#
# ### 1.1 The Philosophy
# In the previous modules, we explored regression through increasingly flexible lenses:
# - **Classical Statistics** gave us interpretable, closed-form solutions (OLS, GLMs).
# - **Regularization** taught us to constrain complexity (Ridge, Lasso).
# - **Robust Regression** showed how to survive corrupted data (RANSAC, Theil-Sen).
# - **Non-Parametric Methods** abandoned fixed equations entirely (KNN, Kernels, GPs).
#
# Each of these methods has a clear niche. But if you walk into a data science team today and ask *"What do you use for tabular prediction?"*, the answer is almost universal: **Gradient Boosting.**
#
# XGBoost, LightGBM, and CatBoost dominate Kaggle competitions, production ML pipelines, and enterprise analytics. They are not theoretically elegant like GPs, nor interpretable like OLS. They win because they are **relentlessly practical**: fast to train, robust to messy data, and consistently accurate across domains.
#
# ### 1.2 The Core Idea: Ensembles of Weak Learners
# The philosophy is deceptively simple:
#
# > *"A committee of mediocre experts, each correcting the mistakes of the previous one, will outperform any single genius."*
#
# Instead of building one powerful model, we build **hundreds of tiny, shallow Decision Trees** (typically 3-8 levels deep). Each tree is intentionally weak. But each subsequent tree is trained not on the original data, but on the **residual errors** of the ensemble so far.
#
# Formally, at iteration $t$, we fit a new tree $h_t(x)$ to the negative gradient of the loss function:
#
# $$ F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x) $$
#
# Where:
# - $F_t(x)$ is the ensemble prediction after $t$ trees.
# - $h_t(x)$ is the new tree that approximates the gradient (the "direction of improvement").
# - $\eta$ is the **learning rate**, a shrinkage factor that prevents overcorrection.
#
# ### 1.3 Why It Dominates Tabular Data
# Neural Networks rule images, text, and audio. But for structured/tabular data (CSVs, databases, spreadsheets), Gradient Boosting consistently wins. Why?
#
# 1. **Heterogeneous Features:** Tables mix continuous, categorical, and ordinal data. Trees handle this natively. Neural networks require extensive preprocessing.
# 2. **Feature Interactions:** A tree split like `IF carat > 1.0 AND clarity < VS2` captures interactions automatically. Linear models need manual feature engineering.
# 3. **Missing Values:** XGBoost and LightGBM learn optimal "default directions" for missing data during training. No imputation needed.
# 4. **Scale Invariance:** Trees split on rank order, not magnitude. No normalization required.
# 5. **Sample Efficiency:** They perform well with thousands of rows, not millions.
#
# ### 1.4 The Roadmap
# In this notebook, we will build the intuition and machinery behind the four dominant implementations:
#
# 1. **Gradient Boosting from Scratch:** The core algorithm using Decision Stumps and MSE.
# 2. **XGBoost:** The original "fast and regularized" implementation that started the revolution.
# 3. **LightGBM:** Microsoft's histogram-based splitting for massive speed gains.
# 4. **CatBoost:** Yandex's approach, specialized for categorical features with ordered boosting.
# 5. **Comparison:** Head-to-head benchmarks on real data, analyzing speed, accuracy, and behavior under different data conditions.
#
# **Dataset:** We use the **Diamonds** dataset (53,940 rows) from seaborn. Its mix of continuous features (carat, depth, table), ordinal categories (cut, color, clarity), and a highly non-linear, heteroscedastic target (price) makes it a perfect stress test for boosting methods.
#
# ---
# *We follow the **ATLAS** protocol: First we understand the **Intuition**, then we build it from **Scratch** using only NumPy, and finally we apply the **Pro** tools used in industry.*

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set aesthetics
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15

# %% [markdown]
# ## 2. Data Preparation: The Diamonds Dataset
#
# Gradient Boosting thrives on **messy, heterogeneous** data. The Diamonds dataset is ideal because:
# - **Volume:** 53,940 observations. Enough to train deep ensembles without subsampling.
# - **Mixed types:** 3 categorical features (cut, color, clarity) and 6 continuous features (carat, depth, table, x, y, z).
# - **Non-linearity:** The price-carat relationship is exponential, not linear. A 2-carat diamond is not twice the price of a 1-carat diamond; it can be 4-5x.
# - **Heteroscedasticity:** Variance in price explodes at higher carats. Linear models struggle here.

# %%
# Load and inspect
import os
_cache = os.path.expanduser('~/seaborn-data/diamonds.csv')
df = pd.read_csv(_cache) if os.path.exists(_cache) else sns.load_dataset('diamonds')

print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print("\nBasic Statistics:")
df.describe()

# %% [markdown]
# **53,940 rows, 10 columns.** Three categorical features (`cut`, `color`, `clarity`) stored as `object`, and seven numeric columns. Notice the target variable `price`: its mean (&#36;3,933) and standard deviation (&#36;3,989) are nearly identical, signaling extreme right-skew. The minimum price is &#36;326 while the max reaches &#36;18,823. The physical dimensions (`x`, `y`, `z`) contain zeros, which are likely measurement errors, but tree-based models handle these naturally without imputation.

# %%
# Ordinal encoding (natural order matters for these features)
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # J=worst, D=best
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

df['cut'] = pd.Categorical(df['cut'], categories=cut_order, ordered=True).codes
df['color'] = pd.Categorical(df['color'], categories=color_order, ordered=True).codes
df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_order, ordered=True).codes

# Define features and target
X = df.drop(columns=['price'])
y = df['price'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: predicting the mean
baseline_preds = np.full_like(y_test, np.mean(y_train), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
print(f"Baseline RMSE (Mean Prediction): ${baseline_rmse:,.2f}")

# %% [markdown]
# **43,152 training / 10,788 test samples.** The ordinal encoding preserves the natural hierarchy of each categorical variable (e.g., Fair=0 < Ideal=4 for cut). This matters: unlike one-hot encoding, it tells the tree that `Ideal > Premium > Very Good`, which reduces the number of splits needed.
#
# The **Baseline RMSE is &#36;3,987** (simply predicting the mean price for every diamond). This is our floor. Any model worth its compute must beat this significantly. Given the high variance in the data, we expect gradient boosting to cut this error by 80-90%.

# %%
# Visualize the core challenge: non-linear heteroscedasticity
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Price distribution (heavy right skew)
axes[0].hist(y, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Price Distribution')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Count')

# 2. Price vs Carat (the exponential relationship)
axes[1].scatter(df['carat'], y, alpha=0.08, s=5, color='steelblue')
axes[1].set_title('Price vs. Carat')
axes[1].set_xlabel('Carat')
axes[1].set_ylabel('Price ($)')

# 3. Price vs Clarity (ordinal categorical)
clarity_labels = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
axes[2].boxplot([y[df['clarity'] == i] for i in range(8)], tick_labels=clarity_labels)
axes[2].set_title('Price by Clarity Grade')
axes[2].set_xlabel('Clarity')
axes[2].set_ylabel('Price ($)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# Three key observations from the plots:
#
# 1. **Price Distribution** is heavily right-skewed. The bulk of diamonds cluster below &#36;2,500, with a long tail reaching &#36;18,823. This is typical of luxury goods and suggests that log-transforming the target could help linear models, but boosting handles skewed targets natively.
# 2. **Price vs. Carat** reveals a clearly **exponential** relationship with expanding variance (heteroscedasticity). At 0.5 carats, prices are tight (&#36;500-&#36;2,000). At 2 carats, the same plot spans &#36;5,000-&#36;18,000. A linear model would systematically under-predict expensive diamonds and over-predict cheap ones. Decision trees, by partitioning the space recursively, adapt to this local variance naturally.
# 3. **Price by Clarity** shows a surprising pattern: the median price *decreases* from lower clarity (I1, SI2) to higher clarity (VVS1, IF). This is **Simpson's Paradox** in action: larger diamonds (higher carat, higher price) tend to have lower clarity grades. Clarity alone is a misleading predictor. Gradient Boosting captures these confounded interactions through its tree structure without any manual feature engineering.

# %% [markdown]
# ## 3. Gradient Boosting from Scratch
#
# ### 3.1 The Algorithm, Step by Step
# Before touching XGBoost or LightGBM, we need to understand what they are optimizing under the hood. The vanilla Gradient Boosting algorithm for regression (with MSE loss) works as follows:
#
# 1. **Initialize** with a constant prediction: $F_0(x) = \bar{y}$ (the mean of training targets).
# 2. **For each round** $t = 1, 2, \ldots, T$:
#    - Compute **residuals**: $r_i^{(t)} = y_i - F_{t-1}(x_i)$. These are the errors the current ensemble still makes.
#    - Fit a **Decision Tree** $h_t(x)$ to the residuals $r^{(t)}$. The tree learns *where* and *how much* to correct.
#    - **Update** the ensemble: $F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$.
# 3. **Output** the final ensemble $F_T(x)$.
#
# The learning rate $\eta \in (0, 1]$ controls how aggressively we correct. Smaller $\eta$ means more trees are needed, but the model generalizes better (less overfitting).
#
# ### 3.2 The Decision Stump
# A **Decision Stump** is a tree with a single split: one feature, one threshold. It divides the data into two groups and predicts the mean of each group. It is the simplest possible "weak learner". Despite its simplicity, hundreds of stumps combined through boosting can approximate arbitrarily complex functions.

# %%
class DecisionStump:
    """A single-split decision tree (depth=1). The simplest weak learner."""

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals, n_candidates=50):
        """Find the best single split that minimizes MSE on the residuals."""
        X = np.array(X)
        best_mse = np.inf
        n_samples, n_features = X.shape

        for feat in range(n_features):
            values = X[:, feat]
            # Sample candidate thresholds (percentiles) for speed
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
        X = np.array(X)
        return np.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_value,
            self.right_value
        )


class ScratchGradientBoosting:
    """Gradient Boosting Regressor built from scratch using Decision Stumps."""

    def __init__(self, n_estimators=200, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.initial_prediction = None
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
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
        preds = np.full(len(X), self.initial_prediction)
        for stump in self.stumps:
            preds += self.learning_rate * stump.predict(X)
        return preds

# %% [markdown]
# The `DecisionStump` is deliberately minimal: for each feature it samples 50 candidate thresholds (percentiles), tries each split, and keeps the one that minimizes MSE on the residuals. The `ScratchGradientBoosting` orchestrates the loop: initialize with the mean, compute residuals, fit a stump, update predictions, repeat.
#
# Note that we track both training and validation RMSE at each iteration. This will let us visualize the classic **overfitting curve** where train error keeps dropping but validation error eventually plateaus or rises.

# %%
# Train our scratch model (stumps only, 200 rounds)
scratch_gb = ScratchGradientBoosting(n_estimators=200, learning_rate=0.1)
scratch_gb.fit(X_train.values, y_train, X_val=X_test.values, y_val=y_test)

# Evaluate
scratch_preds = scratch_gb.predict(X_test.values)
scratch_rmse = np.sqrt(mean_squared_error(y_test, scratch_preds))
scratch_r2 = r2_score(y_test, scratch_preds)
print(f"Scratch GB (200 stumps) - RMSE: ${scratch_rmse:,.2f} | R2: {scratch_r2:.4f}")
print(f"Improvement over baseline: {(1 - scratch_rmse / baseline_rmse) * 100:.1f}%")

# %% [markdown]
# **RMSE: &#36;1,119 | R2: 0.9212 | 71.9% improvement over baseline.** Even with the simplest possible weak learner (single-split stumps), gradient boosting explains 92% of the variance in diamond prices and cuts the prediction error from &#36;3,987 to &#36;1,119. Not bad for an algorithm built entirely from NumPy with no hyperparameter tuning. Deeper trees would improve this further, but the point is clear: the *boosting loop itself* is what creates the power, not the individual trees.

# %%
# === VISUALIZATION 1: Learning Curve ===
fig, ax = plt.subplots(figsize=(14, 7))

rounds = np.arange(1, len(scratch_gb.train_losses) + 1)
ax.plot(rounds, scratch_gb.train_losses, color='#2196F3', linewidth=2.5, label='Train RMSE', alpha=0.9)
ax.plot(rounds, scratch_gb.val_losses, color='#FF5722', linewidth=2.5, label='Validation RMSE', alpha=0.9)

# Mark the best validation point
best_round = np.argmin(scratch_gb.val_losses) + 1
best_val = min(scratch_gb.val_losses)
ax.axvline(x=best_round, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=1.5)
ax.scatter([best_round], [best_val], color='#4CAF50', s=120, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Best: Round {best_round}\nRMSE: ${best_val:,.0f}',
            xy=(best_round, best_val), xytext=(best_round + 20, best_val + 300),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

# Baseline reference
ax.axhline(y=baseline_rmse, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(5, baseline_rmse + 80, f'Baseline: ${baseline_rmse:,.0f}', fontsize=11, color='gray')

ax.set_xlabel('Boosting Round')
ax.set_ylabel('RMSE ($)')
ax.set_title('Scratch Gradient Boosting: Learning Curve', fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.set_xlim(0, len(rounds) + 5)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The learning curve reveals the core dynamics of gradient boosting:
#
# - **Rapid initial descent:** The first 10 rounds drop validation RMSE from &#36;3,987 to &#36;2,394; a 40% error reduction. Each new stump corrects the biggest remaining systematic error (e.g., "carat > 1 implies higher price").
# - **Diminishing returns:** By round 50, RMSE reaches &#36;1,374. The next 150 rounds squeeze out only another &#36;255. Each new stump captures smaller and smaller residual patterns.
# - **No overfitting:** The best round is 200 (the last one), and the train/validation curves stay nearly overlapping. With stumps (depth=1), overfitting is almost impossible. The model is too simple to memorize noise. This is why the learning rate matters: it keeps each correction small, forcing the ensemble to build consensus across many trees.
#
# With deeper trees (depth 4-8), we would see the validation curve eventually diverge upward from the training curve. That is the classic bias-variance tradeoff in action.

# %%
# === VISUALIZATION 2: Residual Evolution ===
# Show how residuals shrink as boosting progresses
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
checkpoints = [1, 10, 50, 200]
colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5']

for idx, (n_trees, color) in enumerate(zip(checkpoints, colors)):
    # Compute predictions using only first n_trees stumps
    preds = np.full(len(y_test), scratch_gb.initial_prediction)
    for stump in scratch_gb.stumps[:n_trees]:
        preds += scratch_gb.learning_rate * stump.predict(X_test.values)
    residuals = y_test - preds

    axes[idx].hist(residuals, bins=60, color=color, edgecolor='white', alpha=0.85)
    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    rmse = np.sqrt(np.mean(residuals ** 2))
    axes[idx].set_title(f'After {n_trees} tree{"s" if n_trees > 1 else ""}\nRMSE: ${rmse:,.0f}',
                        fontweight='bold', fontsize=13)
    axes[idx].set_xlabel('Residual ($)')
    axes[idx].set_ylabel('Count' if idx == 0 else '')
    axes[idx].set_xlim(-10000, 10000)

plt.suptitle('Residual Distribution as Boosting Progresses', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# This is the visual proof of how boosting works. Watch the residual distribution evolve:
#
# - **After 1 tree (RMSE: &#36;3,752):** Residuals are wide and bimodal. The single stump can only split the data into two groups, leaving massive errors for most diamonds.
# - **After 10 trees (RMSE: &#36;2,394):** The distribution tightens noticeably. The ensemble has learned the major structural patterns (carat being the dominant price driver). A 36% error drop in just 9 additional stumps.
# - **After 50 trees (RMSE: &#36;1,374):** Most residuals cluster tightly around zero. The tails are shrinking as more specific interactions (carat x clarity, color x cut) get captured.
# - **After 200 trees (RMSE: &#36;1,119):** A sharp, concentrated distribution. The remaining errors are the "hard cases" where diamond price depends on factors not in our features (brand, certification, market conditions).
#
# Each successive tree is literally "eating" the tails of the residual distribution. This is gradient descent in function space.

# %%
# === VISUALIZATION 3: Predicted vs Actual ===
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter with density coloring
hb = ax.hexbin(y_test, scratch_preds, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
cb = fig.colorbar(hb, ax=ax, label='Count')

# Perfect prediction line
lims = [0, max(y_test.max(), scratch_preds.max()) * 1.05]
ax.plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--', alpha=0.8, label='Perfect Prediction')

ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('Scratch GB: Predicted vs. Actual Price', fontweight='bold')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.legend(fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The hexbin plot shows density of predictions vs. actuals (R2 = 0.9212). Points on the orange dashed line are perfect predictions.
#
# - The **dense cluster** along the diagonal at lower prices (&#36;0-&#36;5,000) shows the model handles the bulk of the data well. This is where most of the 53,940 diamonds live.
# - At **higher prices** (&#36;10,000+), predictions spread further from the diagonal. This is expected: fewer training examples exist in this range, and the heteroscedasticity we observed earlier means there is genuinely more randomness at high carats.
# - The **horizontal banding** is characteristic of stumps: each stump can only output two values per tree, so the ensemble produces a limited number of distinct prediction levels. Deeper trees would smooth this out considerably.
#
# For a from-scratch model using only single-split trees, &#36;1,119 RMSE and 92% R2 is a strong result. Next, we will see how the industry libraries (XGBoost, LightGBM, CatBoost) improve on this by using deeper trees, regularization, and optimized splitting algorithms.

# %% [markdown]
# ## 4. XGBoost: The Revolution
#
# ### 4.1 What XGBoost Changed
# In 2014, Tianqi Chen published XGBoost (eXtreme Gradient Boosting) and the landscape of applied machine learning shifted permanently. Before XGBoost, gradient boosting existed (Friedman, 2001), but it was slow, memory-hungry, and lacked regularization. XGBoost introduced three key innovations:
#
# 1. **Regularized Objective:** XGBoost adds L1 and L2 penalties on the leaf weights, plus a term penalizing the number of leaves. This prevents individual trees from becoming too complex.
# 2. **Second-Order Approximation:** Instead of using only the gradient (first derivative) of the loss, XGBoost uses a second-order Taylor expansion (gradient + Hessian). This gives better split quality estimates and faster convergence.
# 3. **System Optimizations:** Column-block structure for parallel split finding, cache-aware memory access, out-of-core computation for datasets that exceed RAM, and built-in handling of sparse data and missing values.
#
# The regularized objective at each step is:
#
# $$ \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) $$
#
# Where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ penalizes tree complexity ($T$ = number of leaves, $w_j$ = leaf weights, $\gamma$ and $\lambda$ are regularization strengths).
#
# ### 4.2 Key Hyperparameters
# | Parameter | Role | Typical Range |
# |-----------|------|---------------|
# | `n_estimators` | Number of boosting rounds | 100 - 2000 |
# | `max_depth` | Maximum tree depth | 3 - 10 |
# | `learning_rate` | Shrinkage per round | 0.01 - 0.3 |
# | `subsample` | Row sampling ratio per tree | 0.5 - 1.0 |
# | `colsample_bytree` | Column sampling ratio per tree | 0.5 - 1.0 |
# | `reg_alpha` | L1 regularization on weights | 0 - 10 |
# | `reg_lambda` | L2 regularization on weights | 0 - 10 |
# | `min_child_weight` | Minimum sum of Hessians in a leaf | 1 - 10 |

# %%
import xgboost as xgb
import time

# XGBoost with sensible defaults
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

start = time.time()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
xgb_time = time.time() - start

xgb_preds = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print("XGBoost (500 trees, depth=6)")
print(f"  RMSE:  ${xgb_rmse:,.2f}")
print(f"  MAE:   ${xgb_mae:,.2f}")
print(f"  R2:    {xgb_r2:.4f}")
print(f"  Time:  {xgb_time:.2f}s")
print(f"  vs Scratch: {(1 - xgb_rmse / scratch_rmse) * 100:.1f}% RMSE reduction")
print(f"  vs Baseline: {(1 - xgb_rmse / baseline_rmse) * 100:.1f}% RMSE reduction")

# %% [markdown]
# **RMSE: &#36;543 | MAE: &#36;271 | R2: 0.9814 | 86.4% improvement over baseline.** The leap from scratch to XGBoost is dramatic: a 51% RMSE reduction (&#36;1,119 to &#36;543) and R2 jumping from 0.92 to 0.98. XGBoost now explains 98% of the variance in diamond prices. The median prediction error (MAE &#36;271) means XGBoost is typically within &#36;271 of the actual price, remarkable for a dataset spanning &#36;326 to &#36;18,823. Training took ~7 seconds for 500 trees on 43K rows, orders of magnitude faster than our scratch implementation.
#
# The sources of improvement over scratch: (1) deeper trees (depth=6 vs depth=1) capture complex feature interactions in a single tree, (2) row/column subsampling (80% each) adds stochastic regularization, (3) L1/L2 penalties on leaf weights prevent individual trees from overfitting, and (4) second-order gradient information gives more precise split decisions.

# %%
# === VISUALIZATION 4: XGBoost Learning Curve ===
xgb_results = xgb_model.evals_result()

fig, ax = plt.subplots(figsize=(14, 7))

train_rmse_hist = xgb_results['validation_0']['rmse']
val_rmse_hist = xgb_results['validation_1']['rmse']
rounds_xgb = np.arange(1, len(train_rmse_hist) + 1)

ax.plot(rounds_xgb, train_rmse_hist, color='#2196F3', linewidth=2, label='Train RMSE', alpha=0.9)
ax.plot(rounds_xgb, val_rmse_hist, color='#FF5722', linewidth=2, label='Validation RMSE', alpha=0.9)

best_xgb_round = np.argmin(val_rmse_hist) + 1
best_xgb_val = min(val_rmse_hist)
ax.scatter([best_xgb_round], [best_xgb_val], color='#4CAF50', s=120, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Best: Round {best_xgb_round}\nRMSE: ${best_xgb_val:,.0f}',
            xy=(best_xgb_round, best_xgb_val), xytext=(best_xgb_round + 40, best_xgb_val + 200),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

# Reference lines
ax.axhline(y=scratch_rmse, color='#9C27B0', linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(5, scratch_rmse + 30, f'Scratch GB: ${scratch_rmse:,.0f}', fontsize=11, color='#9C27B0')

ax.set_xlabel('Boosting Round')
ax.set_ylabel('RMSE ($)')
ax.set_title('XGBoost: Learning Curve (500 Rounds, Depth=6)', fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.set_xlim(0, len(rounds_xgb) + 10)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The XGBoost learning curve tells a different story from our scratch implementation:
#
# - **Even faster convergence:** By round 25, XGBoost already beats our 200-stump scratch model (&#36;1,119). Deeper trees capture more information per round.
# - **Classic overfitting signal:** Unlike the scratch model, here the train and validation curves visibly diverge. Train RMSE keeps dropping toward &#36;300, but validation RMSE bottoms out at **&#36;532 (round 142)** and then plateaus/creeps up. This is the regularization-overfitting tradeoff in action.
# - **Early stopping opportunity:** The best validation RMSE is at round 142, not 500. The last 358 trees are wasted compute (or slightly harmful). In production, we would use `early_stopping_rounds` to halt training automatically when validation error stops improving.
#
# The purple reference line shows our scratch model at &#36;1,119. XGBoost cuts through it in just ~25 rounds.

# %%
# === VISUALIZATION 5: Feature Importance ===
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

feature_names = X_train.columns.tolist()

# 1. Gain-based importance (how much each feature improves the loss)
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
gain_vals = [importance_gain.get(name, 0) for name in feature_names]
sorted_idx = np.argsort(gain_vals)
axes[0].barh([feature_names[i] for i in sorted_idx], [gain_vals[i] for i in sorted_idx], color='#2196F3', edgecolor='white')
axes[0].set_title('Gain (Loss Reduction)', fontweight='bold')
axes[0].set_xlabel('Average Gain')

# 2. Weight-based importance (how often each feature is used)
importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
weight_vals = [importance_weight.get(name, 0) for name in feature_names]
sorted_idx_w = np.argsort(weight_vals)
axes[1].barh([feature_names[i] for i in sorted_idx_w], [weight_vals[i] for i in sorted_idx_w], color='#FF9800', edgecolor='white')
axes[1].set_title('Weight (Split Frequency)', fontweight='bold')
axes[1].set_xlabel('Number of Splits')

# 3. Cover-based importance (number of samples affected)
importance_cover = xgb_model.get_booster().get_score(importance_type='cover')
cover_vals = [importance_cover.get(name, 0) for name in feature_names]
sorted_idx_c = np.argsort(cover_vals)
axes[2].barh([feature_names[i] for i in sorted_idx_c], [cover_vals[i] for i in sorted_idx_c], color='#4CAF50', edgecolor='white')
axes[2].set_title('Cover (Samples Affected)', fontweight='bold')
axes[2].set_xlabel('Average Cover')

plt.suptitle('XGBoost Feature Importance (Three Perspectives)', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Three perspectives on feature importance reveal complementary insights:
#
# - **Gain (left):** `y` (length in mm) and `carat` dominate loss reduction. This makes physical sense: diamond dimensions are direct proxies for carat weight. `clarity` and `color` rank third and fourth, capturing the quality premium. The physical dimension features (y, x, z) are highly informative because they encode carat information in a more granular way.
# - **Weight (center):** `carat` has the most splits by count, meaning the trees keep subdividing on carat at finer and finer granularities. `clarity` and `color` also appear frequently, refining the price within each carat band.
# - **Cover (right):** `carat` affects the most samples per split, consistent with it being the primary splitter early in each tree (when all samples are still available).
#
# The takeaway: **carat and physical dimensions (y, x, z) drive the bulk of predictions**, while quality grades (clarity, color, cut) provide the fine-grained adjustments. This aligns perfectly with the domain knowledge: size determines the "ballpark" price, and quality determines where within that range.

# %%
# === VISUALIZATION 6: XGBoost Predicted vs Actual ===
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Left: Scratch GB
hb1 = axes[0].hexbin(y_test, scratch_preds, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
fig.colorbar(hb1, ax=axes[0], label='Count')
lims = [0, max(y_test.max(), max(scratch_preds.max(), xgb_preds.max())) * 1.05]
axes[0].plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--', alpha=0.8)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'Scratch GB (Stumps)\nRMSE: ${scratch_rmse:,.0f} | R2: {scratch_r2:.4f}', fontweight='bold')
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_aspect('equal')

# Right: XGBoost
hb2 = axes[1].hexbin(y_test, xgb_preds, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
fig.colorbar(hb2, ax=axes[1], label='Count')
axes[1].plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--', alpha=0.8)
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].set_title(f'XGBoost (Depth=6)\nRMSE: ${xgb_rmse:,.0f} | R2: {xgb_r2:.4f}', fontweight='bold')
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_aspect('equal')

plt.suptitle('Scratch vs. XGBoost: Prediction Quality', fontsize=18, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# Side-by-side comparison makes the improvement unmistakable:
#
# - **Left (Scratch):** The horizontal banding from stumps is clearly visible. Predictions cluster in discrete bands, and high-price diamonds show significant scatter.
# - **Right (XGBoost):** The predictions form a tight, continuous band along the diagonal. The hexbin density is concentrated on the perfect-prediction line across the entire price range. Even at &#36;15,000+, XGBoost stays close to the diagonal.
#
# The R2 jump from 0.9212 to 0.9814 means XGBoost explains an additional 6% of variance, but that 6% was the hardest 6%: the non-linear interactions, the high-price tail, and the categorical feature effects that stumps could not capture.

# %% [markdown]
# ## 5. LightGBM: Speed Through Histograms
#
# LightGBM (Light Gradient Boosting Machine), developed by Microsoft Research in 2017, introduced two algorithmic innovations that changed the performance-speed tradeoff:
#
# ### 5.1. Key Innovations
#
# **Histogram-based splitting:** Instead of sorting features and evaluating every possible split (O(n * features)), LightGBM buckets continuous values into discrete bins (default 255). Finding the best split becomes O(bins * features), a massive speedup on large datasets.
#
# **Leaf-wise (best-first) tree growth:** While XGBoost grows trees level-by-level (all nodes at depth d before depth d+1), LightGBM picks the leaf with the highest loss reduction and splits that one. This produces asymmetric trees that can model complex patterns with fewer total leaves.
#
# **GOSS (Gradient-based One-Side Sampling):** Keeps all instances with large gradients (hard examples) but randomly samples from instances with small gradients (easy examples). This focuses computation on the data points that matter most for learning.
#
# **EFB (Exclusive Feature Bundling):** Bundles mutually exclusive sparse features into a single feature, reducing effective dimensionality without information loss. Particularly useful for one-hot encoded categoricals.
#
# ### 5.2. Hyperparameters
#
# | Parameter | Value | Rationale |
# |---|---|---|
# | `n_estimators` | 500 | Same budget as XGBoost for fair comparison |
# | `max_depth` | -1 (unlimited) | Leaf-wise growth self-regulates via `num_leaves` |
# | `num_leaves` | 63 | Controls model complexity (2^6 - 1) |
# | `learning_rate` | 0.1 | Same as XGBoost |
# | `subsample` | 0.8 | Row sampling, same as XGBoost |
# | `colsample_bytree` | 0.8 | Column sampling, same as XGBoost |
# | `reg_alpha` | 0.1 | L1 regularization |
# | `reg_lambda` | 1.0 | L2 regularization |
# | `min_child_samples` | 20 | Minimum samples per leaf |

# %%
import lightgbm as lgb
import time

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=-1,
    num_leaves=63,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_callbacks = [lgb.log_evaluation(period=0)]
lgb_evals = {}

start = time.time()
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    callbacks=[
        lgb.log_evaluation(period=0),
        lgb.record_evaluation(lgb_evals)
    ]
)
lgb_time = time.time() - start

lgb_preds = lgb_model.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
lgb_r2 = r2_score(y_test, lgb_preds)
lgb_mae = mean_absolute_error(y_test, lgb_preds)

print("LightGBM Results:")
print(f"  RMSE:  ${lgb_rmse:,.2f}")
print(f"  R2:    {lgb_r2:.4f}")
print(f"  MAE:   ${lgb_mae:,.2f}")
print(f"  Time:  {lgb_time:.2f}s")
print(f"  vs XGBoost RMSE: {(1 - lgb_rmse / xgb_rmse) * 100:+.1f}%")
print(f"  Speedup vs XGBoost: {xgb_time / lgb_time:.1f}x")

# %% [markdown]
# LightGBM achieves RMSE &#36;548, R2 0.9811, MAE &#36;268, essentially matching XGBoost (&#36;543 RMSE) within 1%. The accuracy difference is negligible for practical purposes.
#
# The surprise here is **training time**: LightGBM took 22.8s vs XGBoost's 7.3s, making it ~3x *slower* on this dataset. This is counterintuitive since LightGBM is famous for speed. The explanation: at 54K rows, the dataset is too small for histogram-based splitting to pay off. The overhead of building histograms, GOSS sampling, and EFB bundling exceeds the savings. LightGBM's speed advantage emerges at 100K+ rows, becoming dramatic at millions of rows. On a dataset this size, XGBoost's optimized exact splitting is simply faster.
#
# Key insight: **"faster library" does not mean "faster on your data."** Always benchmark on your actual dataset size.

# %%
# === VISUALIZATION 7: LightGBM Learning Curve ===
lgb_train_rmse = lgb_evals['training']['rmse']
lgb_val_rmse = lgb_evals['valid_1']['rmse']
rounds_lgb = np.arange(1, len(lgb_train_rmse) + 1)

best_lgb_round = np.argmin(lgb_val_rmse) + 1
best_lgb_val = min(lgb_val_rmse)

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(rounds_lgb, lgb_train_rmse, color='#2196F3', linewidth=2, label='Train RMSE', alpha=0.9)
ax.plot(rounds_lgb, lgb_val_rmse, color='#FF5722', linewidth=2, label='Validation RMSE', alpha=0.9)

ax.scatter([best_lgb_round], [best_lgb_val], color='#4CAF50', s=120, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Best: Round {best_lgb_round}\nRMSE: ${best_lgb_val:,.0f}',
            xy=(best_lgb_round, best_lgb_val), xytext=(best_lgb_round + 50, best_lgb_val + 200),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.axhline(y=xgb_rmse, color='#9C27B0', linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(5, xgb_rmse + 30, f'XGBoost final: ${xgb_rmse:,.0f}', fontsize=11, color='#9C27B0')

ax.axhline(y=scratch_rmse, color='#FF9800', linestyle=':', alpha=0.4, linewidth=1.5)
ax.text(5, scratch_rmse + 30, f'Scratch GB: ${scratch_rmse:,.0f}', fontsize=11, color='#FF9800')

ax.set_xlabel('Boosting Round')
ax.set_ylabel('RMSE ($)')
ax.set_title('LightGBM: Learning Curve (Leaf-wise, 500 Rounds)', fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.set_xlim(0, len(rounds_lgb) + 10)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# LightGBM's learning curve reveals a familiar pattern: best validation RMSE of &#36;530 at round 86, with gradual overfitting pushing the final RMSE to &#36;548 by round 500. The train-validation gap is wider than XGBoost's (train RMSE drops to &#36;315 vs validation &#36;548), suggesting that leaf-wise growth's flexibility allows more aggressive fitting of the training data.
#
# Both reference lines (XGBoost at &#36;543, Scratch at &#36;1,119) show that LightGBM's optimal point (round 86) slightly beats XGBoost's optimal (round 142, &#36;532). The fact that both libraries converge to nearly identical validation performance confirms that with similar hyperparameters and sufficient iterations, the choice of splitting strategy (exact vs histogram) has minimal impact on accuracy; it mainly affects training speed.

# %%
# === VISUALIZATION 8: Speed Comparison ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Training time bar chart
models = ['XGBoost', 'LightGBM']
times = [xgb_time, lgb_time]
bar_colors = ['#FF5722', '#4CAF50']

bars = axes[0].bar(models, times, color=bar_colors, edgecolor='white', width=0.5)
for bar, t in zip(bars, times):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{t:.2f}s', ha='center', fontsize=14, fontweight='bold')

axes[0].set_ylabel('Training Time (seconds)')
axes[0].set_title('Training Speed', fontweight='bold')
axes[0].set_ylim(0, max(times) * 1.3)

# Right: RMSE comparison
rmses = [xgb_rmse, lgb_rmse]
bars2 = axes[1].bar(models, rmses, color=bar_colors, edgecolor='white', width=0.5)
for bar, r in zip(bars2, rmses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'${r:,.0f}', ha='center', fontsize=14, fontweight='bold')

axes[1].set_ylabel('RMSE ($)')
axes[1].set_title('Prediction Accuracy', fontweight='bold')
axes[1].set_ylim(0, max(rmses) * 1.3)

plt.suptitle('XGBoost vs. LightGBM: Speed-Accuracy Tradeoff', fontsize=18, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The bar charts tell a clear story: accuracy is virtually identical (&#36;543 vs &#36;548 RMSE, a 0.8% difference), but XGBoost is 3.1x faster on this 54K-row dataset. This is an important lesson in engineering: library benchmarks on millions of rows do not translate to small-to-medium datasets.
#
# When to pick LightGBM over XGBoost: (1) datasets with 100K+ rows where histogram speedup dominates, (2) very high-dimensional data where EFB feature bundling helps, (3) when you need native categorical handling (which we are not using here). On this diamonds dataset, XGBoost wins the speed contest handily.

# %%
# === VISUALIZATION 9: LightGBM vs XGBoost Predicted vs Actual ===
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Left: XGBoost
hb1 = axes[0].hexbin(y_test, xgb_preds, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
fig.colorbar(hb1, ax=axes[0], label='Count')
lims = [0, max(y_test.max(), max(xgb_preds.max(), lgb_preds.max())) * 1.05]
axes[0].plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--', alpha=0.8)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'XGBoost\nRMSE: ${xgb_rmse:,.0f} | R2: {xgb_r2:.4f}', fontweight='bold')
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_aspect('equal')

# Right: LightGBM
hb2 = axes[1].hexbin(y_test, lgb_preds, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
fig.colorbar(hb2, ax=axes[1], label='Count')
axes[1].plot(lims, lims, color='#FF5722', linewidth=2, linestyle='--', alpha=0.8)
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].set_title(f'LightGBM\nRMSE: ${lgb_rmse:,.0f} | R2: {lgb_r2:.4f}', fontweight='bold')
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_aspect('equal')

plt.suptitle('XGBoost vs. LightGBM: Prediction Quality', fontsize=18, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# The hexbin plots are nearly indistinguishable; both models produce a tight, continuous band along the diagonal from &#36;0 to &#36;18,000+. The density patterns are almost identical, confirming that both XGBoost and LightGBM have learned essentially the same function. Minor differences may exist in the high-price tail (&#36;15,000+), but they are invisible at this resolution.
#
# This visual equivalence aligns with the numbers: R2 0.9814 (XGBoost) vs 0.9811 (LightGBM). For this dataset, the choice between XGBoost and LightGBM is a speed question, not an accuracy question.

# %% [markdown]
# ## 6. CatBoost: Ordered Boosting and Native Categoricals
#
# CatBoost (Categorical Boosting), developed by Yandex in 2017, introduced two fundamental innovations that address subtle but important problems in gradient boosting:
#
# ### 6.1. Key Innovations
#
# **Ordered Boosting:** Standard gradient boosting has a hidden data leakage problem: the same data used to estimate gradients is used to fit the next tree. CatBoost addresses this with "ordered boosting"; it uses random permutations of the training data and computes residuals for each example using only the examples that precede it in the permutation. This is conceptually similar to leave-one-out estimation and reduces overfitting, especially on small datasets.
#
# **Oblivious (Symmetric) Decision Trees:** Instead of growing arbitrary trees, CatBoost uses "oblivious" trees where the same splitting condition is applied at each level. A depth-6 oblivious tree has exactly 64 leaves, and every path from root to leaf encounters the same sequence of feature thresholds. This structure is cache-friendly, enables fast inference, and acts as a strong regularizer.
#
# **Native Categorical Handling:** CatBoost can process categorical features directly using target-based statistics with ordered encoding. Instead of one-hot or ordinal encoding, it computes running averages of the target conditioned on each category value, using only "past" observations to avoid leakage. For our diamonds dataset, we already ordinal-encoded the categoricals, so we will not use this feature here, but it is CatBoost's signature capability.
#
# ### 6.2. Hyperparameters
#
# | Parameter | Value | Rationale |
# |---|---|---|
# | `iterations` | 500 | Same budget as XGBoost and LightGBM |
# | `depth` | 6 | Oblivious tree depth (64 leaves, same as XGBoost) |
# | `learning_rate` | 0.1 | Same as other models |
# | `l2_leaf_reg` | 3.0 | L2 regularization (CatBoost default) |
# | `subsample` | 0.8 | Row sampling, same as others |
# | `rsm` | 0.8 | Random subspace method (column sampling) |
# | `bootstrap_type` | 'Bernoulli' | Required when subsample < 1.0 |

# %%
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.8,
    bootstrap_type='Bernoulli',
    random_seed=42,
    verbose=0
)

start = time.time()
cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=0
)
cat_time = time.time() - start

cat_preds = cat_model.predict(X_test)
cat_rmse = np.sqrt(mean_squared_error(y_test, cat_preds))
cat_r2 = r2_score(y_test, cat_preds)
cat_mae = mean_absolute_error(y_test, cat_preds)

# Extract learning curve
cat_evals = cat_model.get_evals_result()
cat_train_rmse = cat_evals['learn']['RMSE']
cat_val_rmse = cat_evals['validation']['RMSE']
best_cat_round = np.argmin(cat_val_rmse) + 1
best_cat_val = min(cat_val_rmse)

print("CatBoost Results:")
print(f"  RMSE:  ${cat_rmse:,.2f}")
print(f"  R2:    {cat_r2:.4f}")
print(f"  MAE:   ${cat_mae:,.2f}")
print(f"  Time:  {cat_time:.2f}s")
print(f"  Best round: {best_cat_round}/500 (val RMSE: ${best_cat_val:,.0f})")
print(f"  vs XGBoost:  {(1 - cat_rmse / xgb_rmse) * 100:+.1f}%")
print(f"  vs LightGBM: {(1 - cat_rmse / lgb_rmse) * 100:+.1f}%")

# %% [markdown]
# CatBoost delivers RMSE &#36;530, R2 0.9823, MAE &#36;282, the best accuracy of the three libraries, beating XGBoost (&#36;543) by 2.4% and LightGBM (&#36;548) by 3.2%. Even more impressive: it trained in just 5.2 seconds, making it the fastest model as well.
#
# Why does CatBoost win here? Two factors: (1) **ordered boosting** reduces overfitting on this medium-sized dataset, allowing the model to extract more signal before the train-validation gap opens up. (2) **Oblivious trees** act as a structural regularizer; the symmetric constraint prevents the tree from over-specializing on specific subgroups, forcing it to learn more generalizable splits.
#
# Best round: 492/500, meaning CatBoost barely overfits at all. The validation RMSE is essentially flat from round 300 onward, showing remarkable stability. Compare this to XGBoost (best at round 142) and LightGBM (best at round 86), which both started degrading much earlier.

# %%
# === VISUALIZATION 10: CatBoost Learning Curve ===
fig, ax = plt.subplots(figsize=(14, 7))

rounds_cat = np.arange(1, len(cat_train_rmse) + 1)

ax.plot(rounds_cat, cat_train_rmse, color='#2196F3', linewidth=2, label='Train RMSE', alpha=0.9)
ax.plot(rounds_cat, cat_val_rmse, color='#FF5722', linewidth=2, label='Validation RMSE', alpha=0.9)

ax.scatter([best_cat_round], [best_cat_val], color='#4CAF50', s=120, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Best: Round {best_cat_round}\nRMSE: ${best_cat_val:,.0f}',
            xy=(best_cat_round, best_cat_val), xytext=(best_cat_round - 200, best_cat_val + 150),
            fontsize=12, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.axhline(y=xgb_rmse, color='#9C27B0', linestyle=':', alpha=0.6, linewidth=1.5)
ax.text(5, xgb_rmse - 35, f'XGBoost final: ${xgb_rmse:,.0f}', fontsize=11, color='#9C27B0')

ax.axhline(y=lgb_rmse, color='#FF9800', linestyle=':', alpha=0.4, linewidth=1.5)
ax.text(5, lgb_rmse + 20, f'LightGBM final: ${lgb_rmse:,.0f}', fontsize=11, color='#FF9800')

ax.set_xlabel('Boosting Round')
ax.set_ylabel('RMSE ($)')
ax.set_title('CatBoost: Learning Curve (Oblivious Trees, 500 Rounds)', fontweight='bold')
ax.legend(fontsize=13, loc='upper right')
ax.set_xlim(0, len(rounds_cat) + 10)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# CatBoost's learning curve is the most disciplined of the three libraries. The train-validation gap stays narrow throughout: train RMSE settles at &#36;455 vs validation &#36;530, a much tighter gap than LightGBM's (&#36;315 vs &#36;548). The validation curve barely increases after round 300, confirming that ordered boosting and symmetric trees provide strong implicit regularization.
#
# Both reference lines (XGBoost at &#36;543, LightGBM at &#36;548) sit above CatBoost's validation curve for the entire second half of training. CatBoost achieves its best at round 492, nearly the full 500 iterations, meaning it could potentially benefit from even more rounds, whereas XGBoost and LightGBM were already overfitting long before round 500.

# %%
# === VISUALIZATION 11: Three-Library Predicted vs Actual ===
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

all_preds = [xgb_preds, lgb_preds, cat_preds]
all_names = ['XGBoost', 'LightGBM', 'CatBoost']
all_rmses = [xgb_rmse, lgb_rmse, cat_rmse]
all_r2s = [xgb_r2, lgb_r2, cat_r2]
all_colors = ['#FF5722', '#4CAF50', '#2196F3']

lims = [0, max(y_test.max(), max(p.max() for p in all_preds)) * 1.05]

for i, (name, pred, r, r2, col) in enumerate(zip(all_names, all_preds, all_rmses, all_r2s, all_colors)):
    hb = axes[i].hexbin(y_test, pred, gridsize=60, cmap='viridis', mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=axes[i], label='Count')
    axes[i].plot(lims, lims, color=col, linewidth=2, linestyle='--', alpha=0.8)
    axes[i].set_xlabel('Actual Price ($)')
    axes[i].set_ylabel('Predicted Price ($)')
    axes[i].set_title(f'{name}\nRMSE: ${r:,.0f} | R2: {r2:.4f}', fontweight='bold')
    axes[i].set_xlim(lims)
    axes[i].set_ylim(lims)
    axes[i].set_aspect('equal')

plt.suptitle('XGBoost vs. LightGBM vs. CatBoost: Prediction Quality', fontsize=18, fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# All three libraries produce visually similar hexbin patterns; tight bands along the diagonal from &#36;0 to &#36;18,000+. At this resolution, the differences are subtle, but they exist: CatBoost (right) has slightly tighter concentration in the &#36;5,000-&#36;10,000 range, where the bulk of prediction errors occur.
#
# The numbers confirm: CatBoost R2 0.9823 > XGBoost 0.9814 > LightGBM 0.9811. The margin is small (0.1% R2 difference), but CatBoost achieves this with the fastest training time (5.2s) and the least overfitting. On this dataset, CatBoost is the Pareto-optimal choice: best accuracy, best speed, most stable learning curve.

# %% [markdown]
# ## 7. Grand Comparison
#
# We have now trained four gradient boosting models on the same diamonds dataset, from a 200-stump scratch implementation to three production-grade libraries. Let us bring all the results together.

# %%
# === VISUALIZATION 12: Grand Comparison Dashboard ===
import pandas as pd

results = pd.DataFrame({
    'Model': ['Baseline (Mean)', 'Scratch GB', 'XGBoost', 'LightGBM', 'CatBoost'],
    'RMSE ($)': [baseline_rmse, scratch_rmse, xgb_rmse, lgb_rmse, cat_rmse],
    'R2': [0.0, scratch_r2, xgb_r2, lgb_r2, cat_r2],
    'MAE ($)': [np.nan, np.nan, xgb_mae, lgb_mae, cat_mae],
    'Training Time (s)': [0.0, np.nan, xgb_time, lgb_time, cat_time],
    'Best Round': [np.nan, 200, best_xgb_round, best_lgb_round, best_cat_round],
    'Total Rounds': [np.nan, 200, 500, 500, 500]
})

results['RMSE Reduction vs Baseline'] = (1 - results['RMSE ($)'] / baseline_rmse) * 100

print(results.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# %% [markdown]
# The results table tells the full story of this notebook's journey:
#
# | Model | RMSE | R2 | Time | RMSE Reduction |
# |---|---|---|---|---|
# | Baseline (Mean) | &#36;3,987 | 0.00 | N/A | 0% |
# | Scratch GB (200 stumps) | &#36;1,119 | 0.9212 | N/A | 71.9% |
# | XGBoost (500 trees) | &#36;543 | 0.9814 | 7.3s | 86.4% |
# | LightGBM (500 trees) | &#36;548 | 0.9811 | 22.8s | 86.3% |
# | **CatBoost (500 trees)** | **&#36;530** | **0.9823** | **5.2s** | **86.7%** |
#
# The first big jump (baseline to scratch: 72% RMSE reduction) comes from the algorithm itself; even 200 stumps with learning rate 0.1 capture the bulk of the signal. The second jump (scratch to libraries: another 14-15 percentage points) comes from engineering: deeper trees, regularization, second-order gradients, and optimized splitting.

# %%
# === VISUALIZATION 13: Final Dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

model_names = ['Scratch GB', 'XGBoost', 'LightGBM', 'CatBoost']
model_rmses = [scratch_rmse, xgb_rmse, lgb_rmse, cat_rmse]
model_r2s = [scratch_r2, xgb_r2, lgb_r2, cat_r2]
model_colors = ['#FF9800', '#FF5722', '#4CAF50', '#2196F3']
model_times = [np.nan, xgb_time, lgb_time, cat_time]

# --- Panel 1: RMSE Comparison ---
bars = axes[0, 0].bar(model_names, model_rmses, color=model_colors, edgecolor='white', width=0.6)
for bar, r in zip(bars, model_rmses):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'${r:,.0f}', ha='center', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('RMSE ($)')
axes[0, 0].set_title('RMSE: Lower is Better', fontweight='bold', fontsize=14)
axes[0, 0].set_ylim(0, max(model_rmses) * 1.25)

# --- Panel 2: R2 Comparison ---
bars2 = axes[0, 1].bar(model_names, model_r2s, color=model_colors, edgecolor='white', width=0.6)
for bar, r2 in zip(bars2, model_r2s):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{r2:.4f}', ha='center', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('R2 Score')
axes[0, 1].set_title('R2: Higher is Better', fontweight='bold', fontsize=14)
axes[0, 1].set_ylim(0.90, 1.0)

# --- Panel 3: Training Time (library models only) ---
lib_names = ['XGBoost', 'LightGBM', 'CatBoost']
lib_times = [xgb_time, lgb_time, cat_time]
lib_colors = ['#FF5722', '#4CAF50', '#2196F3']
bars3 = axes[1, 0].bar(lib_names, lib_times, color=lib_colors, edgecolor='white', width=0.5)
for bar, t in zip(bars3, lib_times):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{t:.1f}s', ha='center', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].set_title('Training Speed: Lower is Better', fontweight='bold', fontsize=14)
axes[1, 0].set_ylim(0, max(lib_times) * 1.3)

# --- Panel 4: Learning Curves Overlay ---
rounds_xgb_lc = np.arange(1, len(val_rmse_hist) + 1)
rounds_lgb_lc = np.arange(1, len(lgb_val_rmse) + 1)
rounds_cat_lc = np.arange(1, len(cat_val_rmse) + 1)

axes[1, 1].plot(rounds_xgb_lc, val_rmse_hist, color='#FF5722', linewidth=2, label='XGBoost', alpha=0.8)
axes[1, 1].plot(rounds_lgb_lc, lgb_val_rmse, color='#4CAF50', linewidth=2, label='LightGBM', alpha=0.8)
axes[1, 1].plot(rounds_cat_lc, cat_val_rmse, color='#2196F3', linewidth=2, label='CatBoost', alpha=0.8)
axes[1, 1].set_xlabel('Boosting Round')
axes[1, 1].set_ylabel('Validation RMSE ($)')
axes[1, 1].set_title('Validation Curves: All Libraries', fontweight='bold', fontsize=14)
axes[1, 1].legend(fontsize=12, loc='upper right')
axes[1, 1].set_xlim(0, 510)
axes[1, 1].set_ylim(400, 800)

plt.suptitle('Gradient Boosting: From Scratch to State-of-the-Art', fontsize=20, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# %% [markdown]
# The four-panel dashboard summarizes everything:
#
# - **RMSE (top-left):** The dramatic drop from Scratch (&#36;1,119) to the three libraries (&#36;530-548) is immediately visible.
# - **R2 (top-right):** Zoomed to the 0.90-1.00 range, we see the Scratch model at 0.9212 and the three libraries clustered tightly around 0.98. CatBoost's 0.9823 leads by a small but consistent margin.
# - **Training time (bottom-left):** CatBoost (5.2s) and XGBoost (7.3s) are practical for rapid iteration. LightGBM's 22.8s on this small dataset is a reminder that "fastest library" depends on data size.
# - **Learning curves (bottom-right):** The overlay reveals distinct overfitting behaviors. XGBoost and LightGBM reach their best validation RMSE early (rounds 142 and 86) then slowly degrade. CatBoost's validation curve is remarkably flat from round 200 onward, sitting below both competitors.
#
# *Note: training times may vary slightly between runs depending on system load, CPU thermal state, and background processes. The relative ranking is consistent, but exact seconds will differ.*

# %% [markdown]
# ## 8. Conclusion
#
# ### What We Learned
#
# 1. **The algorithm is simple, the engineering is not.** Gradient boosting at its core is just "fit residuals, add corrections, repeat." Our scratch implementation in ~80 lines of code achieved R2 0.92, competitive with many classical methods. But the jump from 0.92 to 0.98 required sophisticated engineering: regularization, second-order gradients, histogram binning, ordered boosting.
#
# 2. **Library choice depends on your data.** On this 54K-row dataset, CatBoost was both fastest and most accurate. But on datasets with millions of rows, LightGBM's histogram-based approach would dominate. On datasets with many categorical features, CatBoost's native handling would shine even more. There is no universal "best" library.
#
# 3. **Overfitting is the central challenge.** Every model showed train-validation divergence. XGBoost peaked at round 142/500, LightGBM at 86/500. CatBoost's ordered boosting delayed this to round 492/500. In production, early stopping is not optional; it is essential.
#
# 4. **Feature importance is multi-dimensional.** XGBoost's three importance metrics (gain, weight, cover) tell different stories. Physical dimensions (y, carat) dominate gain; depth has the most splits by count. No single metric captures the full picture.
#
# ### When to Use Gradient Boosting
#
# Gradient boosting is the default choice for **tabular data** with structured features. It consistently wins or ties deep learning on Kaggle competitions, insurance pricing, credit scoring, medical diagnosis, and virtually any task where the input is a table of numbers and categories. It struggles with images, text, and sequential data; those domains belong to CNNs, transformers, and RNNs.
#
# ### Production Checklist
#
# - Always use **early stopping** (monitor validation loss, stop when it plateaus).
# - **Tune learning rate and n_estimators together** (lower LR = more trees needed).
# - Use **cross-validation** for hyperparameter tuning, not a single train/test split.
# - Consider **feature engineering** (interaction terms, log transforms); GB finds splits, but it cannot create new features.
# - **Monitor prediction latency**; 500 trees with depth 6 means 500 sequential evaluations at inference time. For real-time applications, consider pruning or distillation.
