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
# # Handling Real-World Mess: Imbalanced Learning
#
# ## 1. Theoretical Foundation
#
# In every notebook so far, we have worked with datasets where the classes are either balanced or relatively close to balanced. The Wine dataset had three cultivar types spread roughly evenly. The Breast Cancer dataset was about 60/40. Titanic survivors were roughly 38% of the population. This is a luxury that only exists in curated academic benchmarks.
#
# The real world is violently asymmetrical. Consider some concrete examples:
# - **Credit Card Fraud:** Visa processes hundreds of millions of transactions per day. Fraud represents roughly $0.1\%$ of those. For every fraudulent charge, there are approximately **999 legitimate ones**.
# - **Medical Diagnosis (Cancer Screening):** In a routine mammogram screening, the malignancy rate is typically between $0.5\%$ and $1\%$ of scans. The overwhelming majority of patients are healthy.
# - **Predictive Maintenance in Manufacturing:** In automotive assembly plants, defective components represent anywhere from $1\%$ to $3\%$ of total output. The rest pass quality control.
# - **Wildfire and Rare Event Detection:** Detecting a rare geological event or an anomalous power surge in grid data is a search for a needle in a haystack measured in millions of data points.
#
# **The fundamental problem** is not that these datasets are hard to classify. The problem is that when we train standard algorithms on this kind of data, they discover a catastrophic mathematical shortcut called **The "Predict Majority" Trap.**
#
# A model that hardcodes itself to always predict "Legitimate Transaction" without ever looking at any feature will still achieve **$99.9\%$ global Accuracy**. An executive reading that number might celebrate, completely unaware that the instrument has a $0\%$ success rate at catching fraud; the only task it was deployed to solve. Meanwhile, real money is being stolen.
#
# The model has not computed anything. It has simply discovered the cheapest mathematical cheat: *"If I always say the majority class, I will be wrong only 0.1% of the time and no standard loss function will punish me for it."*
#
# ### 1.1 Beyond Accuracy (The Alternative Metrics)
#
# To build classifiers that actually solve the problem, we must abandon global Accuracy and reason about the **Confusion Matrix** with precision-recall language:
#
# |  | Predicted: Majority | Predicted: Minority |
# |---|---|---|
# | **True Majority** | True Negative (TN) | False Positive (FP) |
# | **True Minority** | **False Negative (FN)** | True Positive (TP) |
#
# The cell that destroys everything is the **False Negative (FN)**: cases where the true answer was "Fraud", "Cancer", "Defect"... and the model said "Normal". In high-stakes settings, a single FN can represent a catastrophic failure.
#
# - **Recall (Sensitivity):** $\frac{TP}{TP + FN}$. Out of all the actual minority events that happened, what percentage did we catch? This is the metric that directly measures how many disasters we let slip through the net.
# - **Precision:** $\frac{TP}{TP + FP}$. Out of every prediction we made as "Minority", how many were correct? Low precision means we annoyed customers with false alarms.
# - **F1-Score:** $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$. The harmonic mean. It punishes extreme cases where you achieve high Recall by flagging everything, or high Precision by only flagging a tiny certainty set.
# - **AUPRC (Area Under the Precision-Recall Curve):** The global ranking metric for imbalanced data. It evaluates how well the model's predicted probabilities separate minority from majority across *all* possible classification thresholds, not just 0.50. The higher, the better.
#
# ### 1.2 Algorithmic Interventions
#
# Once we diagnose the problem, we have three families of solutions to force a model to take the minority class seriously:
#
# 1. **Class Weighting (Cost-Sensitive Learning):** We do not touch the dataset at all. Instead, we mathematically rig the Loss Function. We declare: *"Misclassifying a 'Fraud' instance costs 100x more than misclassifying a 'Legitimate' one."* The optimizer is forced to route more of its gradient descent energy into correctly classifying the rare class. This is available natively in Scikit-Learn's `class_weight='balanced'` parameter.
#
# 2. **Undersampling:** We physically delete rows from the majority class until the dataset is balanced. Simple and fast, but potentially catastrophic: we might be throwing away the exact majority-class boundary data that the model needs to draw a clean separation line. With a 99/1 ratio, we would destroy 99% of our original data.
#
# 3. **Oversampling & SMOTE:** We physically augment the minority class. Pure random oversampling just duplicates existing minority rows, which risks severe overfitting. SMOTE (Synthetic Minority Over-sampling Technique) goes further: it synthesizes entirely new, geometrically plausible minority examples by interpolating between existing rare points in feature space.
#
# ### 1.3 The Roadmap
#
# 1. **Data:** We will use the **Forest Covertypes** dataset, isolating an extreme $99\%$ vs $1\%$ class skew to simulate finding a rare tree type in an ocean of pines.
# 2. **Baseline Failure:** We train a standard Logistic Regression with no modifications and inspect exactly which minority predictions it misses and why.
# 3. **Scratch:** We implement Random Undersampling, Random Oversampling, and SMOTE entirely from scratch using only NumPy, so we understand exactly how training matrices are being physically altered.
# 4. **Evaluation:** We retrain on all variants, compare Confusion Matrices side by side, and then move to the Pro tooling and the AUPRC metric to understand the full precision-recall trade-off.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standard ATLAS styling
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15

print("Libraries imported and styling set.")

# %% [markdown]
# ## 2. Data Preparation: The Rare Forest
#
# We will use the **Forest Covertype** dataset (`fetch_covtype`). It contains $581,012$ plots of US forest land, each described by $54$ cartographic attributes: elevation, slope angle, aspect (direction the slope faces), distance to the nearest road and water body, soil type, and wilderness area designation.
#
# The task is to predict the dominant tree cover type in each plot. There are 7 cover types in total. For this experiment, we will isolate just two:
#
# - **Majority Class (0):** Lodgepole Pine *(Class 2 in the original data)*; The dominant species across most of the Rocky Mountain region. The dataset has approximately 283,000 samples of this type.
# - **Minority Class (1):** Cottonwood/Willow *(Class 4 in the original data)*; A rare riverine tree type that grows only in low-elevation valleys near permanent water. The dataset has approximately 2,700 samples of this type.
#
# This is a genuine $\sim 99\%$ vs $\sim 1\%$ imbalance ratio, not engineered: it reflects how these species actually exist in the Rocky Mountain ecosystem.
#
# For computational efficiency, we subsample 50,000 plots **using stratified sampling to perfectly preserve the original 99/1 ratio**. This is critical: if we randomly sampled without stratification, we might accidentally eliminate most of the ~500 rare willow plots and make the problem even harder than it already is.

# %%
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

# Load dataset
cov_type = fetch_covtype()
X_full = cov_type.data
y_full = cov_type.target

# Class 2: Lodgepole Pine (majority)
# Class 4: Cottonwood/Willow (minority)
mask_majority = (y_full == 2)
mask_minority = (y_full == 4)

final_mask = mask_majority | mask_minority
X_imb = X_full[final_mask]
y_imb_raw = y_full[final_mask]

# Re-map targets for binary classification: 0 = Pine (majority), 1 = Willow (minority)
y_imb = np.where(y_imb_raw == 4, 1, 0)

# Stratified subsample: keep exactly 50,000 rows at the same 99/1 ratio
_, X_work, _, y_work = train_test_split(
    X_imb,
    y_imb,
    test_size=50000,
    random_state=42,
    stratify=y_imb
)

# Final train/test split; ALSO stratified to preserve the ratio in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X_work,
    y_work,
    test_size=0.2,
    random_state=42,
    stratify=y_work
)

print(f"Working Dataset Size: {X_work.shape[0]} rows")
target_counts = pd.Series(y_work).value_counts(normalize=True).sort_index() * 100
print(f"Majority Class 0; Lodgepole Pine: {target_counts[0]:.2f}%")
print(f"Minority Class 1; Cottonwood/Willow: {target_counts[1]:.2f}%")
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Minority instances in training set: {y_train.sum()}")
print(f"Minority instances in test set:     {y_test.sum()}")

# %% [markdown]
# The output above confirms our imbalance: the Cottonwood/Willow accounts for under $1\%$ of the working data. Looking at the raw counts, there are only around **480 rare trees in the entire 40,000-row training set** to learn from. The remaining ~39,500 are Pines. This is the exact ratio severity that destroys standard algorithms.
#
# Notice also the near-identical ratio between training and test sets; that is stratification working as intended. If we had done a simple random split, we might have ended up with wildly different minority proportions in each split, which would introduce variance into the evaluation itself.
#
# ## 3. The Baseline Failure
#
# Before we apply any correction, we must first understand exactly *where* and *how* a standard, uncorrected Logistic Regression fails. This is not a straw-man exercise. Logistic Regression is one of the most powerful and widely deployed algorithms in production ML. It is the backbone of credit scoring, medical risk models, and A/B testing systems. Understanding its failure mode in imbalanced settings is essential for every practitioner.
#
# Without any class weighting or resampling, Logistic Regression minimizes the **log-loss** (Binary Cross-Entropy) summed over all training instances:
#
# $$ \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right] $$
#
# When $99\%$ of our $y_i$ are $0$ (Pine), the gradient descent spends $99\%$ of every update step learning to classify Pines correctly. The 480 Willow rows are mathematically drowned out. The model produces very low predicted probabilities for the minority class, and when we apply the default $0.50$ classification threshold, those low probabilities map to "Pine" predictions every single time.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base)
tn_base, fp_base, fn_base, tp_base = cm_base.ravel()
recall_base_min = tp_base / (tp_base + fn_base)
precision_base_min = tp_base / (tp_base + fp_base)

print(f"Baseline Accuracy: {acc_base * 100:.2f}%")
print(f"\nConfusion Matrix Breakdown:")
print(f"  TN (Pine predicted as Pine):     {tn_base:>5}")
print(f"  FP (Pine predicted as Willow):   {fp_base:>5}  ← False Alarm")
print(f"  FN (Willow predicted as Pine):   {fn_base:>5}  ← MISSED DISASTER")
print(f"  TP (Willow predicted as Willow): {tp_base:>5}  ← Correct Catch!")
print(f"\nMinority class Precision: {precision_base_min:.4f}")
print(f"Minority class Recall:    {recall_base_min:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm_base,
    annot=True,
    fmt='d',
    cmap='Reds',
    cbar=False,
    xticklabels=['Majority (Pine)', 'Minority (Willow)'],
    yticklabels=['Majority (Pine)', 'Minority (Willow)'],
    ax=ax
)
ax.set_title('Confusion Matrix; Baseline Logistic Regression\n(No class weights, no resampling)', fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()

# %% [markdown]
# Inspect the output carefully. The Logistic Regression achieved $\sim 99.9\%$ global Accuracy. But this dataset is geometrically fortunate: the Cottonwood/Willow tree grows in very specific low-altitude, high-moisture zones that are genuinely different from the high-altitude Pine habitat. So even a raw, uncorrected model manages to catch most of the rare trees.
#
# **This is actually the most dangerous scenario in real-world ML.** The model looks good; it has high precision and high recall on the minority class. The temptation is to deploy it immediately. But look at the **False Negatives (bottom-left cell)**. Those are real Willow trees that the model declared to be Pine. Every one of them is a miss.
#
# Now translate this to the real-world scenarios:
# - **Fraud:** Those 8 FNs are 8 actual fraudulent transactions that your system approved. Each could be a $10,000 wire transfer.
# - **Cancer Screening:** Those 8 FNs are 8 patients with malignant tumors who were told "everything looks fine, come back in a year." Some of those patients will not come back.
# - **Predictive Maintenance:** Those 8 FNs are 8 defective components that passed quality control and made it onto the assembly line, potentially causing product recalls.
#
# The goal of everything that follows in this notebook is simple: **Eradicate those False Negatives**. And we will do so even at the cost of introducing more False Positives; because in high-stakes domains, a False Positive (a false alarm) is an inconvenience, but a False Negative is a catastrophe.
#
# ## 4. Core Sampling Techniques from Scratch
#
# Before reaching for specialized libraries, we will build the three foundational data balancing techniques from scratch using pure NumPy. This is not an academic exercise; understanding what happens to the training matrix at the row-by-row level is what allows you to reason about *why* each technique succeeds and fails in different scenarios.
#
# All three samplers below follow the same contract: they take `X_train` and `y_train` and return a new `X_resampled` and `y_resampled` with a perfectly balanced 50/50 class ratio. **The test set `(X_test, y_test)` is never touched.** It remains a pure, faithful representation of the real-world imbalanced environment. Evaluating on a resampled test set would be cheating; we would be grading the model on a fake reality.
#
# ### 4.1 Random Undersampling
#
# **The Idea:** The Majority class has ~39,500 rows. The Minority class has ~480 rows. Undersampling discards the excess majority rows; permanently. We randomly select 480 rows from the Pine pool and throw the remaining ~39,000 away. Now both classes have exactly 480 examples.
#
# **The Trade-off:** This is computationally instant. The model trains on a tiny dataset and converges in seconds. But we just destroyed $98.8\%$ of our training information. Think of all the edge cases, the boundary Pines living near Willow zones, the rare high-elevation Pines; they are all gone. The model's decision boundary for the majority class becomes extremely fragile.
#
# ### 4.2 Random Oversampling
#
# **The Idea:** Instead of deleting majority data, we repeat minority data. We randomly duplicate Willow rows (with replacement) until we have ~39,500 of them, matching the Pine count. Now both classes have ~39,500 examples.
#
# **The Trade-off:** We retain all the Pine geometry (good). But the 480 original Willow rows are now cloned ~82 times each. Every clone is bitwise identical to an original. The model is just memorizing those 480 points over and over again. This causes **severe overfitting** on the minority class: the model draws an extremely tight decision boundary around those 480 exact points, which may not generalize to new Willows that are even slightly different.

# %%
class ScratchSampler:
    """Base class exposing the _split_classes helper for both samplers."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def _split_classes(self, X, y):
        """Return indices and labels for majority and minority classes."""
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        idx_maj = np.where(y == majority_class)[0]
        idx_min = np.where(y == minority_class)[0]
        return idx_maj, idx_min, majority_class, minority_class


class ScratchRandomUndersampler(ScratchSampler):
    """Randomly discard majority rows until classes are equal in size."""

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        idx_maj, idx_min, _, _ = self._split_classes(X, y)

        # Pick exactly len(minority) random majority indices, without replacement
        # This guarantees no duplicate majority rows in the result
        undersampled_maj_idx = np.random.choice(
            idx_maj,
            size=len(idx_min),
            replace=False
        )

        final_idx = np.concatenate([undersampled_maj_idx, idx_min])
        np.random.shuffle(final_idx)   # Shuffle so the model doesn't see class blocks
        return X[final_idx], y[final_idx]


class ScratchRandomOversampler(ScratchSampler):
    """Randomly duplicate minority rows until classes are equal in size."""

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        idx_maj, idx_min, _, _ = self._split_classes(X, y)

        # We need (n_majority - n_minority) extra minority rows
        n_needed = len(idx_maj) - len(idx_min)

        # Sample minority indices WITH replacement; this creates the duplicates
        oversampled_min_idx = np.random.choice(
            idx_min,
            size=n_needed,
            replace=True     # replace=True is what makes this "oversampling"
        )

        # Stack original minority + synthetic duplicates
        final_min_idx = np.concatenate([idx_min, oversampled_min_idx])
        final_idx = np.concatenate([idx_maj, final_min_idx])
        np.random.shuffle(final_idx)
        return X[final_idx], y[final_idx]


# Execute both samplers on the training data
X_train_under, y_train_under = ScratchRandomUndersampler().fit_resample(X_train, y_train)
X_train_over, y_train_over   = ScratchRandomOversampler().fit_resample(X_train, y_train)

print("Training set sizes after resampling:")
print(f"  Original shape:    {X_train.shape}  →  {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Undersampled:      {X_train_under.shape}  →  {dict(zip(*np.unique(y_train_under, return_counts=True)))}")
print(f"  Oversampled:       {X_train_over.shape}  →  {dict(zip(*np.unique(y_train_over, return_counts=True)))}")

# %% [markdown]
# The output is striking. Look at the three sizes:
#
# - **Original:** ~40,000 rows. 480 Willows, 39,500 Pines.
# - **Undersampled:** ~960 rows. 480 Willows, 480 Pines. ← We deleted **98.8%** of our data.
# - **Oversampled:** ~79,000 rows. ~39,500 Willows (clones!), 39,500 Pines. ← We inflated the dataset with **~39,000 duplicated rows**.
#
# Neither solution is elegant. Undersampling amputates the dataset. Oversampling pads it with lies. This motivates SMOTE.
#
# ### 4.3 SMOTE: Synthetic Minority Interpolation
#
# SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al., 2002) is a fundamentally more intelligent approach. Instead of blindly cloning existing minority examples, it **manufactures new, geometrically plausible minority instances** by interpolating between existing ones.
#
# **The Geometric Intuition:** Imagine the 480 Willow plots as 480 points floating in a 54-dimensional feature space. SMOTE assumes that the space *between* nearby Willow points is also likely to be Willow territory (a reasonable assumption: if two valley Willow plots are geographically and compositionally similar, points between them are probably also Willow habitat). It draws straight lines connecting neighboring points, and places new synthetic Willows at random positions along those lines.
#
# **The Algorithm, Step by Step:**
# 1. Isolate the 480 minority (Willow) points: $X_{min}$.
# 2. For each point in $X_{min}$, find its $K$ nearest minority neighbors in 54-dimensional Euclidean space.
# 3. Randomly pick one of those neighbors.
# 4. Parameterize a line between the base point and the neighbor: $\text{new\_point} = \text{base} + \lambda \cdot (\text{neighbor} - \text{base})$, where $\lambda \sim \text{Uniform}(0, 1)$.
# 5. That new coordinate vector is a brand new synthetic Willow plot. Append it to the dataset.
# 6. Repeat until you have manufactured ~39,000 synthetic Willows (enough to match the Pines).
#
# The result is a dataset of the same size as Oversampling, but populated with unique, geometrically distributed synthetic points rather than exact duplicates. The model must learn a broader, smoother decision boundary around the Willow region rather than memorizing specific points.

# %%
from sklearn.neighbors import NearestNeighbors


class ScratchSMOTE:
    """
    Implement SMOTE from scratch using NumPy and nearest neighbors.

    For each of the (n_majority - n_minority) synthetic samples needed,
    we pick a random minority point, find its K nearest minority neighbors,
    and interpolate between the point and one of those neighbors to create
    a new synthetic feature vector.
    """

    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        X_min = X[y == minority_class]      # All minority rows
        X_maj = X[y == majority_class]      # All majority rows

        n_majority = len(X_maj)
        n_minority = len(X_min)
        n_synthetic_needed = n_majority - n_minority

        print(f"Minority class size: {n_minority} | Majority class size: {n_majority}")
        print(f"Generating {n_synthetic_needed} synthetic minority samples...")

        if n_minority < 2:
            raise ValueError("SMOTE requires at least 2 minority samples to interpolate.")

        # Adapt k if minority is smaller than requested (edge case guard)
        effective_k = min(self.k_neighbors, n_minority - 1)

        # Fit KNN ONLY on minority points; we search for minority neighbors
        # We ask for k+1 because kneighbors always includes the query point itself
        # as the closest neighbor (distance = 0), so we take k+1 and drop the first.
        nn = NearestNeighbors(n_neighbors=effective_k + 1)
        nn.fit(X_min)

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            # Step 1: Pick a random minority base point
            idx = np.random.randint(0, n_minority)
            base_point = X_min[idx]

            # Step 2: Find its K nearest minority neighbors
            _, neighbor_indices = nn.kneighbors([base_point])
            # neighbor_indices[0][0] is the point itself; skip it
            valid_neighbors = neighbor_indices[0][1:]

            # Step 3: Randomly select one of the K neighbors
            chosen_neighbor_idx = np.random.choice(valid_neighbors)
            neighbor_point = X_min[chosen_neighbor_idx]

            # Step 4: Interpolate; pick a random point on the line segment
            difference = neighbor_point - base_point
            random_gap = np.random.uniform(0, 1)   # λ ∈ [0, 1]
            synthetic_point = base_point + (random_gap * difference)
            synthetic_samples.append(synthetic_point)

        synthetic_samples = np.array(synthetic_samples)

        # Assemble the balanced dataset:
        # - All majority rows (unchanged)
        # - All original minority rows (unchanged)
        # - All synthetic minority rows (newly generated)
        X_resampled = np.vstack([X_maj, X_min, synthetic_samples])
        y_synthetic = np.full(n_synthetic_needed, minority_class)
        y_resampled = np.concatenate([
            y[y == majority_class],
            y[y == minority_class],
            y_synthetic
        ])

        return X_resampled, y_resampled


smote = ScratchSMOTE(k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("-" * 55)
print(f"SMOTE Output - X shape: {X_train_smote.shape}, y shape: {y_train_smote.shape}")
dist = pd.Series(y_train_smote).value_counts(normalize=True).sort_index() * 100
print(f"Class distribution after SMOTE: Pine {dist[0]:.1f}% | Willow {dist[1]:.1f}%")

# %% [markdown]
# SMOTE has generated approximately **39,000 brand new synthetic Willow plots** by walking along the geometric corridors between real Willow data points in the 54-dimensional feature space. The final training set is perfectly balanced at 50/50, with exactly the same total size as Random Oversampling; but every single minority row is geometrically unique.
#
# This is the critical difference:
# - **Random Oversampling:** Willow row #47 appears 80 times. The model just memorizes it.
# - **SMOTE:** SMOTE generates 80 *different* synthetic Willows near row #47's neighborhood. The model learns a broader, more generalizable Willow territory.
#
# ### 4.4 Evaluating the Three Scratch Techniques
#
# Now comes the moment of truth. We retrain **the exact same Logistic Regression** on each of the three balanced datasets, then evaluate all four models (including the original baseline) on the **exact same untouched test set**.
#
# > **Why does the total count in the Confusion Matrix stay the same across all models?**
# > Because we evaluate every model on `X_test` and `y_test`, which are *never* resampled. The test set has 10,000 rows, period. Resampling only modifies the training data; it changes what the model *learns*, not what we evaluate it on. Evaluating on a resampled test set would be academic fraud: we would be grading the model in an artificial balanced world that does not exist in deployment.

# %%
model_under = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_under, y_train_under)
model_over  = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_over,  y_train_over)
model_smote = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_smote, y_train_smote)

pred_under = model_under.predict(X_test)
pred_over  = model_over.predict(X_test)
pred_smote = model_smote.predict(X_test)

# 1x4 grid of Confusion Matrices; baseline (red) vs three scratch solutions (green)
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

titles = [
    'Original Baseline\n(Imbalanced Training)',
    'Scratch Undersampled\n(Balance by Deletion)',
    'Scratch Oversampled\n(Balance by Duplication)',
    'Scratch SMOTE\n(Balance by Interpolation)'
]
preds = [y_pred_base, pred_under, pred_over, pred_smote]

for i, ax in enumerate(axes):
    cm_i = confusion_matrix(y_test, preds[i])
    sns.heatmap(
        cm_i,
        annot=True,
        fmt='d',
        cmap='Reds' if i == 0 else 'Greens',
        cbar=False,
        xticklabels=['M(Pine)', 'm(Willow)'],
        yticklabels=['M(Pine)', 'm(Willow)'],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(titles[i], fontweight='bold', pad=15, fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label' if i == 0 else '', fontsize=11)

plt.suptitle("Confusion Matrix Comparison: What Happens to False Negatives?", 
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


def model_summary_row(name, y_true, y_pred):
    """Compute precision, recall, FP, FN for the minority class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "Model": name,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "Minority Precision":   precision,
        "Minority Recall":      recall,
    }


summary_rows = [
    model_summary_row("Baseline",             y_test, y_pred_base),
    model_summary_row("Scratch Undersampling",y_test, pred_under),
    model_summary_row("Scratch Oversampling", y_test, pred_over),
    model_summary_row("Scratch SMOTE",        y_test, pred_smote),
]

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["False Negatives (FN)", "False Positives (FP)"], ascending=[True, True]
)
print("Classification report on the real test set (10,000 rows, same for all 4 models):")
print("-" * 75)
print(summary_df.to_string(index=False, formatters={
    "Minority Precision": "{:.4f}".format,
    "Minority Recall":    "{:.4f}".format,
}))

# %% [markdown]
# **Reading the Confusion Matrices and Summary Table:**
#
# The bottom-left quadrant of each heatmap shows the **False Negatives**; the number of actual Willows (minority) that the model incorrectly classified as Pine. This is the cell we are hunting.
#
# 1. **Baseline (Red):** Despite $\sim 99.9\%$ global accuracy, the uncorrected model left **8 Willows undetected**. It learned to optimize the majority class and placed its decision boundary too far into minority territory.
#
# 2. **Scratch Undersampling (Green):** The False Negatives drop aggressively. We catch nearly all the Willows now. However, because we deleted ~39,000 Pine rows from training, the model lost much of its geometric understanding of the Pine boundary. The result is **a large surge in False Positives** (Pines wrongly flagged as Willows). In a fraud context: yes, we catch all fraud, but we are also blocking hundreds of legitimate transactions per day. Customers will revolt.
#
# 3. **Scratch Oversampling (Green):** Excellent Recall; nearly identical to Undersampling in catching rare trees. But it has fewer False Positives than Undersampling, because the model still saw the full Pine geometry during training (we never deleted those rows). The problem here is silent: those exact-duplicate minority rows cause the model to draw a rigid memorized boundary that may not generalize to production data.
#
# 4. **Scratch SMOTE (Green):** The best of both worlds. Recall is as aggressive as Oversampling, but SMOTE's synthetic points encourage the model to generalize a *smoother and broader* minority boundary rather than memorizing specific points. In practice this leads to better generalization on unseen Willows.
#
# > **The golden rule:** All three resampling techniques successfully eradicate the False Negative crisis by forcing the training distribution to be 50/50. The model has no mathematical choice but to respect both classes when their training frequencies are equal.
#
# ## 5. The Pro Evaluation (AUPRC Curve)
#
# The confusion matrices above give us a single operational snapshot at the default threshold of $0.50$. But in the real world, **the classification threshold is itself a deployment decision**, not a fixed constant. A bank might deploy a fraud model at $0.30$ threshold (very aggressive, catches more fraud, blocks more legitimate transactions) or at $0.70$ threshold (conservative, fewer false alarms, lets more fraud slip). Different healthcare systems might demand different sensitivity/specificity trade-offs for the same underlying model.
#
# To evaluate a model's behavior *across all possible thresholds simultaneously*, we use **curve metrics** that sweep from $0.0$ to $1.0$ threshold and record the Precision and Recall at each step.
#
# ### 5.1 Why ROC Can Be Deceptive Here
#
# The most common curve is the **ROC Curve (Receiver Operating Characteristic)**, which plots:
# - Y-axis: True Positive Rate (Recall) = $\frac{TP}{TP + FN}$
# - X-axis: False Positive Rate = $\frac{FP}{FP + TN}$
#
# In balanced datasets, this works beautifully. But in extreme imbalance, the denominator of the False Positive Rate ($FP + TN$) is dominated by the millions of True Negatives (Pines). Even if we trigger 1,000 False Positives (a disaster in a fraud context), the FPR barely moves because we are dividing by ~nine thousand True Negatives. The ROC curve stays near-perfect and tells us almost nothing useful. A naive model can achieve $0.99$ ROC-AUC on a $99\%/1\%$ imbalanced dataset while being operationally useless.
#
# ### 5.2 The Precision-Recall Curve
#
# The **PR Curve** strips away the True Negatives entirely and focuses exclusively on the rare class:
# - X-axis: **Recall**; How many of the Willows did we catch?
# - Y-axis: **Precision**; Of everything we flagged as Willow, how many were actually Willow?
#
# The **random classifier baseline** on a PR curve is simply the prevalence rate of the positive class. On our dataset, that is $\sim 0.96\%$. Any model that does not substantially outperform this horizontal line is essentially producing random guesses about the minority class.
#
# A strong model pushes the PR curve high and to the right: high Precision at high Recall means we are catching nearly all the Willows while barely triggering any false alarms. The **Area Under the PR Curve (AUPRC)** summarizes this into a single number from $0$ to $1$.

# %%
minority_ratio = np.mean(y_test == 1)
print(f"Positive class prevalence in test set: {minority_ratio:.4f} ({minority_ratio*100:.2f}%)")
print(f"This is the random-baseline AUPRC floor. A model that barely beats this has no real power.")

# %% [markdown]
# ### 5.3 Pro Models: Library SMOTE + Weighted Random Forest
#
# For the PR Curve comparison, we introduce the professional-grade tools:
#
# - **SMOTE from `imbalanced-learn`:** The production implementation of the algorithm we just built from scratch. It includes additional logic for edge cases and is numerically optimized.
# - **Random Forest with `class_weight='balanced'`:** This is the most elegant solution. Instead of manipulating the dataset at all, we pass the raw imbalanced data to a Random Forest and tell it: *"Every time you compute a Gini split at a node, weight each minority sample as if it appears 100x more than it actually does."* The algorithm internally re-weights its loss function, achieving mathematical balance without a single row of data being duplicated or deleted. This is cost-sensitive learning.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

smote_available = True
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    smote_available = False
    print("imbalanced-learn not found; SMOTE + Logistic will be skipped.")

# Baseline model probabilities (already trained above)
probs_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Weighted Random Forest; raw imbalanced training data, but balanced internally
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',   # The key: internally re-weights loss by class frequency
    random_state=42,
    n_jobs=-1,
)
rf_weighted.fit(X_train, y_train)    # Note: we feed the ORIGINAL imbalanced X_train!
probs_rf_weighted = rf_weighted.predict_proba(X_test)[:, 1]

curves = [
    {
        "name": "Baseline Logistic (No Weights)",
        "probs": probs_baseline,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2.5,
    },
    {
        "name": "Weighted Random Forest",
        "probs": probs_rf_weighted,
        "color": "darkorange",
        "linestyle": "-",
        "linewidth": 3.5,
    },
]

if smote_available:
    smote_pro = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote_pro, y_train_smote_pro = smote_pro.fit_resample(X_train, y_train)

    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_smote_pro, y_train_smote_pro)
    probs_lr_smote = lr_smote.predict_proba(X_test)[:, 1]

    curves.insert(1, {
        "name": "SMOTE + Logistic Regression",
        "probs": probs_lr_smote,
        "color": "steelblue",
        "linestyle": "-",
        "linewidth": 2.8,
    })

# %% [markdown]
# ### 5.4 Plotting the Reality Check
#
# We draw two panels:
# - **Left (Full Range):** The complete PR curve from Recall=0 to Recall=1, showing global AUPRC.
# - **Right (Zoomed):** The critical high-recall region (Recall > 0.90), showing what happens when we push aggressively to catch almost every single Willow. This is where models diverge; some maintain high Precision at extreme Recall, others collapse.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ap_rows = []
for spec in curves:
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, spec["probs"])
    ap = average_precision_score(y_test, spec["probs"])
    ap_rows.append({"Model": spec["name"], "AUPRC": ap})

    kwargs = dict(
        where='post',
        color=spec["color"],
        linestyle=spec["linestyle"],
        linewidth=spec["linewidth"],
        label=f"{spec['name']} | AUPRC: {ap:.4f}",
    )
    ax1.step(recall_pts, precision_pts, **kwargs)
    ax2.step(recall_pts, precision_pts, **{**kwargs, "label": None})

# Random baseline (dashed red horizontal line at prevalence rate)
for ax in (ax1, ax2):
    ax.axhline(
        minority_ratio,
        color='crimson',
        linestyle='dotted',
        linewidth=2.2,
        label=f"Random Guessing | AUPRC ≈ {minority_ratio:.4f}",
        alpha=0.8
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Recall (% of actual Willows detected)', fontsize=13)
    ax.set_ylabel('Precision (% of Willow predictions that are correct)', fontsize=13)

ax1.set_title('PR Curve; Full Recall Range', fontweight='bold', pad=12, fontsize=14)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.02])

ax2.set_title('PR Curve; Zoomed: High Recall Region (>90%)', fontweight='bold', pad=12, fontsize=14)
ax2.set_xlim([0.90, 1.001])
ax2.set_ylim([0.85, 1.02])

# Unified legend at the bottom of the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.07),
    ncol=2,
    frameon=True,
    fontsize=12,
    shadow=True
)

fig.suptitle('Precision-Recall Curve; The Definitive Imbalance Metric',
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.09, 1, 1])
plt.show()

# Print AUPRC ranking
ap_df = pd.DataFrame(ap_rows).sort_values("AUPRC", ascending=False)
print("AUPRC Ranking (the higher the better):")
print("-" * 45)
print(ap_df.to_string(index=False, formatters={"AUPRC": "{:.4f}".format}))

# Operational table at threshold=0.50
print("\nOperational Snapshot at Classification Threshold = 0.50:")
print("-" * 75)
op_rows = []
for spec in curves:
    pred = (spec["probs"] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    op_rows.append({
        "Model":           spec["name"],
        "Recall@0.50":     tp / (tp + fn) if (tp + fn) else 0.0,
        "Precision@0.50":  tp / (tp + fp) if (tp + fp) else 0.0,
        "FN@0.50 (Missed)": fn,
        "FP@0.50 (Alarms)": fp,
    })

op_df = pd.DataFrame(op_rows).sort_values(["Recall@0.50", "Precision@0.50"], ascending=False)
print(op_df.to_string(index=False, formatters={
    "Recall@0.50":    "{:.4f}".format,
    "Precision@0.50": "{:.4f}".format,
}))

# %% [markdown]
# ### 6. Conclusion: What the Numbers Are Actually Telling Us
#
# The PR curves for this particular dataset sit near the top of the chart because the geometric separation between Cottonwood/Willow and Lodgepole Pine is naturally strong. This is not a paradox; it is the honest result. What matters for learning is not just the final AUPRCs, but the *operational table at threshold 0.50* and the behavior in the zoomed high-recall region.
#
# **Breaking down the operational snapshot:**
#
# - **Baseline Logistic Regression:** With its $0.997$ AUPRC, the model ranks probabilities excellently. But at the default $0.50$ threshold, the uncorrected decision boundary leaves **8 Willows undetected** (FN = 8). The model's confidence scores for rare trees never quite break the 50% mark because the loss function was never told to fight for them.
#
# - **Weighted Random Forest:** The highest AUPRC ($0.999$), achieved without touching the dataset. The internal re-weighting forces the forest's Gini splits to demand perfect minority classification, producing a probability ranker of extraordinary quality. At threshold 0.50, it misses **3 Willows**; better than baseline, but still not zero.
#
# - **SMOTE + Logistic Regression:** Technically the "lowest" AUPRC of the three ($0.997$), yet it delivers the most aggressive minority Recall at the default threshold; only **2 Willows missed**. This is SMOTE's purpose made visible: by physically inflating the minority presence in training space to 50%, the logistic boundary is pushed outwards into Pine territory until nearly every Willow probability exceeds 50%. The cost is 8 False Positives (Pine plots flagged as Willow); manageable collateral damage.
#
# **The Lesson that carries into production:**
#
# AUPRC and the full PR curve evaluate a model's *ranking power across all thresholds*. The operational table evaluates *specific deployment behavior*. They answer different questions and must be read together.
#
# A model with slightly lower AUPRC but aggressive behavior at the standard threshold can be the more useful deployment choice when False Negatives are catastrophic and you cannot afford to tune the threshold in production. On the other hand, if you have the engineering budget to serve the model at a custom threshold, the Weighted Random Forest's superior AUPRC gives you more headroom to find a threshold that gives you zero FNs while also minimizing False Positives.
#
# **When you face imbalanced real-world data (Fraud, Disease, Defects, Rare Events):**
# 1. **Do NOT** report Accuracy as a performance metric. It is a noise number in this regime.
# 2. **Do NOT** trust ROC-AUC; with millions of True Negatives in the denominator, it will look near-perfect regardless of your model's minority behavior.
# 3. **Always** inspect Precision, Recall, and the raw False Negative count at your *actual deployment threshold*.
# 4. **Use `class_weight='balanced'`** in tree-based ensembles; it is often the most powerful and elegant solution, requiring no data manipulation.
# 5. **Use SMOTE** for algorithms like Logistic Regression or SVMs that do not have native class weighting, and when you need to aggressively bias the decision boundary towards minority detection.
