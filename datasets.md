# ATLAS: Dataset & Library Registry for Autonomous Agents

This registry maps the ATLAS project structure to specific Python ecosystem datasets.
**Directive:** Use the specified loading commands to initialize the `Intuition` and `Pro` phases of the implementation loop.

---

## MODULE 1: Supervised Inference

### 1.1. Regression
#### 1.1.1. Classical Statistics (OLS, GLM)
* **Library:** `statsmodels`
* **Dataset:** Guerry (Moral Statistics of France)
* **Load Command:** `sm.datasets.get_rdataset("Guerry", "HistData").data`
* **Agent Insight:** Contains categorical regions and skewed data, ideal for demonstrating R-style formulas, dummy coding, and log-transformations.

#### 1.1.2. Regularization (Lasso, Ridge)
* **Library:** `sklearn`
* **Dataset:** Ames Housing
* **Load Command:** `fetch_openml(name="house_prices", as_frame=True)`
* **Agent Insight:** High dimensionality (80 features) forces the model to penalize irrelevant variables, demonstrating feature selection (Lasso) and multicollinearity management (Ridge).

#### 1.1.3. Robust Regression
* **Library:** `sklearn`
* **Dataset:** Diabetes
* **Load Command:** `load_diabetes(scaled=True)`
* **Agent Insight:** Pre-scaled data where deviations in blood serum measurements act as high-leverage points to test RANSAC or Huber loss.

#### 1.1.4. Non-Parametric (KNN, SVR, GP)
* **Library:** `sklearn`
* **Dataset:** California Housing
* **Load Command:** `fetch_california_housing()`
* **Agent Insight:** Contains Latitude/Longitude, allowing Gaussian Processes to model spatial covariance and local clusters rather than global linear trends.

#### 1.1.5. State-of-the-Art Tabular (XGBoost/LightGBM)
* **Library:** `seaborn`
* **Dataset:** Diamonds
* **Load Command:** `sns.load_dataset('diamonds')`
* **Agent Insight:** Large volume (53k rows) with non-linear heteroscedasticity (price vs. carat) that defeats linear models but highlights gradient boosting power.

### 1.2. Classification
#### 1.2.1. Probabilistic (Logistic, Naive Bayes)
* **Library:** `sklearn`
* **Dataset:** Wine
* **Load Command:** `load_wine()`
* **Agent Insight:** Continuous chemical features allow for clean Gaussian Naive Bayes assumption modeling conditioned on cultivar class.

#### 1.2.2. Geometric & Discriminant (SVM, LDA)
* **Library:** `sklearn`
* **Dataset:** Breast Cancer Wisconsin (Diagnostic)
* **Load Command:** `load_breast_cancer()`
* **Agent Insight:** High-dimensional features (texture, radius) allow SVM RBF kernels to project data into a space where a hyperplane achieves near-perfect separation.

#### 1.2.3. Trees & Ensembles
* **Library:** `seaborn`
* **Dataset:** Titanic
* **Load Command:** `sns.load_dataset('titanic')`
* **Agent Insight:** Perfect for "Feature Interaction" (e.g., women + children first) and handling missing values in a production-like messy table.

#### 1.2.4. Handling Real-World Mess (Imbalanced)
* **Library:** `sklearn`
* **Dataset:** Forest Covertypes
* **Load Command:** `fetch_covtype()`
* **Agent Insight:** Extreme class skew (rare forest types) provides a rigorous environment to test SMOTE and F1/AUPRC metrics over accuracy.

### 1.3. Computer Vision
#### 1.3.1. Hand-Crafted Features (HOG/SIFT)
* **Library:** `sklearn`
* **Dataset:** Digits (8x8)
* **Load Command:** `load_digits()`
* **Agent Insight:** Low resolution allows treating pixels as raw features or applying manual gradient filters before the deep learning era.

#### 1.3.2. CNN Architectures
* **Library:** `keras` / `tensorflow`
* **Dataset:** Fashion-MNIST (Grayscale) or CIFAR-10 (Color)
* **Load Command:** `keras.datasets.fashion_mnist.load_data()`
* **Agent Insight:** Fashion-MNIST tests structural recognition; CIFAR-10 forces the network to learn color-based features alongside spatial textures.

#### 1.3.3. Object Detection (YOLO/SSD)
* **Library:** `torchvision`
* **Dataset:** VOC / MS COCO
* **Load Command:** `torchvision.datasets.VOCDetection()`
* **Agent Insight:** Provides bounding box annotations ($x_{min}, y_{min}$) required to train regression heads for localization, not just classification.

#### 1.3.4. 3D & Neural Rendering (NeRF)
* **Library:** `torch_geometric`
* **Dataset:** S3DIS (Stanford Large-Scale 3D Indoor Spaces)
* **Load Command:** `torch_geometric.datasets.S3DIS(root='/tmp/S3DIS')`
* **Agent Insight:** Point clouds allow moving from 2D grids to 3D voxel representations to infer depth and geometry.

---

## MODULE 2: Unsupervised Structure Discovery

### 2.1. Clustering
#### 2.1.1. Centroid-Based (K-Means)
* **Library:** `seaborn`
* **Dataset:** Penguins
* **Load Command:** `sns.load_dataset('penguins')`
* **Agent Insight:** Multi-modal distribution allows testing if K-Means centroids align with biological species labels despite overlapping physical features.

#### 2.1.2. Density-Based (DBSCAN)
* **Library:** `seaborn`
* **Dataset:** Planets (Exoplanets)
* **Load Command:** `sns.load_dataset('planets')`
* **Agent Insight:** Irregular clusters in mass-distance space representing "discovery regimes" are ideal for finding dense regions vs. noise.

#### 2.1.3. Hierarchical & Probabilistic
* **Library:** `sklearn`
* **Dataset:** Iris (Probabilistic) or Wine (Hierarchical)
* **Load Command:** `load_iris()`
* **Agent Insight:** Overlapping distributions in Iris (Versicolor/Virginica) provide a sophisticated test for Gaussian Mixture Models' soft clustering probabilities.

### 2.2. Dimensionality Reduction
#### 2.2.1. Linear (PCA)
* **Library:** `sklearn`
* **Dataset:** Digits
* **Load Command:** `load_digits()`
* **Agent Insight:** Eigenvector decomposition of the 64x64 covariance matrix reveals clear number separations in just 2 or 3 dimensions.

#### 2.2.2. Non-Linear (t-SNE/UMAP)
* **Library:** `keras`
* **Dataset:** MNIST
* **Load Command:** `keras.datasets.mnist.load_data()`
* **Agent Insight:** Projects 784-dim space into a 2D map, proving the "manifold of human handwriting" is low-dimensional.

### 2.3. Anomaly Detection
* **Library:** `sklearn`
* **Dataset:** Kddcup99 (Intrusion Detection)
* **Load Command:** `fetch_kddcup99(subset=None, percent10=True)`
* **Agent Insight:** Massive dataset where "Normal" traffic vastly outnumbers "Attacks," perfect for Isolation Forests.

### 2.4. Association Rule Mining
* **Library:** `pycaret` / `mlxtend`
* **Dataset:** France (Retail)
* **Load Command:** `get_data('france')`
* **Agent Insight:** Transactional data convertible to binary matrices for discovering rules like "{Bread} -> {Butter}".

---

## MODULE 3: Generative AI

### 3.1. Adversarial Learning (GANs)
* **Library:** `torchvision` / `PyTorch`
* **Dataset:** CelebA
* **Load Command:** `torchvision.datasets.CelebA()`
* **Agent Insight:** High volume of faces with attributes (glasses, smiling) for training Discriminators and controlling Generator outputs.

### 3.2. Probabilistic & VAEs
* **Library:** `keras`
* **Dataset:** MNIST
* **Load Command:** `keras.datasets.mnist.load_data()`
* **Agent Insight:** Allows visualization of a continuous latent space where morphing from '0' to '1' yields realistic intermediate digits.

### 3.3. Diffusion Models
* **Library:** `keras`
* **Dataset:** CIFAR-100
* **Load Command:** `keras.datasets.cifar100.load_data()`
* **Agent Insight:** Fine-grained super-classes (e.g., "Bee" vs "Wasp") test the diffusion model's ability to denoise subtle textures.

---

## MODULE 4: Sequential Intelligence

### 4.1. NLP
#### 4.1.1. Topic Modeling (LDA)
* **Library:** `sklearn`
* **Dataset:** 20 Newsgroups
* **Load Command:** `fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))`
* **Agent Insight:** Use `remove` parameter to prevent metadata leakage; classic benchmark for TF-IDF and Bag of Words.

#### 4.1.2. Vectors & Embeddings
* **Library:** `nltk`
* **Dataset:** Brown Corpus
* **Load Command:** `nltk.download('brown'); nltk.corpus.brown`
* **Agent Insight:** Categorized by genre, allowing Word2Vec to show how word context shifts (e.g., "bank") based on the category.

#### 4.1.3. Recurrent Architectures (LSTM)
* **Library:** `keras`
* **Dataset:** IMDB Reviews
* **Load Command:** `keras.datasets.imdb.load_data()`
* **Agent Insight:** Pre-tokenized sequences focus the agent on temporal dynamics and sentiment reversal (e.g., "but").

#### 4.1.4. Transformers (BERT)
* **Library:** `nltk`
* **Dataset:** Reuters-21578
* **Load Command:** `nltk.download('reuters'); nltk.corpus.reuters`
* **Agent Insight:** Multi-label classification (one doc = "Grain" AND "Wheat") tests self-attention mechanisms.

### 4.2. Time Series
#### 4.2.1. Statistical (SARIMA)
* **Library:** `sktime`
* **Dataset:** Airline
* **Load Command:** `load_airline()`
* **Agent Insight:** Clear seasonality and trend make it the definitive test for Exponential Smoothing and ARIMA.

#### 4.2.2. ML & Deep Learning
* **Library:** `sktime`
* **Dataset:** US Change (ML) or Italy Power Demand (DL)
* **Load Command:** `sktime.datasets.load_uschange()`
* **Agent Insight:** US Change allows "lagging" features for regression; Italy Power Demand tests classification of daily profiles.

### 4.3. Audio
* **Library:** `torchaudio`
* **Dataset:** SpeechCommands
* **Load Command:** `torchaudio.datasets.SPEECHCOMMANDS()`
* **Agent Insight:** Requires converting 1D waveforms to Mel-Spectrograms (2D) to apply Vision Transformers or CNNs.

---

## MODULE 5: Agents & Strategy

### 5.1. Optimization
* **Library:** `scipy`
* **Dataset:** Rosenbrock / Rastrigin Functions
* **Load Command:** `scipy.optimize.rosen`
* **Agent Insight:** Mathematical landscapes to test convergence speed of evolutionary or simplex algorithms.

### 5.2. Bandit Algorithms & RL
* **Library:** `gymnasium` (implied) / `sklearn`
* **Dataset:** FrozenLake (RL) or Adult Census (Contextual Bandits)
* **Load Command:** `gym.make('FrozenLake-v1')`
* **Agent Insight:** FrozenLake tests Q-Learning on sparse rewards; Adult Census simulates policy-making for demographic cohorts.

---

## MODULE 6: Relational Intelligence

### 6.1. Recommender Systems
* **Library:** `surprise`
* **Dataset:** MovieLens 100k
* **Load Command:** `Dataset.load_builtin('ml-100k')`
* **Agent Insight:** Standard matrix for SVD decomposition to find "latent genres" via user-item embeddings.

### 6.2. Graph Neural Networks
* **Library:** `planetoid` / `torch_geometric`
* **Dataset:** Cora (Node Classification)
* **Load Command:** `Planetoid(root='/tmp/Cora', name='Cora')`
* **Agent Insight:** Uses citation links to predict paper topics, testing message passing (GCN/GAT) capabilities.

---

## MODULE 7: Scientific Intelligence

### 7.1. Causal Inference
* **Library:** `dowhy`
* **Dataset:** IHDP (Infant Health)
* **Load Command:** `dowhy.datasets.linear_dataset(...)` (or equivalent simulation)
* **Agent Insight:** Includes "counterfactuals" (what would have happened without treatment), allowing verification of causal models.

### 7.2. Uncertainty & Explainability
* **Library:** `sklearn`
* **Dataset:** California Housing (Uncertainty) or Breast Cancer (XAI)
* **Load Command:** `fetch_california_housing()`
* **Agent Insight:** Use California Housing to generate Conformal Prediction intervals; use Breast Cancer for SHAP value feature decomposition.

---

## MODULE 8: The Engineering Toolkit

### 8.1. Tuning & AutoML
* **Library:** `sklearn`
* **Dataset:** Diabetes (Fast) or Adult (Complex)
* **Load Command:** `load_diabetes()`
* **Agent Insight:** Diabetes is small for quick Bayesian Optimization checks; Adult tests AutoML pipeline discovery capabilities.

### 8.3. Vector Search
* **Library:** `keras` / `tensorflow`
* **Dataset:** Fashion-MNIST
* **Load Command:** `keras.datasets.fashion_mnist.load_data()`
* **Agent Insight:** Generate embeddings for "boot" images to test Approximate Nearest Neighbor (ANN) retrieval speed.