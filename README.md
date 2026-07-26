# ATLAS: Algorithmic Techniques for Learning And Statistics

### Practical Case Studies from Classical Statistics to State-of-the-Art Generative AI

Welcome to **ATLAS** (**A**lgorithmic **T**echniques for **L**earning **A**nd **S**tatistics). This repository is designed not as a theoretical roadmap, but as a collection of **practical implementations** addressing specific real-world cases. While many resources focus on the "what", this project focuses on the "how": building the engine before we drive the car.

> **Read it as a website.** Every technique is being turned into a visual, interactive article at **[aleetreny.github.io/ATLAS](https://aleetreny.github.io/ATLAS/)**: scrollytelling, live models running in your browser, and every number measured rather than quoted. Seventeen articles published so far: regression and classification complete (modules 1.1 and 1.2, every chip lit), and a four-article arc through clustering (2.1): k-means, mixtures, density and hierarchies.

The structure of this repository follows the natural progression of data intelligence, guiding you through the main paradigms and techniques used to analyze, model, generate, and deploy data-driven solutions. Each section builds on the previous, covering the essential approaches and tools needed to address real-world data challenges from start to finish.

---

## Directory Logic

- **`modules/`**: The playground. Each sub-folder (e.g., `modules/01_supervised_inference/`) contains notebooks that combine storytelling with code.
- **`src/`**: The core library. Organized by function rather than by module to ensure reusability:
  - `src/models/`: From-scratch implementations of algorithms (e.g., `linear.py`, `trees.py`). Only NumPy and math allowed.
  - `src/utils/`: Reusable helpers for data loading, metrics, and a dedicated `visuals.py` for premium, consistent plotting.
- **`datasets/`**: To keep the repo light, this folder houses small local datasets or fetcher scripts for larger external data.

## The Implementation Loop

Every technique follows a strict, four-step execution:

1. **Intuition**: What is the real-world problem? (e.g., "How do we predict house prices when the data is full of outliers?").
2. **Scratch**: Building the solution from the ground up in `src/`. No shortcuts, just matrices and logic.
3. **Pro**: Implementing the same solution using industry-standard libraries to see how it's done in production.
4. **Visual**: Making it click with high-quality, interactive visualizations because if you can't see the result, it didn't happen.

---

## MODULE 1: Supervised Inference (Prediction & Discrimination)

You have the questions (input data) and the correct answers (labels). Your goal is to train a model to map the inputs to the outputs so accurately that it can predict the answers for new questions it has never seen before. This covers everything from predicting a house price (a number) to diagnosing a disease (a category) or recognizing a face in a photo.

### 1.1. Regression (Predicting Continuous Values)

- **Classical Statistics:**
  - **Linear Regression (OLS):** The foundation. Fitting a straight line to data.
  - **Polynomial Regression:** Modeling curved relationships.
  - **GLMs (Generalized Linear Models):** Handling non-normal error distributions (Poisson, Gamma).
  - **PLS (Partial Least Squares):** Dealing with high multicollinearity by projecting features into latent structures before regression.
- **Regularization (Controlling Complexity):**
  - **Lasso (L1):** Shrinking coefficients to zero (feature selection).
  - **Ridge (L2):** Shrinking coefficients to prevent overfitting.
  - **ElasticNet:** The hybrid approach.
- **Robust Regression (Ignoring Outliers):**
  - **RANSAC (Random Sample Consensus):** Iteratively fitting models to inliers and ignoring outliers.
  - **Theil-Sen Estimator:** Calculating the median of slopes (extremely robust to outliers).
  - **Huber Regression:** Using a loss function that is quadratic for small errors but linear for large errors (less sensitive to outliers).
- **Non-Parametric & ML:**
  - **KNN Regression:** Predicting based on the average of nearest neighbors.
  - **SVR (Support Vector Regression):** Finding a tube of tolerance around the prediction.
  - **Kernel Ridge Regression (KRR):** Combining Ridge Regression with the Kernel Trick to learn non-linear functions.
  - **Gaussian Processes (GP):** A Bayesian non-parametric approach that predicts a distribution (uncertainty) rather than just a value.
- **State-of-the-Art (Tabular):**
  - **XGBoost / LightGBM:** Optimized Gradient Boosting (The industry standard).
  - **CatBoost:** specialized for categorical data.
- **Deep Learning for Tables:**
  - **TabNet:** Attentive Interpretable Tabular Learning.
  - **Neural Additive Models (NAMs):** Deep learning with interpretability graphs.

### 1.2. Classification (Predicting Labels/Categories)

- **Probabilistic:**
  - **Logistic Regression:** Predicting probabilities (Yes/No).
  - **Naive Bayes:** Fast, probabilistic classification based on Bayes' Theorem.
- **Geometric & Discriminant Analysis:**
  - **SVM (Support Vector Machines):** Finding the optimal hyperplane to separate classes (Kernel Trick).
  - **LDA (Linear Discriminant Analysis):** Projecting data to maximize class separation assuming equal covariance.
  - **QDA (Quadratic Discriminant Analysis):** Similar to LDA but allows for non-linear boundaries (different covariances per class).
  - **KDA (Kernel Discriminant Analysis):** Extending LDA/QDA with kernels to separate complex, non-linear clusters.
- **Trees & Ensembles:**
  - **Decision Trees (CART/C4.5):** Creating explicit "If-Then" rules.
  - **Random Forest:** Bagging (Bootstrap Aggregating) logic to reduce variance.
  - **ExtraTrees:** Randomized trees for faster convergence.
- **Handling Real-World Mess:**
  - **Imbalanced Learning:** SMOTE, ADASYN, Class Weights.
  - **Calibration:** Isotonic Regression, Platt Scaling (Ensuring 70% confidence actually means 70% accuracy).

### 1.3. Computer Vision (Visual Perception)

- **Hand-Crafted Features (The Pre-DL Era):**
  - **HOG (Histogram of Oriented Gradients):** Detecting edges and shapes.
  - **SIFT / SURF / ORB:** Keypoint detection (stitching images and recognizing objects).
  - **Viola-Jones Algorithm:** The classic real-time face detector (Haar Cascades).
- **CNN Architectures (The Deep Learning Revolution):**
  - **LeNet / AlexNet:** The pioneers.
  - **VGG:** The concept of "deep" blocks.
  - **ResNet:** Skip connections (Residuals) allowing extremely deep networks.
  - **EfficientNet:** Optimizing width, depth, and resolution simultaneously.
  - **ConvNeXt:** Modernizing CNNs to compete with Transformers.
- **Object Detection (Where is it?):**
  - **YOLO (You Only Look Once) v8/v9/v10:** Real-time speed.
  - **SSD (Single Shot Detector):** Balancing speed and accuracy.
  - **Faster R-CNN:** Two-stage detection (Region Proposals).
- **Segmentation (Pixel-perfect precision):**
  - **U-Net:** The gold standard for biomedical imaging.
  - **Mask R-CNN:** Instance segmentation.
  - **SAM (Segment Anything Model):** Zero-shot segmentation foundation model.
- **3D & Neural Rendering:**
  - **NeRF (Neural Radiance Fields):** Representing 3D scenes in network weights.
  - **3D Gaussian Splatting:** Real-time 3D rendering using learned Gaussian clouds.

---

## MODULE 2: Unsupervised Structure Discovery

You have piles of evidence (data) but no answers. Nobody tells you what is right or wrong. Your goal is to discover hidden structures, group similar items together, or simplify complex data into something human-readable. It is finding the signal in the noise.

### 2.1. Clustering (Grouping)

- **Centroid-Based:**
  - **K-Means:** Partitioning data into K distinct spheres.
  - **K-Medoids / K-Modes:** Robust versions for outliers and categorical data.
- **Density-Based (Finding arbitrary shapes):**
  - **DBSCAN:** Clustering based on density, handling noise as outliers.
  - **HDBSCAN:** Hierarchical density clustering (The modern standard).
  - **OPTICS:** Addressing varying densities.
  - **Mean Shift:** A sliding-window, centroid-based algorithm that shifts to the mode (densest area) of points. No need to pre-define K.
- **Probabilistic & Exemplar:**
  - **GMM (Gaussian Mixture Models):** Soft clustering where points belong to clusters with a probability.
  - **Affinity Propagation:** Choosing data points as "exemplars" and passing messages between them.
- **Hierarchical:**
  - **Agglomerative Clustering:** Building a tree of clusters (Dendrograms).
  - **BIRCH:** Efficient hierarchical clustering for very large datasets.
- **Graph-Based:**
  - **Spectral Clustering:** Using eigenvalues of the similarity matrix to reduce dimensions before clustering (great for non-convex shapes).

### 2.2. Dimensionality Reduction & Manifold Learning

- **Linear Projection:**
  - **PCA (Principal Component Analysis):** Maximizing variance in orthogonal directions.
  - **ICA (Independent Component Analysis):** Separating mixed signals (The "Cocktail Party Problem").
  - **NMF (Non-negative Matrix Factorization):** Parts-based representation.
  - **Factor Analysis:** Modeling observed variables as linear combinations of potential latent variables.
  - **MDS (Multidimensional Scaling):** Preserving the distances between points in a lower-dimensional space.
- **Non-Linear Manifold Learning (Visualization):**
  - **t-SNE:** Preserving local structure (great for clusters).
  - **UMAP:** Preserving global and local structure (faster and mathematically robust).
  - **Isomap / LLE:** Geodesic distances on a manifold.
  - **SOM (Self-Organizing Maps):** A type of Neural Network that produces a low-dimensional (typically 2D) discretized representation.
- **Neural Compression:**
  - **Autoencoders (AE):** Bottleneck architecture.
  - **Denoising AE:** Learning robust features by fixing corrupted inputs.

### 2.3. Anomaly Detection (Outlier Hunting)

- **Statistical:**
  - **Z-Score / IQR:** Simple deviation checks.
  - **Grubbs' Test:** Detecting a single outlier in univariate data.
  - **Mahalanobis Distance:** Detecting outliers considering the correlation between variables.
- **Algorithmic:**
  - **Isolation Forest:** Randomly slicing data; outliers are isolated faster.
  - **One-Class SVM:** Learning a boundary of "normality".
  - **LOF (Local Outlier Factor):** Density-based anomaly detection.
  - **Minimum Covariance Determinant (MCD):** Robust estimator of covariance for outlier detection.

### 2.4. Association Rule Mining

- **Market Basket Analysis:**
  - **Apriori / Eclat / FP-Growth:** "People who bought X also bought Y."

---

## MODULE 3: Generative AI (Creative Synthesis)

Instead of analyzing data, the model learns the underlying distribution of the reality to create _new_ data samples that never existed before. It is the shift from "Classifying a cat" to "Drawing a cat".

### 3.1. Adversarial Learning (The Game)

- **Core Concepts:**
  - **GANs (Generative Adversarial Networks):** Generator vs. Discriminator game.
- **Architectures:**
  - **DCGAN:** Deep Convolutional GANs.
  - **WGAN (Wasserstein GAN):** Improved training stability using Earth Mover's Distance.
  - **StyleGAN (v2/v3):** Controlling coarse (pose) and fine (texture) features independently.
- **Translation:**
  - **Pix2Pix:** Paired image translation (e.g., Sketch -\> Photo).
  - **CycleGAN:** Unpaired translation (e.g., Horse -\> Zebra).

### 3.2. Probabilistic & Flow Models

- **Latent Variable Models:**
  - **VAEs (Variational Autoencoders):** Mapping data to a probabilistic latent space.
  - **VQ-VAE:** Vector Quantized VAEs (discrete latent codes, high quality).
  - **Energy-Based Models (EBMs):** Learning an energy function that assigns low energy to real data and high energy to fake data.
- **Flow-Based:**
  - **Normalizing Flows:** Exact likelihood estimation using invertible transformations.
  - **RealNVP / Glow:** Flows capable of generating high-quality images.

### 3.3. Diffusion Models (The Modern Era)

- **The Physics:**
  - **DDPM (Denoising Diffusion Probabilistic Models):** Forward diffusion (adding noise) and Reverse diffusion (removing noise).
- **Optimization:**
  - **Latent Diffusion Models (Stable Diffusion):** Running diffusion in a compressed latent space.
  - **ControlNet:** Adding spatial conditioning (edges, depth, pose) to generation.
  - **Consistency Models:** Generating high-quality samples in 1 or 2 steps (distilling diffusion).
  - **Rectified Flow:** Straightening the diffusion paths for faster generation (e.g., Flux).

---

## MODULE 4: Sequential Intelligence (NLP & Time Series)

Data here isn't a snapshot; it flows. Context matters: what happened _before_ dictates what happens _next_. This covers understanding human language and predicting the future based on history.

### 4.1. Natural Language Processing (NLP)

- **Classic Representation & Topic Modeling:**
  - **Bag of Words / TF-IDF:** Frequency-based statistics.
  - **N-Grams:** Statistical probability of word sequences.
  - **LDA (Latent Dirichlet Allocation):** Generative probabilistic model to find "topics" in a collection of documents.
  - **NMF for Topics:** Using Matrix Factorization to extract topics.
- **Vector Space (Embeddings):**
  - **Word2Vec (Skip-gram/CBOW):** Words with similar meanings are close in space.
  - **GloVe:** Global Vectors based on co-occurrence matrix.
  - **FastText:** Embeddings that handle sub-word information (great for typos or rare words).
- **Recurrent Architectures (Pre-Transformer):**
  - **HMM (Hidden Markov Models):** The classical statistical approach to sequences (Pos Tagging, Speech).
  - **RNNs:** Basic recurrence (suffers from vanishing gradient).
  - **LSTM / GRU:** Gating mechanisms to retain long-term memory.
- **The Transformer Era (Attention is All You Need):**
  - **Self-Attention Mechanism:** Weighing the importance of words relative to each other.
  - **Encoder Models (Understanding):** **BERT**, **RoBERTa** (Masked Language Modeling).
  - **Decoder Models (Generation):** **GPT Family**, **LLaMA** (Causal Language Modeling).
  - **Encoder-Decoder:** **T5**, **BART** (Translation, Summarization).
  - **BERTopic:** Combining Transformers (BERT) and Class-based TF-IDF for advanced topic modeling.
- **Post-Transformer:**
  - **Mamba / SSMs (State Space Models):** Linear-time sequence modeling (The challenger).

### 4.2. Time Series Forecasting

- **Statistical:**
  - **ARIMA / SARIMA:** Autoregressive Integrated Moving Average (with Seasonality).
  - **ETS:** Error, Trend, Seasonality (Exponential Smoothing).
  - **VAR:** Vector Autoregression (Multivariate).
  - **GARCH:** Modeling volatility (crucial for finance/risk).
- **Machine Learning:**
  - **Prophet:** Additive regression model for business cycles.
  - **XGBoost for TS:** Transforming time series into a supervised learning problem (windowing).
- **Deep Learning:**
  - **N-BEATS:** Neural Basis Expansion Analysis (Interpretable DL).
  - **TFT (Temporal Fusion Transformers):** Combining attention with static covariates.
  - **TimeGPT / Chronos:** Foundation models for time series.

### 4.3. Audio & Speech

- **Signal Processing:** Spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs).
- **Models:** **WaveNet** (Generative audio), **Whisper** (Transformer for ASR).

---

## MODULE 5: Agents & Strategy (RL & Optimization)

An agent exists in an environment and must take a sequence of actions to maximize a reward. It is not about "what is this?" (Classification), but "what should I do next?".

### 5.1. Mathematical Optimization (Static)

- **Exact Methods:**
  - **Linear Programming (Simplex):** Optimizing a linear objective with constraints.
  - **MILP (Mixed Integer Linear Programming):** Discrete decision making (Branch & Bound).
  - **Quadratic Programming:** Optimization where the objective function is quadratic.
- **Heuristics (Nature-Inspired):**
  - **Genetic Algorithms:** Evolution, crossover, mutation.
  - **Simulated Annealing:** Physics-inspired probabilistic search.
  - **Particle Swarm Optimization:** Swarm intelligence.

### 5.2. Bandit Algorithms (The Bridge)

- **Multi-Armed Bandits:** Exploring vs. Exploiting without states (A/B testing on steroids).
- **Thompson Sampling:** Probabilistic algorithm for decision making.
- **UCB (Upper Confidence Bound):** Optimism in the face of uncertainty.

### 5.3. Reinforcement Learning (Dynamic)

- **Tabular Methods (Small State Spaces):**
  - **Q-Learning / SARSA:** Learning a lookup table of values for every state-action pair.
- **Value-Based Deep RL:**
  - **DQN (Deep Q-Network):** Using Neural Nets to approximate the Q-table (Atari games).
  - **Double DQN / Dueling DQN:** Stability improvements.
- **Policy-Based Deep RL:**
  - **REINFORCE:** Optimizing the policy directly.
  - **A2C / A3C (Actor-Critic):** One net acts, the other critiques.
  - **PPO (Proximal Policy Optimization):** Clipping updates to ensure stable learning (Used for RLHF).
  - **SAC (Soft Actor-Critic):** Maximizing reward + entropy (exploration).
- **Model-Based RL:**
  - **MCTS (Monte Carlo Tree Search):** Planning ahead (AlphaGo).
  - **Dreamer:** Learning a world model and training inside the "dream".

---

## MODULE 6: Relational Intelligence (Graphs & Recommenders)

Entities (nodes) are defined not just by their features, but by their connections (edges) to others. Or, predicting user preferences based on the collective wisdom of the crowd.

### 6.1. Recommender Systems

- **Collaborative Filtering:**
  - **User-Based / Item-Based:** "Users like you liked X."
  - **Matrix Factorization (SVD / ALS):** Decomposing the user-item interaction matrix.
- **Deep Recommendation:**
  - **Neural Collaborative Filtering:** Replacing dot products with MLPs.
  - **Two-Tower Architecture:** Separating User and Item embeddings for fast retrieval (YouTube/TikTok style).
  - **DeepFM:** Combining factorization machines with deep learning.

### 6.2. Graph Neural Networks (GNNs)

- **Message Passing:**
  - **GCN (Graph Convolutional Networks):** Aggregating information from neighbors.
  - **GAT (Graph Attention Networks):** Weighing the importance of specific neighbors.
  - **GraphSAGE:** Inductive representation learning on large graphs.
- **Knowledge Graphs (Embeddings):**
  - **TransE / RotatE:** Modeling relationships as translations/rotations in vector space.
- **Tasks:** Link Prediction (Socials), Node Classification (Fraud), Graph Classification (Drug discovery).

---

## MODULE 7: Scientific Intelligence (Causal & Explainable)

Deep Learning models are often "Black Boxes" that find correlations. This module asks "Why?" and "What if?". It moves from "People with umbrellas get wet" to "Rain causes wetness, not umbrellas".

### 7.1. Causal Inference

- **Theory:**
  - **Correlation vs. Causation:** Spurious correlations.
  - **Confounders:** Variables that influence both cause and effect (The third variable problem).
  - **Simpson's Paradox:** Trends that reverse when groups are combined.
- **Techniques:**
  - **Propensity Score Matching:** Simulating randomized control trials from observational data.
  - **Instrumental Variables:** Using proxies to isolate causal effects.
  - **Do-Calculus (Pearl):** Mathematical framework for causal intervention.
  - **Causal Forests:** Machine Learning for Heterogeneous Treatment Effects.

### 7.2. Uncertainty Quantification (Knowing what you don't know)

- **Conformal Prediction:** Generating rigorous prediction intervals (e.g., "The price is between \$10 and \$20 with 95% guarantee").
- **Bayesian Neural Networks:** Weights are distributions, not fixed numbers.

### 7.3. Explainable AI (XAI)

- **Model-Agnostic:**
  - **SHAP (Shapley Values):** Game theory approach to feature contribution.
  - **LIME:** Local approximation using simple linear models.
  - **Permutation Importance:** Shuffling data to measure feature impact.
- **Visual Explanations:**
  - **Saliency Maps / Grad-CAM:** Highlighting pixels that triggered the CNN.
  - **Attention Visualization:** Seeing what words a Transformer focused on.

---

## MODULE 8: The Engineering Toolkit

Knowing the theory isn't enough; you need the tools to tune, validate, and deploy these engines efficiently. It bridges the gap between a notebook and a production system.

### 8.1. Model Optimization & Tuning

- **Search Strategies:**
  - **Grid Search:** Brute force.
  - **Random Search:** Statistical efficiency.
  - **Bayesian Optimization (Optuna/Hyperopt):** Intelligent search using probability.
- **AutoML:**
  - **TPOT / Auto-sklearn:** Genetic programming to find the best pipeline automatically.
- **Advanced Training:**
  - **Transfer Learning:** Reusing pre-trained weights.
  - **Knowledge Distillation:** Compressing a big model (Teacher) into a small one (Student).
  - **PEFT (Parameter-Efficient Fine-Tuning):** LoRA (Low-Rank Adaptation), QLoRA, Adapters.

### 8.2. Advanced Evaluation

- **Validation Strategies:** Stratified K-Fold, TimeSeriesSplit (No peeking into the future).
- **Complex Metrics:** ROC-AUC, LogLoss, MCC (Matthews Correlation Coefficient), BLEU/ROUGE (for Text).

### 8.3. Vector Search & Retrieval

- **Embeddings Management:**
  - **Cosine Similarity vs. Euclidean Distance.**
  - **ANN (Approximate Nearest Neighbors):** HNSW, FAISS (Facebook AI Similarity Search).
  - **Vector Databases:** Pinecone, Milvus, ChromaDB.
