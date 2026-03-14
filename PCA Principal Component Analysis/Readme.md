

# Principal Component Analysis (PCA)

This tutorial covers Principal Component Analysis (PCA), a dimensionality reduction technique in machine learning, with step-by-step examples using Python. It includes a synthetic dataset example and a real-world example using the "creditcard.csv" dataset.

## Introduction to Dimensionality Reduction

Dimensionality reduction reduces the number of features in a dataset while preserving most of the information. This is useful for:
- Reducing computational complexity
- Mitigating overfitting
- Visualizing high-dimensional data
- Removing noise and redundant features

PCA is one of the most popular dimensionality reduction techniques, transforming the data into a new set of variables called principal components.

## What is PCA?

PCA is a statistical method that transforms a dataset into a new coordinate system where the axes (principal components) are orthogonal and represent directions of maximum variance. The first principal component captures the most variance, the second captures the second most, and so on.

Key concepts:
- **Principal Components**: New features that are linear combinations of the original features.
- **Variance**: Measures how much information each component captures.
- **Orthogonality**: Components are uncorrelated with each other.

## Computing Components in PCA

PCA involves the following steps:
1. **Standardize the data**: Scale features to have zero mean and unit variance.
2. **Compute the covariance matrix**: Understand relationships between features.
3. **Calculate eigenvalues and eigenvectors**: Eigenvectors represent the directions of principal components, and eigenvalues indicate their magnitude (variance).
4. **Sort eigenvectors**: Order by eigenvalues to select the top components.
5. **Transform the data**: Project the data onto the selected principal components.

## Dimensionality Reduction using PCA

PCA reduces dimensions by selecting the top *k* principal components that explain most of the variance (e.g., 95%). This reduces the dataset from *n* features to *k* features.

---

## Step-by-Step PCA Example with Synthetic Data

### Step 1: Generate Synthetic Data
We'll create a synthetic dataset with 3 features and 100 samples.

### Step 2: Standardize the Data
Standardization ensures all features contribute equally.

### Step 3: Apply PCA
We'll reduce the data to 2 dimensions.

### Step 4: Visualize the Results
Plot the transformed data to see the reduced dimensions.

Here's the Python code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 samples, 3 features
X[:, 1] = X[:, 1] * 2 + X[:, 0]  # Introduce correlation
X[:, 2] = X[:, 2] * 0.5 + X[:, 1]  # Introduce more correlation

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5)
plt.title('PCA Result (Synthetic Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('pca_synthetic.png')

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

**Output**:
- A scatter plot saved as `pca_synthetic.png` showing the data in 2D.
- Explained variance ratio (e.g., `[0.71, 0.23]`), indicating the proportion of variance captured by each component.

---

## PCA with Real Dataset (Credit Card Fraud Detection)

We'll use the "creditcard.csv" dataset, which contains transactions labeled as fraudulent or non-fraudulent. The dataset has 30 features (V1-V28, Time, Amount) and a binary class label.

### Dataset Description
- **Source**: Kaggle (Credit Card Fraud Detection dataset)
- **Features**: 30 numerical features (V1-V28 are PCA-transformed features, Time, Amount)
- **Target**: Class (0 = non-fraud, 1 = fraud)
- **Size**: ~284,807 transactions

### Step 1: Load and Explore the Dataset
Download the dataset and inspect its structure.

### Step 2: Preprocess the Data
Handle missing values, standardize features, and separate the target variable.

### Step 3: Apply PCA
Reduce the dimensionality to 2 components for visualization.

### Step 4: Visualize and Interpret
Plot the results and analyze the explained variance.

Here's the Python code:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Assuming 'creditcard.csv' is in the working directory
df = pd.read_csv('creditcard.csv')

# Step 2: Preprocess the data
# Drop rows with missing values (if any)
df = df.dropna()

# Separate features and target
X = df.drop('Class', axis=1)  # Features (V1-V28, Time, Amount)
y = df['Class']  # Target (0 or 1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='green', label='Non-Fraud', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Fraud', alpha=0.5)
plt.title('PCA on Credit Card Fraud Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('pca_creditcard.png')

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

**Output**:
- A scatter plot saved as `pca_creditcard.png` showing fraud and non-fraud transactions in 2D.
- Explained variance ratio, showing how much variance the two components capture.

---

## Credit Card Dataset

Below is a sample of how the "creditcard.csv" dataset should look. Since the actual dataset is large (~284,807 rows), you can download it from Kaggle or use this sample for testing. For this tutorial, I'll provide a small synthetic version of the dataset structure.

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
0,-1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110474,0.066928,0.128539,-0.189115,0.133558,-0.021053,149.62,0
1,1.191857,0.266151,0.166480,0.448154,0.060018,-0.082361,-0.078803,0.085102,-0.255425,-0.166974,1.612727,1.065235,0.489095,-0.143772,0.635558,0.463917,-0.114805,-0.183361,-0.145783,-0.069083,-0.225775,0.638672,0.101288,-0.339846,0.167170,0.125895,-0.008983,0.014724,2.69,0
2,-1.358354,-1.340163,1.773209,0.379780,-0.503198,1.800499,0.791461,0.247676,-1.514654,0.207643,0.624501,0.066084,0.717293,-0.165946,2.345865,-2.890083,1.109969,-0.121359,-2.261857,0.524980,0.247998,0.771679,0.909412,-0.689281,-0.327642,-0.139097,-0.055353,-0.059752,378.66,0
3,-0.966272,-0.185226,1.792993,-0.863291,-0.010309,1.247203,0.237609,0.377436,-1.387024,-0.054952,-0.226487,0.178228,0.507757,-0.287924,-0.631418,1.218632,-1.332284,-1.443983,0.110087,0.851465,0.392048,0.410430,-0.705117,-0.112100,0.313894,0.059784,-0.073425,-0.004222,123.50,0
4,-1.158233,0.877737,1.548718,0.403034,-0.407193,0.095921,0.592941,-0.270533,0.817739,0.753074,-0.822843,0.538196,1.345852,-1.119670,0.175121,-0.451449,-0.237033,-0.038195,0.803487,0.408542,-0.009431,0.798278,-0.137458,0.141267,-0.206010,0.502292,0.219422,0.215153,69.99,0
```

**Instructions to Get the Full Dataset**:
1. Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
2. Download `creditcard.csv` and place it in your working directory.
3. Run the code above with the full dataset.

**Note**: The actual dataset is imbalanced (very few fraud cases). For better visualization, you may want to sample the data or use techniques like SMOTE, but this is beyond the scope of this tutorial.

---

## Key Takeaways
- PCA reduces dimensionality by transforming data into principal components.
- Standardizing data is critical before applying PCA.
- The explained variance ratio helps determine how many components to keep.
- PCA is useful for visualization and improving model performance on high-dimensional datasets like the credit card dataset.

For further exploration:
- Experiment with different numbers of components.
- Use PCA as a preprocessing step for classification models on the credit card dataset.
- Analyze the loadings (pca.components_) to understand feature contributions.
