# 📈 Linear Regression with Gradient Descent

**Machine learning algorithms built from scratch in Python with NumPy: KNN, Linear Regression and Logistic Regression.**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn not used](https://img.shields.io/badge/scikit--learn-not%20used-lightgrey?logo=scikitlearn&logoColor=white)
![Beginner Friendly](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen)

This repository implements five classic machine learning models **from scratch**, without `scikit-learn`, to show exactly what happens inside `fit()` and `predict()`. It contains the **assignment** (a KNN Regressor) and the **lecture code** (KNN Classifier, Linear Regression with Gradient Descent and with OLS, and Logistic Regression).

Every model is explained for beginners in the same four steps:

**💡 Idea → 🧮 Formula → 💻 Code → 📤 Output**

---

## 📑 Table of Contents

1. [About the Project](#-about-the-project)
2. [Project Structure](#-project-structure)
3. [Machine Learning Basics (Start Here)](#-machine-learning-basics-start-here)
4. [Assignment: KNN Regressor from Scratch](#-assignment-knn-regressor-from-scratch)
5. [Lecture: KNN Classifier](#-lecture-knn-classifier)
6. [Lecture: Linear Regression with Gradient Descent](#-lecture-linear-regression-with-gradient-descent)
7. [Lecture: Linear Regression with OLS (Normal Equation)](#-lecture-linear-regression-with-ols-normal-equation)
8. [Lecture: Logistic Regression](#-lecture-logistic-regression)
9. [Results Summary](#-results-summary)
10. [KNN vs Linear Regression on the Same Data](#-knn-vs-linear-regression-on-the-same-data)
11. [Getting Started](#-getting-started)
12. [Tech Stack](#-tech-stack)
13. [Key Learnings](#-key-learnings)
14. [Glossary](#-glossary)
15. [References](#-references)
16. [Author](#-author)

## 🎯 About the Project

**Objective:** understand how machine learning algorithms work internally by writing every step by hand, using only NumPy for the mathematics.

**Why build from scratch?**

- Libraries hide the math behind a single line such as `model.fit(X, y)`.
- Writing each step yourself shows **how** a model learns and **why** it makes a prediction.
- Running different models on the same data reveals their strengths and limits.

**What you will learn**

- 📏 How to measure similarity with **Euclidean distance**
- 👥 How **KNN** makes predictions from the nearest neighbours
- ⛰️ How **Gradient Descent** learns step by step
- 🧮 How **OLS** solves linear regression with one formula
- 🔀 How **Logistic Regression** turns a score into a probability

## 📂 Project Structure

| File | Type | Description |
|---|---|---|
| `Scratch Implementation_Assignment.pdf` | 📄 Assignment | The assignment question: build a KNN Regressor from scratch |
| `knn_regressor_scratch.ipynb` | ✅ Assignment solution | KNN Regressor implementation and its predictions |
| `1-knn.ipynb` | 📓 Lecture code | KNN Classifier with a scatter plot of the data |
| `linear_reg.ipynb` | 📓 Lecture code | Linear Regression with Gradient Descent and with OLS |
| `logistic_reg.ipynb` | 📓 Lecture code | Logistic Regression trained with Gradient Descent |
| `README.md` | 📘 Documentation | This guide |

## 🧠 Machine Learning Basics (Start Here)

New to machine learning? Read this section first. It explains every idea and symbol used in the rest of this guide.

### What is machine learning?

Machine learning means **learning a pattern from examples** and then using that pattern to make predictions on new data. Instead of writing the rule yourself, you show the computer examples and it works out the rule.

### Features and targets

Each example has an **input**, called a *feature*, and a correct **answer**, called a *target*. This is the dataset used in the assignment:

| Input x (feature) | Answer y (target) |
|:---:|:---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
| 5 | 10 |

The pattern here is **y = 2 × x**. A model's job is to discover patterns like this from data.

### Training data and test data

- **Training data** (`X_train`, `y_train`): the examples the model learns from.
- **Test data** (`X_test`): new inputs the model has never seen, used to check its predictions.

### Regression and classification

| Type | Predicts | Example | Models in this repository |
|---|---|---|---|
| **Regression** | A number 🔢 | The price of a house | KNN Regressor, Linear Regression |
| **Classification** | A category 🏷️ | Spam or not spam (1 or 0) | KNN Classifier, Logistic Regression |

### Symbols used in this guide

| Symbol | Meaning |
|:---:|---|
| X | The inputs (features), one row per example |
| y | The correct answers (targets) |
| ŷ | "y-hat": the model's prediction |
| m | Number of training examples |
| n | Number of features in each example |
| k | Number of neighbours used by KNN |
| w, b | Weight and bias: the numbers a model learns (the slope and intercept of a line) |
| α | "Alpha": the learning rate, i.e. the size of each learning step |
| J | Cost: how wrong the model is (lower is better) |
| Σ | Capital sigma: "add up all the terms" |
| ∂J/∂w | Gradient: how much the cost changes when w changes a little |
| := | "Becomes": replace the old value with the new one |
| Xᵀ | "X transpose": X with its rows and columns swapped |
| σ(z) | Small sigma: the sigmoid function, which squeezes any number into the range 0 to 1 |

## 📝 Assignment: KNN Regressor from Scratch

> 📄 **Question:** `Scratch Implementation_Assignment.pdf` · 💻 **Solution:** `knn_regressor_scratch.ipynb`

### 📋 Problem Statement

Implement a **K-Nearest Neighbours (KNN) Regressor from scratch**, without using scikit-learn. Given a dataset of points with input features and continuous target values, the task is to:

1. Compute the distances between data points.
2. Find the **k nearest neighbours** using **Euclidean distance**.
3. Predict the output as the **average** of the neighbours' target values.

**Given data**

```python
# Training data
X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

# Testing data
X_test = [[1.5], [3.5], [6]]
```

### 💡 The Idea in Simple Words

> 🏘️ **In one line:** a new point gets its answer from the training points closest to it.

Imagine you want to guess the price of a house. A simple and sensible way is to find the **3 most similar houses** and take the **average** of their prices. KNN does exactly this with numbers, where "similar" means "close in distance".

KNN is called a **lazy learner** 😴 because it does no real training: `fit()` only stores the data, and all the work happens in `predict()`.

**How one prediction is made**

📍 New point → 📏 Distance to every training point → 🔢 Sort the distances → 👥 Keep the k closest → ➗ Average their y values → 🎯 Prediction

### 🧮 The Formulas

**Step 1: Euclidean distance** (the straight-line distance between two points)

$$d(x, x_i) = \sqrt{\sum_{j=1}^{n} \left(x_j - x_{ij}\right)^2}$$

Here x is the new point, xᵢ is the i-th training point, and j runs over the n features. In words: subtract feature by feature, square each difference, add them up, and take the square root. With a single feature, as in this assignment, the distance is simply the gap |x − xᵢ|.

**Step 2: Find the neighbours.** Sort all the distances and keep the **k smallest**. These k points are the *nearest neighbours* of x, written Nₖ(x).

**Step 3: Average their targets**

$$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i$$

In words: add up the y values of the k neighbours and divide by k.

### 🔍 Worked Example by Hand

Predicting **x = 1.5** with **k = 3**:

| Training x | y | Distance to 1.5 | One of the 3 nearest? |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 0.5 | ✅ |
| 2 | 4 | 0.5 | ✅ |
| 3 | 6 | 1.5 | ✅ |
| 4 | 8 | 2.5 | ❌ |
| 5 | 10 | 3.5 | ❌ |

$$\hat{y}(1.5) = \frac{2 + 4 + 6}{3} = \frac{12}{3} = 4$$

✅ The prediction for x = 1.5 is **4**.

### 💻 Implementation

```python
import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _predict_one(self, x):
        # Compute distances
        distances = self._euclidean_distance(self.X_train, x)

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get k nearest target values
        k_nearest_values = self.y_train[k_indices]

        # Return mean of neighbors
        return np.mean(k_nearest_values)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)
```

**How the code matches the formulas**

| Code | Step | What it does |
|---|:---:|---|
| `fit(X, y)` | – | Stores the training data (no calculation, because KNN is a lazy learner) |
| `np.sqrt(np.sum((x1 - x2) ** 2, axis=1))` | 1 | Distance from the new point to **every** training point at once |
| `np.argsort(distances)[:self.k]` | 2 | Positions of the k smallest distances |
| `self.y_train[k_indices]` | 2 | The y values of those k neighbours |
| `np.mean(k_nearest_values)` | 3 | Their average, which is the prediction |
| `predict(X)` | – | Repeats `_predict_one` for every test point |

> 💡 `x1 - x2` subtracts the test point from every row of `X_train` in one go. This NumPy feature is called **broadcasting**.

**Running the model**

```python
# Training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Test data
X_test = np.array([[1.5], [3.5], [6]])

# Create and train model
knn = KNNRegressor(k=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
print("Predictions:", y_pred)
```

### 📤 Output

```text
Predictions: [4. 6. 8.]
```

| Test x | 3 nearest neighbours (x) | Their y values | Prediction ŷ |
|:---:|:---:|:---:|:---:|
| 1.5 | 1, 2, 3 | 2, 4, 6 | **4.0** |
| 3.5 | 2, 3, 4 | 4, 6, 8 | **6.0** |
| 6 | 3, 4, 5 | 6, 8, 10 | **8.0** |

> 📌 **Note on x = 3.5:** the points x = 2 and x = 5 are both exactly 1.5 away, so they tie for third place. NumPy's default sort does not guarantee which tied point comes first; here it picked x = 2, which gives 6.0. Writing `np.argsort(distances, kind="stable")` keeps tied points in their original order (x = 2 before x = 5), so the result is guaranteed.

### 🔎 Observations

1. **KNN follows the local data.** Each prediction is built only from nearby points.
2. **KNN cannot extrapolate.** For x = 6 the true pattern (y = 2x) gives 12, but KNN predicts 8. It can only average answers it has already seen, so its predictions always stay between the smallest and largest training targets (2 and 10).
3. **The value of k changes the result**, as the next table shows.

### 🔢 Choosing the Value of k

| k | ŷ(1.5) | ŷ(3.5) | ŷ(6) | Behaviour |
|:---:|:---:|:---:|:---:|---|
| 2 | 3 | 7 | 9 | Very local: follows individual points closely |
| **3** | **4** | **6** | **8** | Assignment setting |
| 5 | 6 | 6 | 6 | Uses every point, so it always predicts the overall average (6) |

- **Small k:** flexible, but sensitive to noise and unusual points.
- **Large k:** smooth, but can wash out the real pattern.
- In practice, k is chosen by trying several values and comparing their errors.

## 👥 Lecture: KNN Classifier

> 💻 **Notebook:** `1-knn.ipynb`

### 💡 Idea

The same method as the KNN Regressor, but the answer is a **class** (0 or 1). Instead of averaging, the k neighbours **vote** 🗳️ and the majority wins.

### 🧮 Formula

$$\hat{y} = \text{mode}\left\lbrace y_i : i \in N_k(x) \right\rbrace$$

In words: the prediction is the **most common class** (the *mode*) among the k nearest neighbours.

### 📊 Data

| Point | x₁ | x₂ | Class |
|:---:|:---:|:---:|:---:|
| A | 1 | 2 | 0 |
| B | 2 | 3 | 0 |
| C | 3 | 3 | 0 |
| D | 6 | 5 | 1 |
| E | 7 | 7 | 1 |

The test points are **(2, 2)** and **(6, 6)**, with k = 3.

### 🔍 Worked Example

| Test point | 3 nearest points (distance) | Votes | Prediction |
|:---:|---|:---:|:---:|
| (2, 2) | A (1.00), B (1.00), C (1.41) | 0, 0, 0 | **0** |
| (6, 6) | D (1.00), E (1.41), C (4.24) | 1, 1, 0 | **1** (2 votes to 1) |

### 💻 Key Code

```python
def _euclidean_dist(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def _predict_one(self, x):
    distances = [self._euclidean_dist(x, x_train) for x_train in self.X_train]

    knn_indices = np.argsort(distances)[:self.k]

    knn_classes = [self.y_train[i] for i in knn_indices]

    majority_class = np.argmax(np.bincount(knn_classes))
    return majority_class
```

- `np.bincount(knn_classes)` counts the votes for each class. For example, `[1, 2]` means one vote for class 0 and two votes for class 1.
- `np.argmax(...)` returns the class with the most votes.

### 📤 Output

```text
[0 1]
```

The notebook also draws a **Seaborn scatter plot**: the training points are coloured by class and the test points are marked with a black ✕, so you can see each test point sitting next to its own class.

| | KNN Regressor | KNN Classifier |
|---|---|---|
| Predicts | A number | A class |
| Final step | **Average** of the neighbours | **Majority vote** of the neighbours |
| Distance | Euclidean | Euclidean |

## 📉 Lecture: Linear Regression with Gradient Descent

> 💻 **Notebook:** `linear_reg.ipynb` (class `LinearRegression`)

### 💡 Idea

Find the **best straight line** through the data. A line is described by two numbers:

- **Weight w** (the slope): how steep the line is.
- **Bias b** (the intercept): where the line crosses the y-axis.

The model starts with a poor guess (w = 0, b = 0) and improves it step by step.

> ⛰️ **Analogy:** you are on a foggy mountain and want to reach the lowest valley. You cannot see far, so you feel the slope under your feet and take a small step downhill, again and again, until the ground is flat. The height is the model's **error**, and the slope is the **gradient**.

### 🧮 Formulas

**1. Model (the prediction)**

$$\hat{y} = wx + b$$

For many features and all examples at once, this becomes ŷ = Xw + b.

**2. Cost: Mean Squared Error (how wrong is the line?)**

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(\hat{y}_i - y_i\right)^2$$

Squaring makes every error positive and punishes large errors more. The ½ only makes the derivative tidier. **Lower J means a better line.**

**3. Gradients (which way is downhill?)**

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \left(\hat{y} - y\right) \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}_i - y_i\right)$$

**4. Update rule (one step downhill)**

$$w := w - \alpha \frac{\partial J}{\partial w} \qquad b := b - \alpha \frac{\partial J}{\partial b}$$

Here α (alpha) is the **learning rate**: the size of each step.

### 🔁 The Training Loop

🎲 Start with w = 0, b = 0 → 🔮 Predict ŷ → 📐 Error ŷ − y → 📉 Gradients → 👣 Update w and b → 🔁 Repeat `n_iter` times

### 💻 Implementation

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y): # X_train, y_train
        m, n = X.shape # (number of samples, number of features)

        # STEP1 - initialize params
        self.bias = 0
        self.weights = np.zeros(n) # [0, 0, ..., 0]

        # Gradient Descent
        for i in range(self.n_iter):
            # STEP2 - calc y_pred
            y_pred = self.bias + np.dot(X, self.weights)

            # STEP3 - calc gradient
            db = (1/m) * np.sum(y_pred - y)
            dw = (1/m) * np.dot(X.T, (y_pred - y))

            # STEP4 - convergence theorem - params update
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, X):
        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred
```

**How the code matches the formulas**

| Step | Formula | Code |
|---|---|---|
| 1. Initialise | w = 0, b = 0 | `self.weights = np.zeros(n)` and `self.bias = 0` |
| 2. Predict | ŷ = Xw + b | `y_pred = self.bias + np.dot(X, self.weights)` |
| 3. Gradient for b | ∂J/∂b = (1/m) Σ(ŷ − y) | `db = (1/m) * np.sum(y_pred - y)` |
| 3. Gradient for w | ∂J/∂w = (1/m) Xᵀ(ŷ − y) | `dw = (1/m) * np.dot(X.T, (y_pred - y))` |
| 4. Update b | b := b − α · ∂J/∂b | `self.bias -= self.lr * db` |
| 4. Update w | w := w − α · ∂J/∂w | `self.weights -= self.lr * dw` |

Steps 2 to 4 repeat `n_iter` times. The defaults are α = 0.01 (`learning_rate`) and 1000 iterations (`n_iter`).

### 📤 Output

```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(y_pred)

print(model.bias)
print(model.weights)
```

```text
[2.06850809 4.04226297 6.01601785 7.98977273 9.96352761]
0.09475321533750963
[1.97375488]
```

The three lines are the predictions for the training inputs, the bias b and the weight w.

| Parameter | Learned value | Ideal value (y = 2x) |
|---|:---:|:---:|
| Weight w | 1.9738 | 2 |
| Bias b | 0.0948 | 0 |

✅ The model found a line very close to **y = 2x**. It is not exact because 1000 small steps with α = 0.01 are not quite enough to converge fully. If you track the cost J during training, it falls from **22.0** at the start to about **0.0008** at the end.

### ⚡ Effect of the Learning Rate α

The same data and code, run with different learning rates:

| α | Iterations | Learned w | Learned b | Result |
|:---:|:---:|:---:|:---:|---|
| 0.001 | 1000 | 1.88 | 0.43 | 🐢 Too slow: not converged yet |
| 0.01 | 1000 | 1.97 | 0.09 | 🙂 Close (notebook default) |
| 0.01 | 10000 | 2.00 | 0.00 | ✅ Converged with more steps |
| 0.1 | 1000 | 2.00 | 0.00 | ✅ Converged quickly |
| 0.2 | 1000 | – | – | 💥 Diverged: steps too large, values explode |

## 🧮 Lecture: Linear Regression with OLS (Normal Equation)

> 💻 **Notebook:** `linear_reg.ipynb` (class `LinearRegressionOLS`)

### 💡 Idea

Instead of walking downhill step by step, **OLS** (Ordinary Least Squares) jumps straight to the best line with a single formula called the **normal equation**. It needs no learning rate and no loop.

### 🧮 Formula

$$\theta = \left(X_b^T X_b\right)^{-1} X_b^T y$$

- `X_b` is X with an extra **column of 1s** at the front. This column lets the model learn the bias b.
- θ (theta) holds all the parameters: θ = [b, w₁, …, wₙ].
- ᵀ means **transpose** and ⁻¹ means **matrix inverse**.

### 💻 Implementation

```python
def fit(self, X, y):
    m, n = X.shape

    X_b = np.c_[np.ones((m, 1)), X]

    # normal equation
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    self.bias = theta[0]
    self.weights = theta[1:]
```

- `np.c_[np.ones((m, 1)), X]` adds the column of 1s.
- `@` is matrix multiplication, and `np.linalg.inv` computes the matrix inverse.

### 📤 Output

```text
[ 2.  4.  6.  8. 10.]
4.773959005888173e-15
[2.]
```

✅ A perfect fit: **w = 2** and **b ≈ 0**. The value 4.77 × 10⁻¹⁵ is a tiny floating-point rounding error, which is effectively zero.

### 🤔 Gradient Descent or OLS?

| | Gradient Descent | OLS (Normal Equation) |
|---|---|---|
| How it works | Many small steps downhill | One direct formula |
| Needs α and iterations | ✅ Yes | ❌ No |
| Result on this data | w ≈ 1.97, b ≈ 0.09 | w = 2, b ≈ 0 (exact) |
| Many features or large data | ✅ Scales well | ❌ The matrix inverse becomes slow |
| Also used in | Logistic regression, neural networks | Linear regression only |

## 🔀 Lecture: Logistic Regression

> 💻 **Notebook:** `logistic_reg.ipynb`

### 💡 Idea

Despite its name, logistic regression is used for **classification** (yes or no, 1 or 0). It computes the same kind of score as linear regression, then passes it through the **sigmoid** function, which squeezes any number into a **probability between 0 and 1**.

### 🧮 Formulas

**1. Score and probability**

$$z = Xw + b \qquad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

| z | −5 | −2 | 0 | 2 | 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| σ(z) | 0.007 | 0.119 | 0.5 | 0.881 | 0.993 |

**2. Decision rule** (threshold 0.5): predict class **1** if ŷ ≥ 0.5, otherwise class **0**.

**3. Cost: Binary Cross-Entropy (Log Loss)**

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log\left(\hat{y}_i\right) + \left(1 - y_i\right) \log\left(1 - \hat{y}_i\right) \right]$$

**4. Gradients:** the same form as linear regression; only ŷ is different.

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \left(\hat{y} - y\right) \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}_i - y_i\right)$$

### 💻 Key Code

```python
def _sigmoid(self, z):
    return (1 / (1 + np.exp(-z)))

def get_probabilities(self, X):
    z = self.bias + np.dot(X, self.weights)
    return self._sigmoid(z)

def predict(self, X, threshold=0.5):
    probabilities = self.get_probabilities(X)
    y_pred_bool = probabilities >= threshold
    return y_pred_bool.astype(int)
```

Training uses the same gradient-descent loop as linear regression, with α = 0.1 and 1000 iterations.

### 📤 Output

Data: X = [[1, 2], [2, 3], [3, 4], [4, 5]] and y = [0, 0, 1, 1]

```text
[0 0 1 1]
[0.03701722 0.04694058 0.972997   0.98153106]
```

| Input (x₁, x₂) | True class | P(class = 1) | Predicted class |
|:---:|:---:|:---:|:---:|
| (1, 2) | 0 | 0.037 | 0 ✅ |
| (2, 3) | 0 | 0.047 | 0 ✅ |
| (3, 4) | 1 | 0.973 | 1 ✅ |
| (4, 5) | 1 | 0.982 | 1 ✅ |

✅ All 4 training points are classified correctly (100% training accuracy).

## 📊 Results Summary

| # | Notebook | Algorithm | Task | Output |
|:---:|---|---|---|---|
| 1 | `knn_regressor_scratch.ipynb` | KNN Regressor (k = 3) | Regression | `[4. 6. 8.]` |
| 2 | `1-knn.ipynb` | KNN Classifier (k = 3) | Classification | `[0 1]` |
| 3 | `linear_reg.ipynb` | Linear Regression with Gradient Descent | Regression | w ≈ 1.974, b ≈ 0.095 |
| 4 | `linear_reg.ipynb` | Linear Regression with OLS | Regression | w = 2, b ≈ 0 |
| 5 | `logistic_reg.ipynb` | Logistic Regression | Classification | `[0 0 1 1]` (4 of 4 correct) |

## 🆚 KNN vs Linear Regression on the Same Data

Predictions for the assignment's test points, where the true pattern is y = 2x. The linear regression values come from running the lecture's `LinearRegression` and `LinearRegressionOLS` classes on these test points.

| Test x | True y = 2x | KNN (k = 3) | Linear Regression (GD) | Linear Regression (OLS) |
|:---:|:---:|:---:|:---:|:---:|
| 1.5 | 3 | 4 | 3.06 | 3.00 |
| 3.5 | 7 | 6 | 7.00 | 7.00 |
| 6 | 12 | 8 | 11.94 | 12.00 |

- 🏘️ **KNN is local:** it averages nearby answers, so it cannot go beyond the training range (x = 6 gives 8 instead of 12).
- 📏 **Linear regression is global:** it learns one rule for every input, so it extends the line correctly, because this data really is a straight line.
- 💡 **No model is best everywhere:** on curved (non-linear) data, a straight line would fail where KNN can still follow the shape.

## 🚀 Getting Started

### Prerequisites

- Python 3
- NumPy
- Pandas and Seaborn (only for the plot in `1-knn.ipynb`)
- Jupyter Notebook

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Preetbbhalala/Linear_Regression_with_Gradient_Descent.git
cd Linear_Regression_with_Gradient_Descent

# 2. Install the dependencies
pip install numpy pandas seaborn jupyter

# 3. Launch Jupyter
jupyter notebook
```

Open any notebook and run all cells. You can also open the assignment directly in Google Colab without installing anything:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Preetbbhalala/Linear_Regression_with_Gradient_Descent/blob/main/knn_regressor_scratch.ipynb)

## 🧰 Tech Stack

| Tool | Used for |
|---|---|
| 🐍 Python 3 | Programming language |
| 🔢 NumPy | All the math: distances, sorting, dot products and the matrix inverse |
| 🐼 Pandas | Arranging the data for the plot |
| 📊 Seaborn | Scatter plot in `1-knn.ipynb` |
| 📓 Jupyter Notebook | Running the code step by step |

## 🎓 Key Learnings

- ✅ **KNN** needs no training. It predicts from the k most similar points: an average for regression, a majority vote for classification.
- ✅ **Euclidean distance** is how "similar" is measured.
- ✅ **k** balances a very local model (small k) against an over-smoothed one (large k).
- ✅ **Gradient Descent** learns by repeating four steps: predict, measure the error, compute the gradient and step downhill.
- ✅ The **learning rate α** must be chosen carefully: too small is slow, too large diverges.
- ✅ **OLS** finds the exact linear regression solution with one formula.
- ✅ **Logistic Regression** reuses the same gradient idea, plus the sigmoid, to classify.
- ✅ Every model has limits: KNN cannot extrapolate, and a straight line cannot fit a curve.

## 📖 Glossary

| Term | Simple meaning |
|---|---|
| **Feature** | An input value used to make a prediction |
| **Target** | The correct answer the model should predict |
| **Model** | A method that turns inputs into predictions |
| **Training** | Letting the model learn from examples |
| **Lazy learner** | A model that skips training and does all its work at prediction time, like KNN |
| **Euclidean distance** | The straight-line distance between two points |
| **Parameter** | A value the model learns, such as w and b |
| **Hyperparameter** | A setting you choose before training, such as k or α |
| **Cost function** | A score of how wrong the model is |
| **Gradient** | The slope of the cost; it points uphill, so the model steps the opposite way |
| **Learning rate (α)** | The size of each gradient-descent step |
| **Convergence** | The point where more steps no longer change the parameters much |
| **Sigmoid** | A function that squeezes any number into the range 0 to 1 |
| **Extrapolation** | Predicting outside the range of the training data |

## 🔗 References

- [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), Wikipedia
- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), Wikipedia
- [Linear regression](https://en.wikipedia.org/wiki/Linear_regression), Wikipedia
- [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares), Wikipedia
- [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), Wikipedia
- [NumPy documentation](https://numpy.org/doc/stable/)

## 👤 Author

**Preet Bhalala** · GitHub: [@Preetbbhalala](https://github.com/Preetbbhalala)

⭐ If this repository helped you understand these algorithms, consider giving it a star!
