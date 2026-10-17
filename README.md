# Linear Regression with Gradient Descent (From Scratch)

A beginner-friendly project that builds **Linear Regression** from scratch using **Gradient Descent**, with only NumPy for the math. Also included are related from-scratch notebooks for k-NN and Logistic Regression.

---

## 📁 Repository Contents

| File | What it is |
|---|---|
| `Scratch Implementation_Assignment.pdf` | The assignment instructions |
| `linear_reg.ipynb` | Linear Regression with Gradient Descent |
| `logistic_reg.ipynb` | Logistic Regression (classification) |
| `1-knn.ipynb` | k-Nearest Neighbors (lecture code) |
| `knn_regressor_scratch.ipynb` | k-NN Regressor implemented from scratch |

---

## 🎯 Goal

Given data points, find the straight line that best fits them, **without using ready-made ML libraries** such as `sklearn.LinearRegression`. We learn the line's parameters by repeatedly improving a guess, which is exactly what Gradient Descent does.

---

## 🧠 Core Ideas (Beginner Friendly)

### 1. Linear Regression: the model

We predict a number $\hat{y}$ ("y-hat") from an input $x$ using a line:

$$\hat{y} = \theta_0 + \theta_1 x$$

| Symbol | Meaning |
|---|---|
| $x$ | Input feature (e.g. house size) |
| $y$ | True value (e.g. real price) |
| $\hat{y}$ | Predicted value |
| $\theta_0$ | **Intercept** (bias): where the line crosses the y-axis |
| $\theta_1$ | **Slope** (weight): how much $\hat{y}$ changes when $x$ increases by 1 |

For many features, in vector form:

$$\hat{y} = \mathbf{X}\boldsymbol{\theta}$$

where $\mathbf{X}$ is the data matrix (with a column of 1s for the intercept) and $\boldsymbol{\theta}$ is the vector of parameters.

### 2. Cost Function: "How wrong are we?"

We measure the error with the **Mean Squared Error (MSE)**:

$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

- $m$ = number of training examples
- $\hat{y}^{(i)} - y^{(i)}$ = error for the $i$-th example
- Squaring makes every error positive and punishes big mistakes more
- The $\frac{1}{2}$ is only there to make the derivative cleaner

**Lower $J$ = better line.**

### 3. Gradient Descent: "How do we get less wrong?"

Imagine standing on a hill in fog and wanting to reach the valley. You feel which way the ground slopes down and take a small step that way, then repeat. The **gradient** $\nabla J$ is that slope.

**Update rule** (repeat until convergence):

$$\theta_j := \theta_j - \alpha \,\frac{\partial J}{\partial \theta_j}$$

where, for linear regression,

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

Vectorized (what we actually code):

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \frac{\alpha}{m}\,\mathbf{X}^{\top}\left(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\right)$$

| Symbol | Meaning |
|---|---|
| $:=$ | "Update to" (assignment, not equality) |
| $\alpha$ | **Learning rate**: the size of each step |
| $\frac{\partial J}{\partial \theta_j}$ | Partial derivative: how $J$ changes if only $\theta_j$ changes |
| $\mathbf{X}^{\top}$ | Transpose of $\mathbf{X}$ |

**Choosing $\alpha$:**
- Too **small**: learns very slowly 🐢
- Too **large**: overshoots the minimum and can diverge 💥
- Just right: cost decreases smoothly ✅

### 4. Feature Scaling (Normalization)

When features have very different ranges (e.g. 1–5 vs 1000–5000), gradient descent zig-zags and is slow. We standardize each feature:

$$x' = \frac{x - \mu}{\sigma}$$

- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

---

## 🔁 Algorithm Summary

```
1. Load data and (optionally) scale features
2. Add a column of 1s to X (for the intercept θ₀)
3. Initialize θ (e.g. all zeros)
4. Repeat for N iterations:
     a. Predict:    ŷ = Xθ
     b. Error:      e = ŷ − y
     c. Gradient:   ∇J = (1/m) · Xᵀe
     d. Update:     θ = θ − α · ∇J
     e. Record J(θ) to track progress
5. Plot cost vs. iterations and the fitted line
```

### Minimal Python sketch

```python
import numpy as np

def gradient_descent(X, y, alpha=0.01, n_iters=1000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for _ in range(n_iters):
        y_hat = X @ theta                       # predictions
        error = y_hat - y                       # residuals
        gradient = (X.T @ error) / m            # ∇J(θ)
        theta -= alpha * gradient               # update step
        cost_history.append((error @ error) / (2 * m))  # J(θ)

    return theta, cost_history
```

> This is a teaching sketch. See `linear_reg.ipynb` for the full implementation used in this project.

---

## 📊 Evaluating the Model

| Metric | Formula | Meaning |
|---|---|---|
| **MSE** | $\frac{1}{m}\sum (y-\hat{y})^2$ | Average squared error (lower is better) |
| **RMSE** | $\sqrt{\text{MSE}}$ | Error in the same units as $y$ |
| **R² score** | $1 - \dfrac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2}$ | Fraction of variance explained (1 = perfect) |

---

## 🧩 Other Notebooks in This Repo

- **k-NN (`1-knn.ipynb`)**: predicts using the $k$ closest training points, with distance $d(\mathbf{a},\mathbf{b}) = \sqrt{\sum_j (a_j - b_j)^2}$.
- **k-NN Regressor (`knn_regressor_scratch.ipynb`)**: predicts the *average* $y$ of the $k$ nearest neighbours.
- **Logistic Regression (`logistic_reg.ipynb`)**: same gradient descent idea, but uses the sigmoid $\sigma(z) = \dfrac{1}{1+e^{-z}}$ to output probabilities for classification.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Preetbbhalala/Linear_Regression_with_Gradient_Descent.git
cd Linear_Regression_with_Gradient_Descent

# 2. Install dependencies
pip install numpy pandas matplotlib jupyter

# 3. Launch Jupyter and open a notebook
jupyter notebook linear_reg.ipynb
```

**Requirements:** Python 3.8+, NumPy, Pandas, Matplotlib, Jupyter.

---

## 📈 Results

> *(Fill in with your own numbers after running the notebook.)*

| Setting | Value |
|---|---|
| Learning rate $\alpha$ | _e.g. 0.01_ |
| Iterations | _e.g. 1000_ |
| Final cost $J(\boldsymbol{\theta})$ | _…_ |
| Learned $\theta_0, \theta_1$ | _…_ |
| R² score | _…_ |

*Insert your cost-vs-iteration plot and fitted-line plot here.*

---

## 🔍 Key Takeaways

- Linear regression fits a line by **minimizing the MSE cost** $J(\boldsymbol{\theta})$.
- Gradient descent moves $\boldsymbol{\theta}$ **opposite to the gradient** until the cost stops decreasing.
- The **learning rate** $\alpha$ and **feature scaling** strongly affect convergence.
- A from-scratch version should match scikit-learn's results closely, which makes a good sanity check.

---

## 📚 References

- Course lecture notebooks and the provided assignment PDF
- Andrew Ng, *Machine Learning* (Stanford / Coursera): Linear Regression & Gradient Descent
- [scikit-learn documentation: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

---

## 👤 Author

**Preet Bhalala**: [GitHub](https://github.com/Preetbbhalala)
