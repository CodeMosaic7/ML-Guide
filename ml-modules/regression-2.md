# REGRESSION

### Overview

Regression is a **supervised learning technique** used to model the relationship between a **dependent variable (target)** and one or more **independent variables (features)**.
It is primarily used for **prediction and forecasting**, where the output is **continuous** (e.g., predicting prices, temperatures, or growth rates).

---

### Table of Contents

1. [Introduction to Regression](#1-introduction-to-regression)
2. [Types of Regression](#2-types-of-regression)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Implementation in Python](#5-implementation-in-python)
6. [Projects and Notebooks](#6-projects-and-notebooks)
7. [References](#7-references)

---

## 1. Introduction to Regression

Regression helps us understand **how changes in input variables affect the output variable**.

📌 Example: Predicting house price using features like area, number of rooms, and location.

**Equation:**
[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon
]

Where:

* ( y ): predicted value
* ( x_i ): feature variables
* ( \beta_i ): coefficients
* ( \epsilon ): error term

---

## 2. Types of Regression

| Type                           | Description                                       | Example Use Case                        | Notebook Link                                                           |
| ------------------------------ | ------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| **Linear Regression**          | Fits a straight line to data                      | Predicting house prices                 | [Linear Regression Notebook](notebooks/linear_regression.ipynb)         |
| **Multiple Linear Regression** | Uses multiple predictors                          | Predicting sales from multiple features | [Multiple Regression Notebook](notebooks/multiple_regression.ipynb)     |
| **Polynomial Regression**      | Models nonlinear relationships                    | Predicting growth curves                | [Polynomial Regression Notebook](notebooks/polynomial_regression.ipynb) |
| **Ridge & Lasso Regression**   | Regularized linear models to prevent overfitting  | Predicting energy consumption           | [Regularization Notebook](notebooks/ridge_lasso.ipynb)                  |
| **Logistic Regression**        | Used for binary classification (despite the name) | Spam detection                          | [Logistic Regression Notebook](notebooks/logistic_regression.ipynb)     |

---

## 3. Mathematical Foundation

### Cost Function

Mean Squared Error (MSE):
[
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
]

### Optimization

Model parameters are optimized using **Gradient Descent**:
[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
]

📓 See step-by-step derivation: [Gradient Descent Explained](notebooks/gradient_descent_derivation.ipynb)

---

## 4. Evaluation Metrics

| Metric                        | Formula                               | Description                           |   |                             |
| ----------------------------- | ------------------------------------- | ------------------------------------- | - | --------------------------- |
| **MAE (Mean Absolute Error)** | (\frac{1}{n}\sum                      | y_i - \hat{y_i}                       | ) | Average absolute difference |
| **MSE (Mean Squared Error)**  | (\frac{1}{n}\sum (y_i - \hat{y_i})^2) | Penalizes larger errors               |   |                             |
| **RMSE (Root MSE)**           | (\sqrt{MSE})                          | Interpretable in same units as target |   |                             |
| **R² Score**                  | (1 - \frac{SS_{res}}{SS_{tot}})       | Measures model fit quality            |   |                             |

Try: [Regression Metrics Comparison Notebook](notebooks/regression_metrics.ipynb)

---

## 5. Implementation in Python

**Libraries Used:**

* `pandas`, `numpy` for data handling
* `matplotlib`, `seaborn` for visualization
* `scikit-learn` for model building

Example: `linear_regression.ipynb`

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression().fit(X_train, y_train)

pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))
```

---

## Projects and Notebooks

| Project                                   | Description                                         | Notebook                                                            |
| ----------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| House Price Prediction                 | Predict house prices using multiple regression      | [House Price Project](notebooks/project_house_price.ipynb)          |
| Energy Consumption Forecasting          | Use Ridge/Lasso regression to predict power usage   | [Energy Forecast Notebook](notebooks/project_energy_forecast.ipynb) |
| Polynomial Curve Fitting               | Nonlinear data modeling using polynomial regression | [Curve Fitting Notebook](notebooks/project_polynomial_fit.ipynb)    |
| Logistic Regression for Classification | Classify whether a customer will buy a product      | [Customer Purchase Notebook](notebooks/project_logistic.ipynb)      |

---

### 🔗 Navigation

← [Back to ML Basics](../README.md) | [Next → Classification](classification.md)

---