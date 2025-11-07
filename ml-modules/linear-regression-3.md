# 📘 Simple Linear Regression

## 🧠 1. Definition

**Simple Linear Regression (SLR)** is a **supervised learning algorithm** used to model the **relationship between two continuous variables**:

- **Independent Variable (X)** — Predictor or input  
- **Dependent Variable (Y)** — Response or output  

It assumes a **linear relationship** between X and Y, expressed as:

\[
Y = β_0 + β_1X + ε
\]

where:  
- \( β_0 \): Intercept (value of Y when X = 0)  
- \( β_1 \): Slope (change in Y for each unit change in X)  
- \( ε \): Error term (unexplained variation)

---

## 🎯 2. Goal

The goal of SLR is to find the **best-fitting line** through data points that **minimizes the difference** between the actual and predicted Y values.  
This best-fit line is called the **regression line**.

---

## ⚙️ 3. Working Principle

1. **Collect Data**  
   Gather paired data points \((X_i, Y_i)\).

2. **Assume Linear Model**  
   \[
   Y = β_0 + β_1X + ε
   \]

3. **Estimate Coefficients (β₀, β₁)**  
   Using **Ordinary Least Squares (OLS)** — minimizes the **Sum of Squared Errors (SSE)**:
   \[
   SSE = \sum (Y_i - \hat{Y_i})^2
   \]

   Coefficients are computed as:
   \[
   β_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}
   \]
   \[
   β_0 = \bar{Y} - β_1\bar{X}
   \]

4. **Make Predictions**  
   \[
   \hat{Y} = β_0 + β_1X
   \]

---

## 📊 4. Example

Predicting **exam score (Y)** based on **hours studied (X)**:

| Hours (X) | Score (Y) |
|------------|-----------|
| 1 | 45 |
| 2 | 50 |
| 3 | 55 |
| 4 | 60 |
| 5 | 65 |

Regression Line:
\[
\hat{Y} = 40 + 5X
\]

For 6 hours of study:  
\[
\text{Predicted Score} = 40 + 5(6) = \mathbf{70}
\]

---

## 📐 5. Assumptions of Simple Linear Regression

| Assumption | Description |
|-------------|--------------|
| **Linearity** | Relationship between X and Y is linear. |
| **Independence** | Observations are independent of each other. |
| **Homoscedasticity** | Constant variance of residuals across X values. |
| **Normality of Errors** | Residuals are normally distributed. |
| **No Multicollinearity** | Not applicable — only one predictor. |

---

## 📏 6. Evaluation Metrics

Used to evaluate model performance:

| Metric | Formula | Interpretation |
|---------|----------|----------------|
| **Mean Squared Error (MSE)** | \(\frac{1}{n}\sum (Y_i - \hat{Y_i})^2\) | Average squared difference between actual and predicted values. |
| **Root MSE (RMSE)** | \(\sqrt{MSE}\) | Error in the same units as Y. |
| **Mean Absolute Error (MAE)** | \(\frac{1}{n}\sum |Y_i - \hat{Y_i}|\) | Average absolute deviation from actual values. |
| **R² (Coefficient of Determination)** | \(1 - \frac{SS_{res}}{SS_{tot}}\) | Proportion of variance in Y explained by X (0–1 scale). |

---

## 📉 7. Interpretation of Coefficients

- **Intercept (β₀):** Predicted value of Y when X = 0.  
- **Slope (β₁):** Average change in Y for each 1-unit increase in X.  
  - If \( β_1 > 0 \): Positive relationship  
  - If \( β_1 < 0 \): Negative relationship

---

## 💻 8. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([45, 50, 55, 60, 65])

# Create Model
model = LinearRegression()
model.fit(X, Y)

# Predictions
Y_pred = model.predict(X)

# Coefficients
print("Intercept (β₀):", model.intercept_)
print("Slope (β₁):", model.coef_[0])

# Visualization
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()
