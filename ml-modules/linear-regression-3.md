# 📘 Simple Linear Regression

## 1. Definition

**Simple Linear Regression (SLR)** is a **supervised machine learning algorithm** used to model the **relationship between two continuous variables**:

- **Independent variable (X)** — the predictor or input  
- **Dependent variable (Y)** — the response or output  

It assumes a **linear relationship** between X and Y, expressed as:

\[
Y = β_0 + β_1X + ε
\]

where:
- \( β_0 \) = intercept (value of Y when X = 0)  
- \( β_1 \) = slope (change in Y for one unit change in X)  
- \( ε \) = error term (unexplained variation)

---

## 2. Goal

The goal of SLR is to find the **best-fit line** through the data points that minimizes the error between predicted and actual values of Y.

This line is called the **regression line**.

---

## 3. Working Principle

1. **Collect Data**  
   Gather paired data points \((X_i, Y_i)\).

2. **Assume Linear Model:**  
   \( Y = β_0 + β_1X + ε \)

3. **Estimate Coefficients (β₀, β₁):**  
   Using **Ordinary Least Squares (OLS)** method, which minimizes the **Sum of Squared Errors (SSE)**:

   \[
   SSE = \sum (Y_i - \hat{Y_i})^2
   \]

   OLS Formulas:
   \[
   β_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}
   \]
   \[
   β_0 = \bar{Y} - β_1\bar{X}
   \]

4. **Make Predictions:**  
   \[
   \hat{Y} = β_0 + β_1X
   \]

---

## 4. Example

Relationship between **hours studied (X)** and **exam score (Y)**:

| Hours (X) | Score (Y) |
|------------|------------|
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
Predicted Score = 40 + 5×6 = **70**

---

## 5. Assumptions of Simple Linear Regression

| Assumption | Description |
|-------------|--------------|
| **Linearity** | Relationship between X and Y is linear. |
| **Independence** | Observations are independent of each other. |
| **Homoscedasticity** | Constant variance of residuals across X values. |
| **Normality of Errors** | Residuals are normally distributed. |
| **No Multicollinearity** | (Not applicable here; only one predictor.) |

---

## 6. Evaluation Metrics

To evaluate model performance:

1. **Mean Squared Error (MSE):**  
   \[
   MSE = \frac{1}{n}\sum (Y_i - \hat{Y_i})^2
   \]

2. **Root Mean Squared Error (RMSE):**  
   \[
   RMSE = \sqrt{MSE}
   \]

3. **Mean Absolute Error (MAE):**  
   \[
   MAE = \frac{1}{n}\sum |Y_i - \hat{Y_i}|
   \]

4. **R² (Coefficient of Determination):**  
   Measures how much variation in Y is explained by X.

   \[
   R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
   \]

   - \( SS_{res} \): residual sum of squares  
   - \( SS_{tot} \): total sum of squares  
   - \( R^2 \) ranges from 0 to 1 (higher = better fit)

---

## 7. Interpretation of Coefficients

- **Intercept (β₀):** Predicted Y when X = 0  
- **Slope (β₁):** Average change in Y for each unit change in X  
  - If β₁ > 0 → Positive relationship  
  - If β₁ < 0 → Negative relationship  

---

## 8. Implementation (Python Example)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([45, 50, 55, 60, 65])

# Create model
model = LinearRegression()
model.fit(X, Y)

# Predictions
Y_pred = model.predict(X)

# Coefficients
print("Intercept (β₀):", model.intercept_)
print("Slope (β₁):", model.coef_[0])

# Visualization
plt.scatter(X, Y, color='blue', label='Actual data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()
