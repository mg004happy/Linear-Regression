# Linear-Regression in multi variable
# Linear Regression from Scratch (NumPy)

This project demonstrates a simple implementation of **Linear Regression** using NumPy — without using any machine learning libraries like scikit-learn or TensorFlow.

## 🔍 Description

We generate synthetic data with **3 features** and define a linear relationship between them. Using **gradient descent**, the model learns the optimal weights (`w`) and bias (`b`) to minimize the Mean Squared Error (MSE) between the predictions and actual labels.

---

## 📌 Features
- Manual implementation of linear regression with gradient descent.
- Cost (loss) calculation using MSE.
- Dynamic updating of weights and bias.
- Cost curve visualization over training iterations.

---

## 🧪 Data

Synthetic data generated with:
- 100 samples
- 3 features per sample
- True relationship:
  \[
  y = 2x_1 + 3x_2 + 4x_3 + 1 + \text{noise}
  \]

---

## 📈 Plot

- Scatter plot: shows relation between first feature and target `y_train`.
- Line plot: shows how the cost decreases over 500 iterations.

---

## 💻 Code Explanation

```python
# Generate random data
x_train = np.random.rand(100, 3)
y_train = 2*x_train[:, 0] + 3*x_train[:, 1] + 4*x_train[:, 2] + 1 + noise

# Initialize weights and bias
w = [[1.1], [2.1], [3.1]]
b = 0

# Gradient Descent Loop
for i in range(500):
    y_pred = x_train @ w + b
    cost = MSE
    update weights and bias using gradients
