import numpy as np
from sklearn.linear_model import LinearRegression


# Training data: features (x) and targets (y)
x = np.array([[2], [4], [6], [8], [10], [12], [14], [16]])
y = np.array([1, 3, 5, 7, 9, 11, 13, 15])
model = LinearRegression()
model.fit(x, y)

# Predict the target for a new feature value
test = np.array([[4]])
y_prediction = model.predict(test)
print(f"Predicted target is: {y_prediction}")
