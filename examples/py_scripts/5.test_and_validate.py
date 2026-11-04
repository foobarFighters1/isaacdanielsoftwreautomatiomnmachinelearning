from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression

# The data folder is found relative to this file, so the script works from any folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

training_data = pd.read_csv(DATA_DIR / "3.course_specifications_data.csv", encoding="utf-8-sig")
x = training_data[["Feature"]].to_numpy()
y = training_data["Target"].to_numpy()

# Create the model
my_model = LinearRegression()
# Fit the model to the training data
my_model.fit(x, y)

testing_data = pd.read_csv(DATA_DIR / "6.model_test_and_validate_data.csv", encoding="utf-8-sig")
x_test = testing_data[["Feature"]].to_numpy()
y_test = testing_data["Target"].to_numpy()

# Plot the model (line) against the testing data (crosses)
y_pred = my_model.predict(x)
plt.plot(x, y_pred, label="Model")
plt.scatter(x_test, y_test, marker='x', c='r', label="Testing data")
plt.title("NESA Course Specifications Data")
plt.ylabel('Target')
plt.xlabel('Feature')
plt.legend()
plt.show()

train_score = my_model.score(x, y)
test_score = my_model.score(x_test, y_test)
print(f'Training data score: {train_score}')
print(f'Testing data score: {test_score}')

test_predictions = my_model.predict(x_test)
table = pd.DataFrame({
    'Feature': x_test.flatten(),  # Flatten x for easy display
    'Target': y_test,
    'Predicted result': test_predictions.round(2),
    'Error': (y_test - test_predictions).round(2)  # Target minus prediction
})
print(table)

print(f'y-intercept: {my_model.intercept_}')
print(f'Coefficient: {my_model.coef_}')
