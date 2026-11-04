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
m = len(x)
print(f"Number of training examples is: {m}")
print(training_data.head())
plt.scatter(x, y, marker='x', c='r')
plt.title("NESA Course Specifications Data")
plt.ylabel('Training Target')
plt.xlabel('Training Feature')
plt.show()
my_model = LinearRegression()
my_model.fit(x, y)
y_pred = my_model.predict(x)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.title("NESA Course Specifications Data")
plt.ylabel('Training Target')
plt.xlabel('Training Feature')
plt.show()
predict = np.array([4.5]).reshape(1, -1)
y_prediction = my_model.predict(predict)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.scatter(predict, y_prediction, marker='D', c='r', zorder=10, s=100)
# plt.text takes the x position (feature) first, then the y position (target)
plt.text(predict[0, 0], y_prediction[0], f"  Target {y_prediction[0]:.2f} is the prediction from input {predict[0, 0]}")
plt.title("NESA Course Specifications Data")
plt.ylabel('Training Target')
plt.xlabel('Training Feature')
plt.show()
print(f"Predicted target is: {y_prediction}")
