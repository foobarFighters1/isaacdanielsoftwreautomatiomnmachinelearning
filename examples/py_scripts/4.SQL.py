from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
import sqlite3 as sql

# The data folder is found relative to this file, so the script works from any folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
db_path = DATA_DIR / "4.SQLite3.db"
if not db_path.exists():
    # sqlite3.connect would silently create a new empty database, so check first
    raise FileNotFoundError(f"Database not found: {db_path}")

con = sql.connect(db_path)
cur = con.cursor()
col_0 = "feature"
col_1 = "target"
x = np.array(cur.execute(f"SELECT {col_0} FROM data").fetchall())
y = np.array(cur.execute(f"SELECT {col_1} FROM data").fetchall()).flatten()
con.close()
m = len(x)
print(f"Number of training examples is: {m}")
table = pd.DataFrame({
    col_0: x.flatten(),  # Flatten x for easy display
    col_1: y
})
print(table.head())
plt.scatter(x, y, marker='x', c='r')
plt.title("NESA Course Specifications Data")
plt.ylabel(f'Training {col_1}')
plt.xlabel(f'Training {col_0}')
plt.show()
my_model = LinearRegression()
my_model.fit(x, y)
y_pred = my_model.predict(x)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.title("NESA Course Specifications Data")
plt.ylabel(f'Training {col_1}')
plt.xlabel(f'Training {col_0}')
plt.show()
predict = np.array([4.5]).reshape(1, -1)
y_prediction = my_model.predict(predict)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.scatter(predict, y_prediction, marker='D', c='r', zorder=10, s=100)
# plt.text takes the x position (feature) first, then the y position (target)
plt.text(predict[0, 0], y_prediction[0], f"  Target {y_prediction[0]:.2f} is the prediction from input {predict[0, 0]}")
plt.title("NESA Course Specifications Data")
plt.ylabel(f'Training {col_1}')
plt.xlabel(f'Training {col_0}')
plt.show()
print(f"Predicted target is: {y_prediction}")
