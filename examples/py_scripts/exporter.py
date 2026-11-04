from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Folders are found relative to this file, so the script works from any folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def save_model():
    # Create a folder for the output files
    OUTPUT_DIR.mkdir(exist_ok=True)
    training_data = pd.read_csv(DATA_DIR / "3.course_specifications_data.csv", encoding="utf-8-sig")
    x = training_data[["Feature"]].to_numpy()
    y = training_data["Target"].to_numpy()
    # Create and train the model
    my_model = LinearRegression()
    my_model.fit(x, y)
    filename = OUTPUT_DIR / "my_saved_model.sav"
    with open(filename, 'wb') as f:
        pickle.dump(my_model, f)
    print(f"Model saved as {filename}")
    return filename


if __name__ == "__main__":
    save_model()
