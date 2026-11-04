import pickle
import numpy as np
import exporter

# Train the model and save it to the output folder
filename = exporter.save_model()

# Load the saved model and use it to make a prediction
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)
predict = np.array([4]).reshape(1, -1)
result = loaded_model.predict(predict)
print(result[0])
