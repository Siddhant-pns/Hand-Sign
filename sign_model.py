import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os

class SignModel:
    def __init__(self, model_file="sign_model.pkl"):
        self.model_file = model_file
        self.data = []
        self.labels = []
        self.model = None
        self.load_model()

    def load_model(self):
        """ Load existing trained model or create a new one """
        if os.path.exists(self.model_file):
            with open(self.model_file, "rb") as f:
                loaded_data = pickle.load(f)
                self.data = loaded_data.get("data", [])
                self.labels = loaded_data.get("labels", [])
                if len(self.data) > 0:  # Ensure data exists before training
                    self.train_model()
        else:
            print("⚠ No existing model found. Starting fresh.")

    def save_model(self):
        """ Save accumulated training data & trained model persistently """
        with open(self.model_file, "wb") as f:
            pickle.dump({"data": self.data, "labels": self.labels}, f)

    def train_model(self):
        """ Train the classifier dynamically only if data exists """
        if len(self.data) > 0:
            self.model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
            self.model.fit(self.data, self.labels)
            self.save_model()  # Save after retraining

    def add_new_sign(self, features, label):
        """ Add new sign dynamically & retrain model """
        if label not in self.labels:  # Prevent duplicate entries
            self.data.append(features)
            self.labels.append(label)
            self.train_model()
        else:
            print(f"⚠ '{label}' already exists.")

    def predict(self, features):
        """ Predict sign and allow small variations """
        if self.model is not None:
            distances, indices = self.model.kneighbors([features], n_neighbors=1)
            min_distance = distances[0][0]

            if min_distance < 0.15:  # Threshold for variation
                return self.labels[indices[0][0]]
            else:
                return "Unknown"
        return "Unknown"
