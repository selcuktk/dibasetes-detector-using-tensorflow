import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# to avoid some warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Main:
    @staticmethod
    def main():

        # Load the CSV data into a NumPy array
        data = np.genfromtxt("health care diabetes.csv", delimiter=',', skip_header=1)

        # Separate features (X) and target (Y)
        X = data[:, :-1]  # All rows, all columns except the last (for features)
        Y = data[:, -1]   # All rows, last column (for target variable)

        # Convert to float32 to reduce computation cost
        X = X.astype("float32")
        Y = Y.astype("float32")
        

        # Compute mean and standard deviation of X before normalization
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)

        # Normalize features using Min-Max Scaling
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # First, split into training (70%) and temp (30%) which will be used for dev + test
        x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Split the remaining 30% into dev (15%) and test (15%)
        x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        # Check the shapes
        print("x_train shape:", x_train.shape)
        print("x_dev shape:", x_dev.shape)
        print("x_test shape:", x_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_dev shape:", y_dev.shape)
        print("y_test shape:", y_test.shape)

        # Print mean and standard deviation for later use
        print("Mean of X:", X_mean)
        print("Standard Deviation of X:", X_std)

        # Sequential API (Very convenient, not very flexible)
        starte_model = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(1, activation='sigmoid'),
            ]
        )

        starte_model.compile(
            loss='binary_crossentropy',
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
        )
        starte_model.fit(x_train, y_train, batch_size=x_train.shape[0], epochs=500, verbose=2)
        starte_model.evaluate(x_dev, y_dev, verbose=2)
        starte_model.evaluate(x_test, y_test, verbose=2)

if __name__ == "__main__":
    Main.main()