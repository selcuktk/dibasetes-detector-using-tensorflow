import numpy as np
from sklearn.model_selection import train_test_split


class Main:
    @staticmethod
    def main():

        # Load the CSV data into a NumPy array
        data = np.genfromtxt("health care diabetes.csv", delimiter=',', skip_header=1)

        # Separate features (X) and target (Y)
        X = data[:, :-1]  # All rows, all columns except the last (for features)
        Y = data[:, -1]   # All rows, last column (for target variable)
        X = X.T

        # Reshaping the target array (y)
        Y = Y.reshape(1, Y.size)

        print(X.shape)
        print("--------------")
        print(Y.shape)
        print("--------------")

        # First, split into training (70%) and temp (30%) which will be used for dev + test
        x_train, x_temp, y_train, y_temp = train_test_split(X.T, Y.T, test_size=0.3, random_state=42)

        # Split the remaining 30% into dev (15%) and test (15%)
        x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        # Transpose back to match the required shape
        x_train = x_train.T
        x_dev = x_dev.T
        x_test = x_test.T
        y_train = y_train.T
        y_dev = y_dev.T
        y_test = y_test.T

        # Check the shapes
        print("x_train shape:", x_train.shape)  # (8, train_samples)
        print("x_dev shape:", x_dev.shape)      # (8, dev_samples)
        print("x_test shape:", x_test.shape)    # (8, test_samples)
        print("y_train shape:", y_train.shape)  # (1, train_samples)
        print("y_dev shape:", y_dev.shape)      # (1, dev_samples)
        print("y_test shape:", y_test.shape)    # (1, test_samples)



        
        
            


if __name__ == "__main__":
    Main.main()