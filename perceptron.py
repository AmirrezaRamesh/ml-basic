import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

CSV_FILE = "circles_dataset.csv"

class Perceptron:
    def __init__(self, learning_rate=0.05, n_epochs=200):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_epochs):
            for xi, yi in zip(X, y):
                score = np.dot(self.w, xi) + self.b

                if yi * score <= 0:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, -1)


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    X = df[["x1", "x2"]].values
    y = df["label"].values

    y = np.where(y == 0, -1, 1)

    return X, y


def plot_decision_boundary(model, scaler, X_train, X_test, y_train, y_test, X, accuracy):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)

    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolor="k",
        s=30,
        label="Train"
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolor="k",
        s=50,
        marker="^",
        label="Test"
    )

    plt.title(f"Perceptron Decision Boundary\nAccuracy = {accuracy:.3f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


X, y = load_dataset(CSV_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Perceptron(
    learning_rate=0.1,
    n_epochs=20
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Perceptron Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

plot_decision_boundary(
    model,
    scaler,
    X_train,
    X_test,
    y_train,
    y_test,
    X,
    accuracy
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["-1", "+1"]
)

disp.plot(cmap="Blues")
plt.title("Perceptron Confusion Matrix")
plt.show()