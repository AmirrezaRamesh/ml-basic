import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


CSV_FILE = "circles_dataset.csv"


class MLP:
    def __init__(self, learning_rate=0.05, n_epochs=10000, hidden_units=20, output_classes=2):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.hidden_units = hidden_units
        self.output_classes = output_classes

    def _initialize_weights(self, input_dim):
        limit1 = np.sqrt(6 / (input_dim + self.hidden_units))
        limit2 = np.sqrt(6 / (self.hidden_units + self.output_classes))

        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, self.hidden_units))
        self.b1 = np.zeros((1, self.hidden_units))

        self.W2 = np.random.uniform(-limit2, limit2, (self.hidden_units, self.output_classes))
        self.b2 = np.zeros((1, self.output_classes))

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - x ** 2

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, input_dim = X.shape
        self._initialize_weights(input_dim)

        y_onehot = np.zeros((n_samples, self.output_classes))
        y_onehot[np.arange(n_samples), y] = 1

        for _ in range(self.n_epochs):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._tanh(z1)

            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._softmax(z2)

            output_error = a2 - y_onehot

            dW2 = np.dot(a1.T, output_error) / n_samples
            db2 = np.sum(output_error, axis=0, keepdims=True) / n_samples

            hidden_error = np.dot(output_error, self.W2.T) * self._tanh_derivative(a1)

            dW1 = np.dot(X.T, hidden_error) / n_samples
            db1 = np.sum(hidden_error, axis=0, keepdims=True) / n_samples

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._softmax(z2)

        return np.argmax(a2, axis=1)


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    X = df[["x1", "x2"]].values
    y = df["label"].values.astype(int)

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

    plt.title(f"MLP Decision Boundary\nAccuracy = {accuracy:.3f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


np.random.seed(42)

X, y = load_dataset(CSV_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLP(
    learning_rate=0.05,
    n_epochs=10000,
    hidden_units=20,
    output_classes=2
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("MLP Accuracy:", accuracy)
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
    display_labels=["0", "1"]
)

disp.plot(cmap="Blues")
plt.title("MLP Confusion Matrix")
plt.show()