import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


CSV_FILE = "xor_dataset.csv"

class RBFNetwork:
    def __init__(self, num_centers=20, gamma=1.0, output_classes=2):
        self.num_centers = num_centers
        self.gamma = gamma
        self.output_classes = output_classes
        self.centers = None
        self.weights = None

    def _rbf_features(self, X):
        diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        squared_distances = np.sum(diff ** 2, axis=2)
        return np.exp(-self.gamma * squared_distances)

    def fit(self, X, y):
        kmeans = KMeans(
            n_clusters=self.num_centers,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        Phi = self._rbf_features(X)

        y_onehot = np.zeros((len(y), self.output_classes))
        y_onehot[np.arange(len(y)), y] = 1

        self.weights = np.linalg.pinv(Phi) @ y_onehot

    def predict(self, X):
        Phi = self._rbf_features(X)
        scores = Phi @ self.weights
        return np.argmax(scores, axis=1)


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

    plt.title(f"RBF Network Decision Boundary\nAccuracy = {accuracy:.3f}")
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
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RBFNetwork(
    num_centers=20,
    gamma=1.0,
    output_classes=2
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("RBF Network Accuracy:", accuracy)
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
plt.title("RBF Network Confusion Matrix")
plt.show()