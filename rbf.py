import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


CSV_FILE = "circles_dataset.csv"
N_CENTERS = 20
GAMMA = 1.0


def rbf_features(X, centers, gamma):
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    squared_distances = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * squared_distances)


# Load data
df = pd.read_csv(CSV_FILE)

X = df[["x1", "x2"]].values
y = df["label"].values


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Find RBF centers
kmeans = KMeans(
    n_clusters=N_CENTERS,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)
centers = kmeans.cluster_centers_


# Build RBF features
Phi_train = rbf_features(X_train_scaled, centers, GAMMA)
Phi_test = rbf_features(X_test_scaled, centers, GAMMA)


# Train linear model
model = LogisticRegression(max_iter=1000)
model.fit(Phi_train, y_train)


# Evaluate model
y_pred = model.predict(Phi_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("RBF Network Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)


# Create decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)
grid_phi = rbf_features(grid_points_scaled, centers, GAMMA)

Z = model.predict(grid_phi)
Z = Z.reshape(xx.shape)


# Plot decision boundary
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


# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")
plt.title("RBF Network Confusion Matrix")
plt.show()