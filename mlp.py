import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


CSV_FILE = "circles_dataset.csv"


# Load data
df = pd.read_csv(CSV_FILE)

X = df[["x1", "x2"]].values
y = df["label"].values


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train model
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Evaluate model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("MLP Accuracy:", accuracy)
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

Z = model.predict(grid_points_scaled)
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

plt.title(f"MLP Decision Boundary\nAccuracy = {accuracy:.3f}")
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
plt.title("MLP Confusion Matrix")
plt.show()