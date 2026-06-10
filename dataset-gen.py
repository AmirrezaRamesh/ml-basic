import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles


def make_noisy_xor(n_samples=1000, noise=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = (X[:, 0] * X[:, 1] < 0).astype(int)

    X = X + rng.normal(0, noise, size=X.shape)

    return X, y


def to_dataframe(X, y):
    return pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "label": y
    })


# Generate datasets
X_xor, y_xor = make_noisy_xor(n_samples=600, noise=0.2, random_state=42)

X_moons, y_moons = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)

X_circles, y_circles = make_circles(
    n_samples=1000,
    noise=0.1,
    factor=0.45,
    random_state=42
)


# Convert to DataFrame
xor_df = to_dataframe(X_xor, y_xor)
moons_df = to_dataframe(X_moons, y_moons)
circles_df = to_dataframe(X_circles, y_circles)


# Save datasets
xor_df.to_csv("xor_dataset.csv", index=False)
moons_df.to_csv("two_moons_dataset.csv", index=False)
circles_df.to_csv("circles_dataset.csv", index=False)


print("Datasets saved successfully.")
print("XOR shape:", xor_df.shape)
print("Two Moons shape:", moons_df.shape)
print("Circles shape:", circles_df.shape)