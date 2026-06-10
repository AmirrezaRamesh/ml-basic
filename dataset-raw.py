import pandas as pd
import matplotlib.pyplot as plt


def show_dataset(name, file_path):
    df = pd.read_csv(file_path)

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("\nShape:")
    print(df.shape)
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\n")

    return df


xor_df = show_dataset("Noisy XOR Dataset", "xor_dataset.csv")
moons_df = show_dataset("Two Moons Dataset", "two_moons_dataset.csv")
circles_df = show_dataset("Circles Dataset", "circles_dataset.csv")


plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(xor_df["x1"], xor_df["x2"], c=xor_df["label"], cmap="coolwarm", s=20)
plt.title("Noisy XOR")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(moons_df["x1"], moons_df["x2"], c=moons_df["label"], cmap="coolwarm", s=20)
plt.title("Two Moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(circles_df["x1"], circles_df["x2"], c=circles_df["label"], cmap="coolwarm", s=20)
plt.title("Circles")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)

plt.tight_layout()
plt.show()