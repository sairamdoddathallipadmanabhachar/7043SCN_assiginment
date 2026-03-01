import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results():

    csv_path = "results/results.csv"

    if not os.path.exists(csv_path):
        print("No results.csv found in results/")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("results.csv is empty.")
        return

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))

    labels = []
    values = []

    for _, row in df.iterrows():
        label = f"{row['model'].split('/')[-1]} vs {row['opponent']}"
        labels.append(label)
        values.append(row['win_rate'])

    plt.bar(labels, values)

    plt.ylabel("Win Rate")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)

    save_path = "results/plots/comparison.png"
    plt.tight_layout()
    plt.savefig(save_path)

    print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_results()