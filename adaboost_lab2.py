import numpy as np
import pandas as pd
import math

# --------------------------------
# Health Insurance Dataset (Lab-2)
# --------------------------------
data = {
    "Age": [38, 52, 45, 29, 61],
    "Income": [420000, 360000, 780000, 300000, 500000],
    "Smoking": [0, 5, 0, 12, 8],
    "y": [-1, +1, +1, +1, +1]   # Illness: No = -1, Yes = +1
}

df = pd.DataFrame(data)

# Initial sample weights
n = len(df)
df["w"] = 1 / n

# --------------------------------
# Decision Stumps
# --------------------------------
def stump_smoking(row):
    return +1 if row["Smoking"] >= 1 else -1

def stump_age(row):
    return +1 if row["Age"] >= 45 else -1

stumps = [stump_smoking, stump_age]
learners = []
alphas = []

# --------------------------------
# AdaBoost Training
# --------------------------------
for t, stump in enumerate(stumps):
    predictions = df.apply(stump, axis=1)

    # Weighted error
    error = sum(df["w"][predictions != df["y"]])

    # Alpha calculation
    alpha = 0.5 * math.log((1 - error) / error)

    learners.append(stump)
    alphas.append(alpha)

    # Update weights
    df["w"] = df["w"] * np.exp(-alpha * df["y"] * predictions)

    # Normalize weights
    df["w"] = df["w"] / df["w"].sum()

    print(f"\nROUND {t+1}")
    print("Weighted Error:", error)
    print("Alpha:", alpha)
    print("Updated Weights:")
    print(df["w"])

# --------------------------------
# Final Strong Classifier
# --------------------------------
def strong_classifier(row):
    score = 0
    for alpha, stump in zip(alphas, learners):
        score += alpha * stump(row)
    return np.sign(score)

df["Final Prediction"] = df.apply(strong_classifier, axis=1)

print("\nFINAL PREDICTIONS")
print(df[["y", "Final Prediction"]])
