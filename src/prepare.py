import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data1/wine.csv")
df["label"] = (df["quality"] >= 6).astype(int)
df = df.drop(columns=["quality"])

train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("data1/processed", exist_ok=True)
train.to_csv("data1/processed/train.csv", index=False)
test.to_csv("data1/processed/test.csv", index=False)

print(f"train: {len(train)} rows, test: {len(test)} rows")