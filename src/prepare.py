import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepare"]

df = pd.read_csv("data1/wine.csv")
df["label"] = (df["quality"] >= params["quality_threshold"]).astype(int)
df = df.drop(columns=["quality"])

train, test = train_test_split(df, test_size=params["test_size"], random_state=42)

os.makedirs("data1/processed", exist_ok=True)
train.to_csv("data1/processed/train.csv", index=False)
test.to_csv("data1/processed/test.csv", index=False)

print(f"train: {len(train)} rows, test: {len(test)} rows")