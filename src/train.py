import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("data1/processed/train.csv")
test = pd.read_csv("data1/processed/test.csv")

X_train, y_train = train.drop(columns=["label"]), train["label"]
X_test, y_test = test.drop(columns=["label"]), test["label"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

with open("metrics.json", "w") as f:
    json.dump({"accuracy": round(acc, 4)}, f)

print(f"accuracy: {acc:.4f}")