import pandas as pd
import json
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

train = pd.read_csv("data1/processed/train.csv")
test = pd.read_csv("data1/processed/test.csv")

X_train, y_train = train.drop(columns=["label"]), train["label"]
X_test, y_test = test.drop(columns=["label"]), test["label"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)