import argparse
import yaml
import csv
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Read config path from command line
parser = argparse.ArgumentParser(description="Train the bank marketing model.")
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
print(f"Loaded config: {config}")

# 2. Load and prepare data
df = pd.read_csv(config["data_path"], sep=";")
df = df.drop(columns=["duration"])          # leakage fix
y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y"])

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 3. Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"],
    random_state=config["random_state"], stratify=y
)

# 4. Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 5. Pick model from config
if config["model"] == "random_forest":
    clf = RandomForestClassifier(n_estimators=100,
                                 random_state=config["random_state"],
                                 class_weight="balanced")
elif config["model"] == "logistic_regression":
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
else:
    raise ValueError(f"Unknown model: {config['model']}")

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

# 6. Train
model.fit(X_train, y_train)
preds = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 3),
    "precision": round(precision_score(y_test, preds), 3),
    "recall": round(recall_score(y_test, preds), 3),
    "f1": round(f1_score(y_test, preds), 3),
}

# 7. Save model
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.joblib")

# 8. Write a report
with open("outputs/report.md", "w") as f:
    f.write("# Training Report\n\n")
    f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Model: **{config['model']}**\n\n")
    f.write("## Metrics\n\n")
    for k, v in metrics.items():
        f.write(f"- **{k}**: {v}\n")

# 9. Append to experiment log
log_path = "outputs/experiments.csv"
log_exists = os.path.exists(log_path)
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "model", "test_size",
                         "accuracy", "precision", "recall", "f1"])
    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"),
                     config["model"], config["test_size"],
                     metrics["accuracy"], metrics["precision"],
                     metrics["recall"], metrics["f1"]])

print("Metrics:", metrics)
print("Saved model, report, and experiment log to outputs/")