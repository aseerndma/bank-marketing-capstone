import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load data (semicolon-separated)
df = pd.read_csv("data/bank-full.csv", sep=";")

# 2. Drop 'duration' — it's data leakage (only known AFTER the call ends)
df = df.drop(columns=["duration"])

# 3. Turn the target into 0/1
y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y"])

# 4. Identify column types automatically
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 5. Split first (prevents leakage), stratify to keep class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Preprocessing: scale numbers, one-hot encode categories
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 7. Two models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                             class_weight="balanced"),
}

# 8. Train and evaluate each
for name, clf in models.items():
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_test, preds), 3))
    print("Precision:", round(precision_score(y_test, preds), 3))
    print("Recall:   ", round(recall_score(y_test, preds), 3))
    print("F1 score: ", round(f1_score(y_test, preds), 3))