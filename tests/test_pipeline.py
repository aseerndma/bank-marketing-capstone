import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def load_data():
    df = pd.read_csv("data/bank-full.csv", sep=";")
    return df


# --- DATA TESTS ---

def test_expected_columns_present():
    df = load_data()
    required = {"age", "job", "balance", "duration", "y"}
    assert required.issubset(df.columns), "A required column is missing"


def test_target_is_yes_no():
    df = load_data()
    assert set(df["y"].unique()) == {"yes", "no"}, "Target must be yes/no only"


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Dataset should have no null values"


def test_age_is_reasonable():
    df = load_data()
    assert df["age"].min() >= 18, "Ages below 18 are unexpected"
    assert df["age"].max() < 120, "Age is unrealistically high"


# --- MODEL TEST ---

def test_model_beats_f1_baseline():
    df = load_data().drop(columns=["duration"])   # same leakage fix
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42,
                                       class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))
    # A trivial model scores F1 ~0 on the rare class; require meaningfully better.
    assert f1 > 0.3, f"Model F1 {f1:.3f} is below acceptable threshold"