"""Vehicle type classification using four scikit-learn classifiers."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Vehicle_Classification.csv"
TARGET = "Vehicle_Type"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    print("Dataset preview:\n", df.head())
    print("\nDataset information:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "k-NN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]),
    }

    rows = []
    labels = sorted(y.unique())
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        })
        print(f"\n===== {name} =====")
        print(classification_report(y_test, pred, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=labels)
        print("Confusion matrix labels:", labels, "\n", cm)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        filename = f"vehicle_confusion_{name.lower().replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(BASE_DIR / filename, dpi=150)
        plt.close()

    results = pd.DataFrame(rows).sort_values("F1_macro", ascending=False)
    print("\nModel comparison:\n", results.to_string(index=False))
    print("\nBest model by macro F1:", results.iloc[0]["Model"])
    results.to_csv(BASE_DIR / "vehicle_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
