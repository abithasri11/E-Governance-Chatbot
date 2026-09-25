import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MLClassifier:

    def __init__(self, data_path):
        self.data_path = data_path

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.model = LogisticRegression(max_iter=2000)

        self.tuning_results = None
        self.best_params = None
        self.evaluation_metrics = None

    def train(self):

        file = os.path.join(
            self.data_path,
            "e_sevai_classification_dataset.csv"
        )

        df = pd.read_csv(file)
        df = df.rename(columns={"Query": "query", "Intent": "intent"})
        df = df[["query", "intent"]].dropna()

        X_train, X_test, y_train, y_test = train_test_split(
            df["query"],
            df["intent"],
            test_size=0.20,
            random_state=42,
            stratify=df["intent"]
        )

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)

        results = []

        for C in [0.01, 0.1, 1, 10, 100]:
            for weight in [None, "balanced"]:

                model = LogisticRegression(
                    C=C,
                    class_weight=weight,
                    max_iter=2000
                )

                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                results.append({
                    "C": C,
                    "Class Weight": "None" if weight is None else "Balanced",
                    "Accuracy": accuracy_score(y_test, prediction),
                    "Precision": precision_score(
                        y_test, prediction,
                        average="weighted",
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, prediction,
                        average="weighted",
                        zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test, prediction,
                        average="weighted",
                        zero_division=0
                    )
                })

        self.tuning_results = pd.DataFrame(results)

        best = self.tuning_results.loc[
            self.tuning_results["F1 Score"].idxmax()
        ]

        best_weight = (
            None if best["Class Weight"] == "None"
            else "balanced"
        )

        self.best_params = {
            "C": best["C"],
            "Class Weight": best["Class Weight"]
        }

        self.model = LogisticRegression(
            C=best["C"],
            class_weight=best_weight,
            max_iter=2000
        )

        self.model.fit(X_train, y_train)

        final_prediction = self.model.predict(X_test)

        self.evaluation_metrics = {
            "Accuracy": accuracy_score(y_test, final_prediction),
            "Precision": precision_score(
                y_test, final_prediction,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, final_prediction,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, final_prediction,
                average="weighted",
                zero_division=0
            )
        }

    def predict(self, question):

        vector = self.vectorizer.transform([question])
        probabilities = self.model.predict_proba(vector)[0]

        index = np.argmax(probabilities)

        return (
            self.model.classes_[index],
            probabilities[index]
        )