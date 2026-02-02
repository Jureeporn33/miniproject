# TF-IDF + Logistic Regression
# with Confidence Threshold & Error Analysis
 
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
 
 
#  Config
INPUT_CSV = "D:\\mini_project\\dataset.csv"
TEXT_COL = "text"
LABEL_COL = "sentiment"
MODEL_PATH = "sentiment_model_v1.joblib"
 
RANDOM_STATE = 42
CONF_THRESHOLD = 0.6
 
 
def main():
    #  Load Data
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
 
    #  Encode Label
    print("Encoding labels...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[LABEL_COL])
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
 
    X = df[TEXT_COL]
    y = df["label"]
 
    #  Train / Test Split
    print("Splitting train / test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
 
    #  Model Pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=3000,
            min_df=5,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])

    #  Train
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
 
    #   Predict with Confidence
    probs = model.predict_proba(X_test)
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
 
    # low confidence → Neutral
    neutral_idx = le.transform(["Neutral"])[0]
    preds_adj = np.where(max_probs < CONF_THRESHOLD, neutral_idx, preds)
 
    #  Evaluation
    print("Evaluation (with confidence threshold)")
    print(confusion_matrix(y_test, preds_adj))
 
    print("Classification Report:")
    print(classification_report(
        y_test,
        preds_adj,
        target_names=le.classes_
    ))
 
    macro_f1 = f1_score(y_test, preds_adj, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")
 
    #  Error Analysis
    error_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(preds_adj),
        "confidence": max_probs
    })
 
    misclassified = error_df[
        error_df["true_label"] != error_df["pred_label"]
    ]
 
    print(f"ตัวอย่างที่โมเดลทำนายผิด : {len(misclassified)}")
 
    print("\n🔍 Sample misclassified examples (up to 10):")
    for _, row in misclassified.head(10).iterrows():
        print("-----")
        print("Text:", row["text"])
        print("True:", row["true_label"])
        print("Pred:", row["pred_label"])
        print("Conf:", round(row["confidence"], 3))
 
    #  Error Count by Class
    print("Error count by class:")
    error_summary = (
        misclassified
        .groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(error_summary)
 
    #  Save Model
    print("Saving model...")
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "confidence_threshold": CONF_THRESHOLD,
            "model_name": "Logistic Regression (TF-IDF)"
        },
        MODEL_PATH
    )
 
    print(f"Model saved to {MODEL_PATH}")
 
 
if __name__ == "__main__":
    main()
 