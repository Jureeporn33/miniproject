# train_model_text.py
import re
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ====== Thai word tokenization ======
# pip install pythainlp
from pythainlp.tokenize import word_tokenize


# =========================
# CONFIG (ปรับตามไฟล์คุณ)
# =========================
CSV_PATH = "D:\\mini_project\\dataset1.csv"          # path ไฟล์ dataset
TEXT_COL = "text"           # คอลัมน์ข้อความ
LABEL_COL = "sentiment"     # คอลัมน์ label
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_VERSION = "v1_tfidf_word_lr_newmm"
MODEL_OUT = f"model_{MODEL_VERSION}.joblib"
INFO_OUT = f"model_{MODEL_VERSION}_info.json"

TOKENIZE_ENGINE = "newmm"   # "newmm" แนะนำสุด | หรือ "deepcut" (ถ้าติดตั้งได้)


# =========================
# PREPROCESS (minimal, not over-clean)
# =========================
def basic_normalize(text: str) -> str:
    """Normalize whitespace only. Keep emojis/slang/punctuation."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # เปลี่ยน newline/tab เป็นช่องว่าง + ลดช่องว่างซ้อน
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def thai_tokenize_to_space_separated(text: str) -> str:
    """Tokenize Thai into space-separated tokens for TF-IDF word-level."""
    text = basic_normalize(text)
    if not text:
        return ""
    tokens = word_tokenize(text, engine=TOKENIZE_ENGINE, keep_whitespace=False)
    # กัน token ว่าง
    tokens = [t for t in tokens if t and not t.isspace()]
    return " ".join(tokens)


# =========================
# LOAD + BASIC CHECKS
# =========================
df = pd.read_csv(CSV_PATH)

required = {TEXT_COL, LABEL_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

# drop rows with missing text/label
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

# normalize + tokenize
df["text_tok"] = df[TEXT_COL].apply(thai_tokenize_to_space_separated)
df = df[df["text_tok"].str.len() > 0].copy()

# optional quick data diagnostics
print("Rows:", len(df))
print("\nClass distribution:\n", df[LABEL_COL].value_counts())
print("\nDuplicate text ratio:", df["text_tok"].duplicated().mean())


# =========================
# SPLIT
# =========================
X = df["text_tok"].astype(str).values
y = df[LABEL_COL].astype(str).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# MODEL (Baseline required)
# =========================
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2
    )),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None
    ))
])

# TRAIN
pipe.fit(X_train, y_train)

# PREDICT + EVAL
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
cm = confusion_matrix(y_test, pred, labels=sorted(pd.unique(y)))

print("\n===== EVALUATION =====")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Confusion Matrix (labels sorted):\n", cm)
print("\nClassification Report:\n", classification_report(y_test, pred))

# =========================
# ERROR EXAMPLES (อย่างน้อย 10 ตัวอย่าง ตามโจทย์)
# =========================
print("\n===== ERROR EXAMPLES =====")
errors = []
for xt, gt, pr in zip(X_test, y_test, pred):
    if gt != pr:
        errors.append((xt, gt, pr))

error_text = ""
if not errors:
    error_msg = "No errors found in test set."
    print(error_msg)
    error_text += error_msg + "\n"
else:
    for i, (xt, gt, pr) in enumerate(errors[:10], start=1):
        msg = f"\n#{i}\nTEXT: {xt[:300]}{'...' if len(xt) > 300 else ''}\nGT: {gt} | PRED: {pr}\n"
        print(msg.replace("\n", "\n"))
        error_text += msg

# =========================
# SAVE MODEL (.joblib) + INFO (สำหรับ /model/info ในเว็บ)
# =========================
payload = {
    "model": pipe,
    "version": MODEL_VERSION,
    "text_col": TEXT_COL,
    "label_col": LABEL_COL,
    "tokenize_engine": TOKENIZE_ENGINE
}
joblib.dump(payload, MODEL_OUT)

info = {
    "version": MODEL_VERSION,
    "vectorizer": {
        "type": "tfidf",
        "analyzer": "word",
        "ngram_range": [1, 2],
        "max_features": 20000,
        "min_df": 2,
        "thai_tokenize_engine": TOKENIZE_ENGINE
    },
    "classifier": {
        "type": "logistic_regression",
        "class_weight": "balanced",
        "max_iter": 1000
    },
    "metrics": {
        "accuracy": acc,
        "macro_f1": macro_f1
    },
    "data": {
        "csv_path": CSV_PATH,
        "rows_used": int(len(df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }
}
with open(INFO_OUT, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

# =========================
# SAVE TRAINING RESULTS TO FILE
# =========================
RESULTS_OUT = f"model_{MODEL_VERSION}_results.txt"
with open(RESULTS_OUT, "w", encoding="utf-8") as f:
    f.write("===== TRAINING RESULTS =====\n\n")
    f.write(f"Rows: {len(df)}\n\n")
    f.write("Class distribution:\n")
    f.write(df[LABEL_COL].value_counts().to_string())
    f.write(f"\n\nDuplicate text ratio: {df['text_tok'].duplicated().mean()}\n")
    
    f.write("\n===== EVALUATION =====\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Macro-F1: {macro_f1}\n")
    f.write(f"Confusion Matrix (labels sorted):\n{cm}\n")
    f.write(f"\nClassification Report:\n{classification_report(y_test, pred)}\n")
    
    f.write("\n===== ERROR EXAMPLES =====\n")
    f.write(error_text)

print(f"\nSaved model: {MODEL_OUT}")
print(f"Saved info : {INFO_OUT}")
print(f"Saved results: {RESULTS_OUT}")
