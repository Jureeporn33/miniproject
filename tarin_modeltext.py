import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import re

# =========================
# CONFIG
# =========================
CSV_PATH = "D:\\mini_project\\dataset1.csv"          # <-- เปลี่ยนเป็น path ของคุณ
TEXT_COL = "text"           # <-- ชื่อคอลัมน์ข้อความ
LABEL_COL = "sentiment"     # <-- ชื่อคอลัมน์ label (เช่น sentiment)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUT = "sentiment_tfidf_lr_v1.joblib"

# =========================
# BASIC PREPROCESS (ไม่ over-clean)
# =========================
def basic_normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # ลดช่องว่างซ้อน
    return text

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

# กันแถวที่ text/label หาย
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

df[TEXT_COL] = df[TEXT_COL].apply(basic_normalize)

X = df[TEXT_COL].astype(str).values
y = df[LABEL_COL].astype(str).values

# =========================
# SPLIT (สำคัญ: stratify)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================
# MODEL (Baseline)
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
        max_iter=1000
    ))
])

# =========================
# TRAIN
# =========================
pipe.fit(X_train, y_train)

# =========================
# EVAL
# =========================
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
cm = confusion_matrix(y_test, pred)

print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, pred))

# =========================
# SAVE .joblib
# =========================
joblib.dump({
    "model": pipe,
    "text_col": TEXT_COL,
    "label_col": LABEL_COL,
    "version": "v1_tfidf_word_lr"
}, MODEL_OUT)

print(f"\nSaved model to: {MODEL_OUT}")
