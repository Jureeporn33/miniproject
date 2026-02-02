# predict_sentiment.py
import joblib
import pandas as pd
from pythainlp.tokenize import word_tokenize
import re

# =========================
# CONFIG
# =========================
MODEL_FILE = "model_v1_tfidf_word_lr_newmm.joblib"
TOKENIZE_ENGINE = "newmm"

# Score mapping: กำหนดช่วงคะแนนตามประเภท sentiment
SCORE_MAP = {
    "Positive": 0.85,   # ดี (0.7-1.0)
    "Neutral": 0.50,    # กลาง (0.4-0.6)
    "Negative": 0.15    # ต่ำ (0-0.3)
}

# =========================
# LOAD MODEL
# =========================
try:
    payload = joblib.load(MODEL_FILE)
    model = payload["model"]
    print(f"✓ โมเดลโหลดสำเร็จ: {payload['version']}\n")
except Exception as e:
    print(f"❌ เรียนโมเดลล้มเหลว: {e}")
    exit(1)


# =========================
# PREPROCESS (เหมือน training)
# =========================
def basic_normalize(text: str) -> str:
    """Normalize whitespace only."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def thai_tokenize_to_space_separated(text: str) -> str:
    """Tokenize Thai into space-separated tokens."""
    text = basic_normalize(text)
    if not text:
        return ""
    tokens = word_tokenize(text, engine=TOKENIZE_ENGINE, keep_whitespace=False)
    tokens = [t for t in tokens if t and not t.isspace()]
    return " ".join(tokens)


# =========================
# PREDICT WITH CONFIDENCE + SCORE
# =========================
def predict_sentiment(user_text: str):
    """
    Predict sentiment and return score based on sentiment type:
    - Positive → Score 0.85 (ดี)
    - Neutral → Score 0.50 (กลาง)
    - Negative → Score 0.15 (ต่ำ)
    """
    # Preprocess
    tokenized = thai_tokenize_to_space_separated(user_text)
    
    if not tokenized:
        return {
            "error": "ข้อความว่างเปล่าหรือไม่มีคำที่รู้จัก"
        }
    
    # Predict
    prediction = model.predict([tokenized])[0]
    probabilities = model.predict_proba([tokenized])[0]
    classes = model.classes_
    
    # Get max probability
    max_prob = max(probabilities)
    
    # Get score based on sentiment type
    score = SCORE_MAP.get(prediction, 0.5)
    
    # Adjust score slightly based on confidence
    confidence_adjustment = (max_prob - 0.5) * 0.2  # ±0.1 adjustment
    adjusted_score = min(1.0, max(0.0, score + confidence_adjustment))
    
    return {
        "original_text": user_text,
        "tokenized_text": tokenized,
        "sentiment": prediction,
        "confidence": round(max_prob * 100, 2),
        "score": round(adjusted_score, 2),
        "score_range": get_score_range(prediction),
        "probabilities": {
            class_name: round(prob * 100, 2) 
            for class_name, prob in zip(classes, probabilities)
        }
    }


def get_score_range(sentiment: str) -> str:
    """Return score range description based on sentiment."""
    if sentiment == "Positive":
        return "0.70 - 1.00 (ดี 👍)"
    elif sentiment == "Negative":
        return "0.00 - 0.30 (ต่ำ 👎)"
    else:  # Neutral
        return "0.40 - 0.60 (กลาง 😐)"


def print_result(result: dict):
    """Pretty print the result."""
    if "error" in result:
        print(f"❌ {result['error']}")
        return
    
    print("=" * 60)
    print(f"📝 ข้อความ: {result['original_text']}")
    print(f"🔍 Token: {result['tokenized_text']}")
    print("=" * 60)
    print(f"😊 Sentiment: {result['sentiment']}")
    print(f"📊 Score: {result['score']} {result['score_range']}")
    print(f"🎯 Confidence: {result['confidence']}%")
    print("\nรายละเอียดความน่าจะเป็น:")
    for class_name, prob in result['probabilities'].items():
        bar = "█" * int(prob / 5)
        print(f"  {class_name:10} : {prob:6.2f}% {bar}")
    print("=" * 60 + "\n")


# =========================
# INTERACTIVE PREDICTION
# =========================
if __name__ == "__main__":
    print("🎯 Sentiment Classifier - Interactive Mode")
    print("พิมพ์ 'exit' หรือ 'quit' เพื่อออก\n")
    
    while True:
        try:
            user_input = input("📥 กรอกข้อความที่ต้องการวิเคราะห์: ").strip()
            
            if user_input.lower() in ["exit", "quit", "ออก"]:
                print("👋 ขอบคุณที่ใช้งาน!")
                break
            
            if not user_input:
                print("⚠️  กรุณากรอกข้อความ\n")
                continue
            
            result = predict_sentiment(user_input)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n👋 ขอบคุณที่ใช้งาน!")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}\n")
