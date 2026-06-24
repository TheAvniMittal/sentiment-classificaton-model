import streamlit as st
import pickle
import string
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return model, vectorizer, labels

model, vectorizer, labels = load_artifacts()
stop_words = set(stopwords.words("english"))

# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    words = word_tokenize(txt)
    return " ".join(w for w in words if w not in stop_words)

# ── Emotion config ───────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "sadness":  {"emoji": "😢", "color": "#5b8dee"},
    "anger":    {"emoji": "😠", "color": "#e05c5c"},
    "love":     {"emoji": "❤️", "color": "#e07ab1"},
    "surprise": {"emoji": "😲", "color": "#f0a500"},
    "fear":     {"emoji": "😨", "color": "#8c5fe6"},
    "joy":      {"emoji": "😄", "color": "#3cba6e"},
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Detector", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .result-box {
        padding: 24px 28px;
        border-radius: 16px;
        margin-top: 20px;
        text-align: center;
    }
    .emotion-label {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .confidence-text {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🧠 Emotion Detector")
st.caption("Logistic Regression + Bag of Words · 88.9% accuracy · 6 emotions")

st.markdown("---")

text_input = st.text_area(
    "Enter a sentence to analyse:",
    placeholder="e.g. I feel so happy and grateful today!",
    height=120,
)

predict_btn = st.button("Detect Emotion", use_container_width=True, type="primary")

if predict_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = preprocess(text_input)
        vec = vectorizer.transform([cleaned])
        pred_idx = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        emotion = labels[pred_idx]
        cfg = EMOTION_CONFIG.get(emotion, {"emoji": "🤔", "color": "#aaa"})
        confidence = probs[pred_idx] * 100

        # Main result card
        st.markdown(f"""
        <div class="result-box" style="background:{cfg['color']}22; border: 2px solid {cfg['color']};">
            <div style="font-size:3rem">{cfg['emoji']}</div>
            <div class="emotion-label" style="color:{cfg['color']};">{emotion.upper()}</div>
            <div class="confidence-text">Confidence: <b>{confidence:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("#### Probability breakdown")
        sorted_probs = sorted(
            [(labels[i], p) for i, p in enumerate(probs)],
            key=lambda x: x[1], reverse=True
        )
        for emo, prob in sorted_probs:
            ecfg = EMOTION_CONFIG.get(emo, {"emoji": "•", "color": "#aaa"})
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(float(prob), text=f"{ecfg['emoji']} {emo.capitalize()}")
            with col2:
                st.markdown(
                    f"<div style='padding-top:8px;text-align:right'><b>{prob*100:.1f}%</b></div>",
                    unsafe_allow_html=True
                )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Logistic Regression · Vectoriser: Bag of Words · Classes: sadness · anger · love · surprise · fear · joy")
