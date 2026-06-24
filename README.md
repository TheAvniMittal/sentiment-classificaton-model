# 🧠 TextEmotionAI

A machine learning web app that detects emotions from text using **Logistic Regression** and **Bag of Words**, trained on 16,000 labelled sentences.

---

## 🎯 Live Demo

🔗 [Click here to try the app](https://sentiment-classificaton-model-6ppvr27zxqyvsjt38culow.streamlit.app/)

---

## 📌 About the Project

**TextEmotionAI** is an end-to-end NLP project that classifies human emotions from raw text input. Type any sentence and the model will predict the emotion behind it with a confidence score and full probability breakdown.

**Detects 6 emotions:**

| Emotion | Emoji |
|---------|-------|
| Joy | 😄 |
| Sadness | 😢 |
| Anger | 😠 |
| Fear | 😨 |
| Love | ❤️ |
| Surprise | 😲 |

---

## 🛠️ Tech Stack

- **Language** — Python
- **NLP** — NLTK
- **ML Model** — Logistic Regression (Scikit-learn)
- **Vectorizer** — Bag of Words (CountVectorizer)
- **Web App** — Streamlit
- **Data Handling** — Pandas, NumPy

---

## ⚙️ How It Works

1. User inputs a sentence
2. Text is preprocessed — lowercased, punctuation removed, stopwords removed, tokenized
3. Transformed using Bag of Words vectorizer
4. Logistic Regression model predicts the emotion
5. Result displayed with confidence score and probability breakdown

---

## 📊 Model Performance

| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| Naive Bayes | BoW | 76.8% |
| Naive Bayes | TF-IDF | 66.1% |
| Logistic Regression | TF-IDF | 86.2% |
| **Logistic Regression** | **BoW** | **88.9%** ✅ |

---

## 📁 Project Structure

```
TextEmotionAI/
│
├── app.py               # Streamlit web app
├── model.pkl            # Trained Logistic Regression model
├── vectorizer.pkl       # Fitted BoW vectorizer
├── labels.pkl           # Emotion label mapping
├── train.txt            # Training dataset
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/TheAvniMittal/TextEmotionAI.git
cd TextEmotionAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Dataset

- 16,000 labelled sentences
- Format: `text;emotion`
- Source: Custom labelled dataset (`train.txt`)

---

## 🙋‍♀️ Author

**Avni Mittal**
- 🐙 GitHub: [@TheAvniMittal](https://github.com/TheAvniMittal)
- 💼 LinkedIn: [Avni Mittal](https://www.linkedin.com/in/avni-mittal-740b1a330/)

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
