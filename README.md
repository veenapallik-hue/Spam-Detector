# 🛡️ SpamShield · Spam Detector

A full-featured **SMS & Email Spam Detection** app built with Streamlit, scikit-learn, NLTK, and more.

## ✨ Features

| Page | Description |
|------|-------------|
| 🏠 Home | Dataset overview, key metrics, and how-it-works walkthrough |
| 🔍 Detect Spam | Real-time single-message classification with token highlights |
| 📊 Model Analytics | Accuracy comparison, confusion matrix, classification report |
| 📈 Data Insights | Feature distributions, correlation heatmap, top token analysis |
| 📋 Batch Analysis | Upload a CSV and classify hundreds of messages at once |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## 🤖 Models Included

- **Naive Bayes** — fast, strong baseline for text

## 🔧 Tech Stack

| Library | Role |
|---------|------|
| `streamlit` | Web UI |
| `scikit-learn` | ML models + TF-IDF vectorisation |
| `nltk` | Tokenisation, stopwords, stemming |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisations |
| `joblib` | Model serialisation |

## 📁 Project Structure

```
spam_detector/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 💡 Usage Tips

- Switch classifiers from the **sidebar** at any time
- Use the **example loader** in Detect Spam to try pre-written messages
- For batch mode, upload a CSV with a single `text` column
- Download classified results as CSV from Batch Analysis
