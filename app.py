import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import joblib
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ── NLTK downloads ──────────────────────────────────────────────────────────
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield · Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── font import ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── palette ──
   background  #0D1117
   surface     #161B22
   card        #1C2330
   border      #30363D
   accent-1    #58A6FF   (electric blue)
   accent-2    #3FB950   (terminal green)
   danger      #F85149   (alert red)
   muted       #8B949E
*/

html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }

/* main bg */
.stApp { background:#0D1117; color:#E6EDF3; }

/* sidebar */
[data-testid="stSidebar"] {
    background:#161B22 !important;
    border-right:1px solid #30363D;
}
[data-testid="stSidebar"] * { color:#E6EDF3 !important; }

/* cards */
.shield-card {
    background:#1C2330;
    border:1px solid #30363D;
    border-radius:12px;
    padding:24px 28px;
    margin-bottom:18px;
}
.shield-card h3 { margin:0 0 6px; font-size:1.05rem; color:#8B949E; font-weight:500; letter-spacing:.04em; }
.shield-card .value { font-size:2.4rem; font-weight:700; color:#E6EDF3; line-height:1.1; }
.shield-card .delta { font-size:.85rem; color:#3FB950; margin-top:4px; }

/* hero headline */
.hero { text-align:center; padding:36px 0 28px; }
.hero h1 { font-size:2.8rem; font-weight:700; letter-spacing:-.02em; margin:0; }
.hero h1 span.blue { color:#58A6FF; }
.hero h1 span.green { color:#3FB950; }
.hero p { color:#8B949E; font-size:1.05rem; margin-top:8px; }

/* verdict badge */
.verdict-spam {
    background:rgba(248,81,73,.15);
    border:1px solid #F85149;
    border-radius:10px;
    padding:20px 28px;
    text-align:center;
}
.verdict-ham {
    background:rgba(63,185,80,.12);
    border:1px solid #3FB950;
    border-radius:10px;
    padding:20px 28px;
    text-align:center;
}
.verdict-spam .label { font-size:2rem; font-weight:700; color:#F85149; }
.verdict-ham  .label { font-size:2rem; font-weight:700; color:#3FB950; }
.verdict-spam .conf, .verdict-ham .conf { font-size:1rem; color:#8B949E; margin-top:4px; }

/* token highlight */
.token-spam {
    display:inline-block;
    background:rgba(248,81,73,.18);
    color:#F85149;
    border-radius:6px;
    padding:2px 8px;
    margin:3px;
    font-family:'JetBrains Mono',monospace;
    font-size:.82rem;
}
.token-ham {
    display:inline-block;
    background:rgba(63,185,80,.15);
    color:#3FB950;
    border-radius:6px;
    padding:2px 8px;
    margin:3px;
    font-family:'JetBrains Mono',monospace;
    font-size:.82rem;
}

/* input area */
textarea { background:#161B22 !important; color:#E6EDF3 !important; border-color:#30363D !important; }
.stSelectbox > div > div { background:#161B22 !important; color:#E6EDF3 !important; }

/* metric overrides */
[data-testid="metric-container"] {
    background:#1C2330;
    border:1px solid #30363D;
    border-radius:10px;
    padding:16px !important;
}
[data-testid="metric-container"] label { color:#8B949E !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#E6EDF3 !important; font-weight:700; }

/* buttons */
.stButton > button {
    background:#58A6FF;
    color:#0D1117;
    font-weight:600;
    border:none;
    border-radius:8px;
    padding:10px 24px;
    font-family:'Space Grotesk',sans-serif;
    font-size:1rem;
    transition:opacity .15s;
}
.stButton > button:hover { opacity:.85; color:#0D1117; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { background:#161B22; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#8B949E; border-radius:8px; font-weight:500; }
.stTabs [aria-selected="true"] { background:#1C2330 !important; color:#E6EDF3 !important; }

/* progress bar */
.stProgress > div > div { background:#58A6FF; }

/* divider */
hr { border-color:#30363D; }

/* scrollbar */
::-webkit-scrollbar { width:6px; } 
::-webkit-scrollbar-track { background:#161B22; }
::-webkit-scrollbar-thumb { background:#30363D; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1C2330", "axes.facecolor": "#1C2330",
    "axes.edgecolor": "#30363D", "axes.labelcolor": "#8B949E",
    "xtick.color": "#8B949E", "ytick.color": "#8B949E",
    "text.color": "#E6EDF3", "grid.color": "#30363D",
    "grid.alpha": .6, "font.family": "DejaVu Sans",
    "legend.facecolor": "#161B22", "legend.edgecolor": "#30363D",
})

# ── Helpers ──────────────────────────────────────────────────────────────────
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\b\d[\d\s]*\b", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    stops = set(stopwords.words("english"))
    tokens = [stemmer.stem(t) for t in tokens if t not in stops and len(t) > 1]
    return " ".join(tokens)

@st.cache_data(show_spinner=False)
def load_data():
    """Generate a realistic synthetic dataset."""
    rng = np.random.default_rng(42)

    spam_templates = [
        "Congratulations! You've won a {prize}. Call {phone} now to claim!",
        "FREE entry in our weekly competition to win £{amount}. Text WIN to {code}",
        "URGENT: Your account has been compromised. Click {url} to verify.",
        "You have been selected for a cash prize of ${amount}. Reply YES to claim.",
        "Get {discount}% off all items. Limited time offer. Shop now at {url}",
        "Your package is on hold. Confirm delivery at {url} or call {phone}.",
        "Win a brand new iPhone! Just complete this survey at {url}",
        "Loan approved for ${amount}. No credit check. Apply at {url}",
        "Hot singles near you! Click {url} to meet them tonight.",
        "Earn ${amount}/day from home. No experience needed. Start at {url}",
        "Congratulations! You are our lucky winner. Send bank details to claim prize.",
        "FINAL NOTICE: Your car warranty is expiring. Call {phone} immediately.",
        "Text STOP to unsubscribe. Reply HELP for help. Std rates apply.",
        "Your subscription has been charged ${amount}. Cancel at {url}",
        "Act now! Exclusive deal expires at midnight. Buy at {url}",
    ]
    ham_templates = [
        "Hey, are you free for lunch tomorrow around noon?",
        "Can you send me the report before the meeting on Friday?",
        "Just got home, dinner is ready when you arrive.",
        "Don't forget we have a dentist appointment at 3pm.",
        "The project deadline has been moved to next Wednesday.",
        "Happy birthday! Hope you have a wonderful day.",
        "I'll be a bit late, traffic is terrible today.",
        "Can you pick up some milk on the way home?",
        "Meeting has been cancelled, we'll reschedule next week.",
        "Thanks for your help yesterday, really appreciated it.",
        "Call me when you get a chance, need to discuss something.",
        "Flight lands at 7pm, can you pick me up from the airport?",
        "The conference was really interesting, learned a lot.",
        "Reminder: your prescription is ready for pickup at the pharmacy.",
        "Hope you're feeling better today. Let me know if you need anything.",
        "Just finished the presentation, please review when you can.",
        "Game starts at 8, want to watch it together?",
        "Your Amazon order has shipped, expected delivery Thursday.",
        "Team lunch is at 12:30 in the main cafeteria, see you there.",
        "Do you want to grab coffee before the 10am standup?",
    ]

    def fill(template):
        return template.format(
            prize=rng.choice(["iPhone 15", "£500 cash", "holiday voucher", "PS5"]),
            phone=f"0800-{rng.integers(100,999)}-{rng.integers(1000,9999)}",
            amount=rng.integers(100, 10000),
            code=rng.integers(10000, 99999),
            url=f"bit.ly/{rng.integers(1000,9999)}",
            discount=rng.integers(30, 90),
        )

    n_spam, n_ham = 600, 1400
    texts, labels = [], []
    for _ in range(n_spam):
        texts.append(fill(rng.choice(spam_templates)))
        labels.append("spam")
    for _ in range(n_ham):
        texts.append(rng.choice(ham_templates))
        labels.append("ham")

    df = pd.DataFrame({"text": texts, "label": labels}).sample(frac=1, random_state=42).reset_index(drop=True)
    df["processed"] = df["text"].apply(preprocess)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["exclaim"] = df["text"].str.count("!")
    df["upper_ratio"] = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df["has_url"] = df["text"].str.lower().str.contains(r"http|www|bit\.ly", regex=True).astype(int)
    df["has_phone"] = df["text"].str.contains(r"\d{4,}", regex=True).astype(int)
    return df

@st.cache_resource(show_spinner=False)
def train_models(df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["processed"], df["label"], test_size=.2, random_state=42, stratify=df["label"]
    )
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
    }
    pipelines, reports = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        rep = classification_report(y_te, y_pred, output_dict=True)
        pipelines[name] = pipe
        reports[name] = {"acc": acc, "report": rep, "y_pred": y_pred, "y_test": y_te}
    return pipelines, reports, X_te, y_te

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SpamShield")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Detect Spam", "📋 Batch Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Active Models**")
    st.markdown("""
    <div class="shield-card" style="padding:12px 16px">
        <h3>CLASSIFIER</h3>
        <div style="font-size:1rem;font-weight:600;color:#58A6FF">Naive Bayes</div>
    </div>""", unsafe_allow_html=True)
    model_choice = "Naive Bayes"
    st.markdown("---")
    st.caption("Built with scikit-learn · NLTK · Streamlit")

# ── Load data & train ────────────────────────────────────────────────────────
with st.spinner("Initialising SpamShield…"):
    df = load_data()
    pipelines, reports, X_te, y_te = train_models(df)

# ════════════════════════════════════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>Spam<span class="blue">Shield</span> <span class="green">·</span> Detector</h1>
        <p>Machine-learning spam & ham classifier for SMS and email — powered by NLP</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    best_model = max(reports, key=lambda k: reports[k]["acc"])
    best_acc   = reports[best_model]["acc"]
    spam_prec  = reports[best_model]["report"]["spam"]["precision"]
    spam_rec   = reports[best_model]["report"]["spam"]["recall"]

    with c1:
        st.markdown(f"""
        <div class="shield-card">
            <h3>TOTAL MESSAGES</h3>
            <div class="value">{len(df):,}</div>
            <div class="delta">↑ Synthetic dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="shield-card">
            <h3>BEST ACCURACY</h3>
            <div class="value">{best_acc:.1%}</div>
            <div class="delta">↑ {best_model}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="shield-card">
            <h3>SPAM PRECISION</h3>
            <div class="value">{spam_prec:.1%}</div>
            <div class="delta">Low false positives</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="shield-card">
            <h3>SPAM RECALL</h3>
            <div class="value">{spam_rec:.1%}</div>
            <div class="delta">Few missed spams</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### How it works")
    cols = st.columns(4)
    steps = [
        ("01", "Input", "Paste any SMS or email text into the detector."),
        ("02", "Clean", "Tokenise, stem, and strip stopwords with NLTK."),
        ("03", "Vectorise", "TF-IDF transforms text into numeric features."),
        ("04", "Classify", "Your chosen model returns a spam/ham verdict."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="shield-card" style="text-align:center">
                <div style="font-size:2rem;font-weight:700;color:#58A6FF;font-family:'JetBrains Mono',monospace">{num}</div>
                <div style="font-weight:600;margin:6px 0 4px">{title}</div>
                <div style="font-size:.85rem;color:#8B949E">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("#### Dataset Distribution")
        spam_count = (df["label"] == "spam").sum()
        ham_count  = (df["label"] == "ham").sum()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["Ham ✉️", "Spam 🚨"], [ham_count, spam_count],
                      color=["#3FB950", "#F85149"], width=.45, zorder=3)
        ax.bar_label(bars, fmt="%d", padding=4, color="#E6EDF3", fontsize=11)
        ax.set_ylabel("Count"); ax.grid(axis="y"); ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_r:
        st.markdown("#### Quick Stats")
        for label, col in [("spam","#F85149"), ("ham","#3FB950")]:
            sub = df[df.label == label]
            st.markdown(f"""
            <div class="shield-card">
                <h3 style="color:{col}">{label.upper()}</h3>
                <div style="font-size:.88rem;color:#8B949E">
                    Avg words: <b style="color:#E6EDF3">{sub.word_count.mean():.0f}</b><br>
                    Avg chars: <b style="color:#E6EDF3">{sub.char_count.mean():.0f}</b><br>
                    Has URL: <b style="color:#E6EDF3">{sub.has_url.mean():.0%}</b><br>
                    Has phone: <b style="color:#E6EDF3">{sub.has_phone.mean():.0%}</b>
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  DETECT SPAM
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detect Spam":
    st.markdown("## 🔍 Detect Spam")
    st.markdown("Paste any SMS or email body below. The selected model will classify it in real time.")

    examples = {
        "— choose an example —": "",
        "🚨 Prize scam": "Congratulations! You've won an iPhone 15. Call 0800-555-1234 now to claim your prize!",
        "🚨 Phishing link": "URGENT: Your account has been compromised. Verify at bit.ly/5678 or lose access.",
        "🚨 Loan offer": "Loan approved for $5000. No credit check needed. Apply now at bit.ly/9999.",
        "✅ Friend message": "Hey, are you free for lunch tomorrow around noon? Let me know!",
        "✅ Work email": "Can you send me the updated report before the Friday meeting? Thanks!",
        "✅ Delivery update": "Your Amazon order has shipped and is expected to arrive Thursday.",
    }

    example_key = st.selectbox("Load an example", list(examples.keys()))
    user_text = st.text_area("Message text", value=examples[example_key],
                              height=160, placeholder="Type or paste your message here…",
                              label_visibility="collapsed")

    if st.button("🛡️  Analyse Message") and user_text.strip():
        pipe = pipelines[model_choice]
        processed = preprocess(user_text)
        prediction = pipe.predict([processed])[0]

        # Probability (fallback for SVM)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba([processed])[0]
            classes = pipe.classes_
            spam_idx = list(classes).index("spam")
            confidence = proba[spam_idx] if prediction == "spam" else 1 - proba[spam_idx]
        else:
            confidence = 0.95 if prediction == "spam" else 0.90

        st.markdown("---")
        col_v, col_info = st.columns([1, 2])
        with col_v:
            if prediction == "spam":
                st.markdown(f"""
                <div class="verdict-spam">
                    <div class="label">🚨 SPAM</div>
                    <div class="conf">Confidence: {confidence:.1%}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-ham">
                    <div class="label">✅ HAM</div>
                    <div class="conf">Confidence: {confidence:.1%}</div>
                </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown("**Text features detected**")
            feats = {
                "Word count": len(user_text.split()),
                "Char count": len(user_text),
                "Exclamation marks": user_text.count("!"),
                "UPPERCASE ratio": f"{sum(1 for c in user_text if c.isupper())/max(len(user_text),1):.1%}",
                "Contains URL": "Yes ⚠️" if re.search(r"http|www|bit\.ly", user_text, re.I) else "No",
                "Contains phone/number": "Yes ⚠️" if re.search(r"\d{4,}", user_text) else "No",
            }
            for k, v in feats.items():
                st.markdown(f"- **{k}:** {v}")

        st.markdown("---")
        st.markdown("**Processed tokens**")
        tokens = processed.split()
        if tokens:
            colour = "token-spam" if prediction == "spam" else "token-ham"
            token_html = " ".join(f'<span class="{colour}">{t}</span>' for t in tokens)
            st.markdown(token_html, unsafe_allow_html=True)
        else:
            st.info("No meaningful tokens after preprocessing.")

    elif user_text.strip() == "":
        st.info("Enter a message above and click **Analyse Message**.")

# ════════════════════════════════════════════════════════════════════════════
#  BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Analysis":
    st.markdown("## 📋 Batch Analysis")
    st.markdown("Upload a CSV with a **`text`** column to classify multiple messages at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            if "text" not in batch_df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                batch_df["processed"]   = batch_df["text"].apply(preprocess)
                pipe = pipelines[model_choice]
                batch_df["prediction"]  = pipe.predict(batch_df["processed"])
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba = pipe.predict_proba(batch_df["processed"])
                    classes = list(pipe.classes_)
                    spam_idx = classes.index("spam")
                    batch_df["spam_prob"] = proba[:, spam_idx]
                else:
                    batch_df["spam_prob"] = batch_df["prediction"].map({"spam": 0.95, "ham": 0.05})

                spam_n = (batch_df["prediction"] == "spam").sum()
                ham_n  = (batch_df["prediction"] == "ham").sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(batch_df))
                c2.metric("Spam detected", spam_n)
                c3.metric("Ham", ham_n)

                st.dataframe(
                    batch_df[["text","prediction","spam_prob"]].rename(
                        columns={"spam_prob":"Spam probability"}
                    ).style.format({"Spam probability": "{:.1%}"}),
                    use_container_width=True, height=320
                )

                csv_out = batch_df[["text","prediction","spam_prob"]].to_csv(index=False)
                st.download_button("⬇️ Download results CSV", csv_out, "spam_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
        <div class="shield-card">
            <h3>📂 No file uploaded</h3>
            <div style="color:#8B949E;font-size:.9rem">
                Your CSV should look like:<br><br>
                <code style="font-family:'JetBrains Mono',monospace;color:#58A6FF">
                text<br>
                "Congratulations you won a prize!"<br>
                "Hey, lunch tomorrow?"<br>
                "Click here to claim your reward"
                </code>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Or try with the built-in dataset")
        sample = df[["text","label"]].sample(20, random_state=99).rename(columns={"label":"true_label"})
        sample["processed"]  = sample["text"].apply(preprocess)
        pipe = pipelines[model_choice]
        sample["prediction"] = pipe.predict(sample["processed"])
        sample["correct"]    = sample["true_label"] == sample["prediction"]
        st.dataframe(
            sample[["text","true_label","prediction","correct"]],
            use_container_width=True, height=350, hide_index=True
        )
        right = sample["correct"].sum()
        st.info(f"✅ {right}/20 correct on this random sample using **{model_choice}**")