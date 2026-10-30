"""
ReelFeel - Movie & Show Review Sentiment Analyzer
=================================================

A Streamlit web app built around the original Colab ML mini project:

    TF-IDF (English stop words, 10,000 features)  ->  Logistic Regression
    80/20 train/test split, random_state=42, classes: positive / negative / neutral

Everything lives in this single file:
    1. Data       - load reviews.csv (or generate the same synthetic reviews as the notebook)
    2. Model      - train once, save to model.pkl, reuse on every later visit
    3. Prediction - real model output (no dummy results)
    4. UI         - custom-styled Streamlit dashboard with charts

Run locally / in Colab:   streamlit run app.py
"""

import hashlib
import html as html_lib
import inspect
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
APP_NAME = "ReelFeel"
BASE_DIR = Path(__file__).resolve().parent          # works locally, in Colab and on Streamlit Cloud
DATA_PATH = BASE_DIR / "reviews.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_FORMAT = 1                                     # bump if the saved bundle layout ever changes

MAX_CHARS = 5000        # longest single review accepted
MAX_BATCH = 5000        # most reviews accepted in one batch run

INK = "#14213D"
GOLD = "#F2B33D"
SENTIMENT_COLORS = {"positive": "#1E9E6A", "negative": "#D63C52", "neutral": "#5B7FBF"}

EXAMPLES = [
    ("Positive example", "This movie was amazing and I really enjoyed it."),
    ("Negative example", "The film was boring, disappointing, and a waste of time."),
    ("Mixed example", "The acting was good but the story was average."),
]


def color_for(label):
    return SENTIMENT_COLORS.get(str(label).lower(), INK)


# ----------------------------------------------------------------------------
# 1. DATA  (same logic as the notebook)
# ----------------------------------------------------------------------------
def generate_synthetic_reviews(n=500, seed=42):
    """The notebook's synthetic review generator. A fixed seed makes it repeatable."""
    rng = random.Random(seed)

    positive_templates = [
        "This {} was absolutely amazing, highly recommend!",
        "Loved the {}, the acting and story were brilliant.",
        "A masterpiece of a {}, genuinely enjoyed every second.",
        "Great {}, fantastic visuals and amazing cast!",
        "One of the best {}s I have seen in a long time.",
    ]
    negative_templates = [
        "This {} was terrible, a complete waste of time.",
        "Awful {} with boring characters and a predictable plot.",
        "I hated the {}, the acting was extremely poor.",
        "Worst {} ever. It fell flat and was way too long.",
        "Extremely disappointing {}, nothing went right.",
    ]
    neutral_templates = [
        "The {} was okay, nothing special but not bad either.",
        "Decent {}, average acting and a standard storyline.",
        "It was a tolerable {}, fine to watch once.",
        "Some parts of the {} were good, others were slow.",
        "An average {} overall, okay for a lazy afternoon.",
    ]
    subjects = ["movie", "film", "show", "series", "story"]

    rows = []
    for _ in range(n):
        sentiment = rng.choice(["positive", "negative", "neutral"])
        subject = rng.choice(subjects)
        if sentiment == "positive":
            review = rng.choice(positive_templates).format(subject)
        elif sentiment == "negative":
            review = rng.choice(negative_templates).format(subject)
        else:
            review = rng.choice(neutral_templates).format(subject)
        rows.append({"review": review, "sentiment": sentiment})
    return pd.DataFrame(rows)


def prepare_dataframe(raw):
    """Notebook step 4: standardise column names, drop missing values, keep 2000 rows."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.lower().str.strip()

    missing = {"review", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(
            "reviews.csv must contain the columns 'review' and 'sentiment'. "
            f"Missing: {', '.join(sorted(missing))}. Found: {', '.join(df.columns)}."
        )

    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df.head(2000)                                # keep up to 2000 reviews for faster execution
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load reviews.csv; if it is missing, generate the notebook's synthetic reviews."""
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        source = "reviews.csv"
    else:
        raw = generate_synthetic_reviews()
        source = "generated"
        try:
            raw.to_csv(DATA_PATH, index=False)
        except OSError:
            pass                                       # read-only disk: fine, we still have the data
    return prepare_dataframe(raw), source


def dataset_signature(df):
    """Fingerprint of the training data, so a changed dataset triggers a retrain."""
    text = df[["review", "sentiment"]].to_csv(index=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# 2. MODEL  (same pipeline as the notebook: split -> TF-IDF -> Logistic Regression)
# ----------------------------------------------------------------------------
def train_bundle(df, signature):
    X = df["review"].astype(str)
    y = df["sentiment"]

    if y.nunique() < 2:
        raise ValueError("The dataset needs at least two different sentiment labels to train a model.")

    # Only stratify if all classes have at least 2 members
    class_counts = y.value_counts()
    stratify_option = y if (len(class_counts) > 1 and class_counts.min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": report["macro avg"]["f1-score"],
    }

    return {
        "format": MODEL_FORMAT,
        "sklearn_version": sklearn.__version__,
        "data_signature": signature,
        "tfidf": tfidf,
        "model": model,
        "labels": labels,
        "metrics": metrics,
        "report": report,
        "confusion": confusion_matrix(y_test, y_pred, labels=labels),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(tfidf.vocabulary_),
    }


@st.cache_resource(show_spinner=False)
def get_bundle():
    """Return (bundle, source). Loads model.pkl when valid, otherwise trains once and saves it."""
    df, _ = load_dataset()
    signature = dataset_signature(df)

    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            if (
                isinstance(saved, dict)
                and saved.get("format") == MODEL_FORMAT
                and saved.get("sklearn_version") == sklearn.__version__
                and saved.get("data_signature") == signature
            ):
                return saved, "loaded from model.pkl"
        except Exception:
            pass                                       # unreadable or incompatible file -> retrain below

    bundle = train_bundle(df, signature)
    try:
        joblib.dump(bundle, MODEL_PATH)
        return bundle, "trained now and saved to model.pkl"
    except OSError:
        return bundle, "trained now (model.pkl could not be saved)"


# ----------------------------------------------------------------------------
# 3. PREDICTION
# ----------------------------------------------------------------------------
def class_weights(model):
    """Per-class word weights, centred so they show what pushes toward each class."""
    coef = np.asarray(model.coef_)
    if coef.shape[0] == 1:                             # binary models store a single row
        coef = np.vstack([-coef, coef])
    return coef - coef.mean(axis=0, keepdims=True)


def predict_many(bundle, texts):
    """Real model output for a list of texts: labels, probabilities, known-word counts."""
    X = bundle["tfidf"].transform(texts)
    model = bundle["model"]
    labels = model.predict(X)
    probs = model.predict_proba(X)
    known = X.getnnz(axis=1)
    return labels, probs, known


def explain_text(bundle, text, label, k=8):
    """Words in the text that pushed the prediction, strongest first."""
    X = bundle["tfidf"].transform([text])
    if X.nnz == 0:
        return []
    classes = list(bundle["model"].classes_)
    ci = classes.index(label)
    W = class_weights(bundle["model"])
    names = np.asarray(bundle["tfidf"].get_feature_names_out())
    cols = X.indices
    contrib = X.data * W[ci, cols]
    order = np.argsort(-np.abs(contrib))[:k]
    return [(str(names[cols[i]]), float(contrib[i])) for i in order]


def analyse_one(bundle, text):
    labels, probs, known = predict_many(bundle, [text])
    label = str(labels[0])
    classes = [str(c) for c in bundle["model"].classes_]
    return {
        "text": text,
        "label": label,
        "classes": classes,
        "probs": [float(p) for p in probs[0]],
        "confidence": float(probs[0][classes.index(label)]) * 100,
        "known": int(known[0]),
        "words": explain_text(bundle, text, label),
    }


def validate_review(text):
    if not text.strip():
        return "Enter a review first. Type or paste at least one sentence."
    if len(text.strip()) < 3:
        return "That review is too short to analyse. Write at least a few words."
    if not re.search(r"[A-Za-z]", text):
        return "The model reads English words. Add some English text to this review."
    return None


def read_reviews_csv(file):
    """Return (list_of_reviews, error_message) for an uploaded CSV."""
    try:
        raw = pd.read_csv(file)
    except Exception:
        return None, "That file could not be read. Upload a valid CSV saved as UTF-8."
    if raw.empty:
        return None, "The file has no rows."
    columns = {str(c).lower().strip(): c for c in raw.columns}
    if "review" in columns:
        series = raw[columns["review"]]
    elif raw.shape[1] == 1:
        series = raw.iloc[:, 0]
    else:
        return None, "No “review” column found. Columns in your file: " + ", ".join(map(str, raw.columns))
    texts = [str(t).strip() for t in series.dropna() if str(t).strip()]
    if not texts:
        return None, "The review column is empty."
    return texts, None


# ----------------------------------------------------------------------------
# 4. UI helpers
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');

:root{
  --ink:#14213D; --soft:#4A5776; --paper:#EEF1F6; --card:#FFFFFF; --line:#D9DFEA;
  --gold:#F2B33D; --pos:#1E9E6A; --neg:#D63C52; --neu:#5B7FBF;
}
.stApp{background:var(--paper);}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp textarea, .stApp input{
  font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
}
.stApp p, .stApp label, .stApp li{color:var(--ink);}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1120px;padding-top:1.6rem;padding-bottom:3rem;}

/* ---------- hero: cinema marquee ---------- */
.stApp .hero{background:var(--ink);border-radius:14px;padding:1.5rem 2.2rem 2rem;margin-bottom:1.6rem;}
.hero .bulbs{height:12px;margin:0 0 1.3rem;
  background-image:radial-gradient(circle,#F2B33D 0 3.2px,transparent 3.8px);
  background-size:24px 12px;background-repeat:repeat-x;}
.hero .brand{font-family:'Fraunces',Georgia,serif;font-weight:600;color:var(--gold);font-size:1.05rem;letter-spacing:.02em;margin-bottom:.5rem;}
.stApp .hero h1{font-family:'Fraunces',Georgia,serif;font-weight:600;color:#fff;font-size:2.7rem;line-height:1.1;
  letter-spacing:-.01em;margin:0 0 .7rem;padding:0;}
.stApp .hero p{color:#C9D2E6;max-width:560px;font-size:1.02rem;line-height:1.55;margin:0 0 1.1rem;}
.hero .tag{display:inline-block;border:1px solid rgba(242,179,61,.55);color:#F6D694;border-radius:999px;
  padding:.22rem .8rem;font-size:.82rem;font-weight:500;margin:0 .5rem .4rem 0;}

/* ---------- section headings ---------- */
.stApp .sec{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:1.45rem;color:var(--ink);margin:.6rem 0 .1rem;}
.stApp .sec-sub{color:var(--soft);font-size:.95rem;margin:0 0 1rem;max-width:640px;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line);overflow-x:auto;}
.stTabs [data-baseweb="tab"]{height:auto;padding:.6rem .9rem;background:transparent;}
.stTabs [data-baseweb="tab"] p{font-weight:600;color:var(--soft);font-size:.98rem;}
.stTabs [aria-selected="true"] p{color:var(--ink);}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--gold);height:3px;}
.stTabs [data-baseweb="tab-border"]{background-color:transparent;}

/* ---------- inputs ---------- */
.stTextArea textarea{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:10px;font-size:1rem;line-height:1.5;}
.stTextArea textarea:focus{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink);}
.stButton>button, .stDownloadButton>button{border-radius:10px;font-weight:600;border:1px solid var(--line);
  background:#fff;color:var(--ink);padding:.5rem 1.1rem;}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--ink);color:var(--ink);}
.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:var(--ink);color:#fff;border:1px solid var(--ink);padding:.6rem 1.5rem;}
.stButton>button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p{color:#fff;}
.stButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{background:#243659;border-color:#243659;color:#fff;}

/* ---------- result ticket ---------- */
.ticket{display:grid;grid-template-columns:190px 1fr;background:#fff;border:1px solid var(--line);
  border-radius:14px;overflow:hidden;margin-bottom:1rem;}
.ticket .stub{background:var(--accent);color:#fff;padding:1.3rem 1.2rem;display:flex;flex-direction:column;justify-content:center;}
.ticket .stub-cap{font-size:.82rem;opacity:.9;margin-bottom:.3rem;}
.ticket .stub-word{font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:2rem;line-height:1.05;}
.ticket .body{padding:1.2rem 1.4rem;border-left:2px dashed var(--line);position:relative;}
.ticket .conf-head{display:flex;justify-content:space-between;align-items:baseline;font-size:.9rem;color:var(--soft);margin-bottom:.45rem;}
.ticket .conf-head b{font-family:'Fraunces',Georgia,serif;font-size:1.6rem;color:var(--ink);}
.ticket .track{height:8px;background:#E6EAF2;border-radius:99px;overflow:hidden;margin-bottom:1rem;}
.ticket .fill{height:100%;background:var(--accent);border-radius:99px;animation:fillbar .9s ease-out;}
@keyframes fillbar{from{width:0}}
.ticket .quote{font-size:.95rem;color:var(--soft);line-height:1.5;margin:0;font-style:italic;}
.empty{border:2px dashed var(--line);border-radius:14px;padding:2rem 1.5rem;color:var(--soft);background:rgba(255,255,255,.5);}
.empty b{font-family:'Fraunces',Georgia,serif;font-size:1.2rem;color:var(--ink);display:block;margin-bottom:.3rem;font-weight:600;}

/* ---------- word chips ---------- */
.chip{display:inline-block;border-radius:999px;padding:.3rem .8rem;margin:.25rem .4rem .25rem 0;font-size:.9rem;font-weight:600;}
.chip.sup{background:#E3F4EC;color:#14704B;}
.chip.opp{background:#FBE6E9;color:#A62B3E;}
.chip small{font-weight:500;opacity:.8;margin-left:.35rem;}
.legend{font-size:.85rem;color:var(--soft);margin-top:.4rem;}

/* ---------- KPI tiles ---------- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:.3rem 0 1.4rem;}
.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem;border-top:3px solid var(--kc,var(--ink));}
.kpi .v{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:2rem;color:var(--ink);line-height:1.1;}
.kpi .l{font-size:.88rem;color:var(--soft);margin-top:.2rem;}

/* ---------- about steps ---------- */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:.4rem 0 1.4rem;}
.step{background:#fff;border:1px solid var(--line);border-radius:8px;padding:1rem 1.1rem;}
.step .n{font-family:'Fraunces',Georgia,serif;font-size:1.5rem;color:var(--gold);font-weight:700;}
.step h4{margin:.1rem 0 .3rem;font-size:1.02rem;color:var(--ink);}
.step p{margin:0;font-size:.92rem;color:var(--soft);line-height:1.5;}
.note{background:#FFF7E6;border:1px solid #F1D9A2;border-radius:8px;padding:.8rem 1rem;color:#6B4E10;font-size:.93rem;margin:.4rem 0 1rem;line-height:1.5;}
.foot{text-align:left;color:var(--soft);font-size:.85rem;margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--line);}

/* ---------- mobile ---------- */
@media (max-width:700px){
  .block-container{padding-left:1rem;padding-right:1rem;}
  .stApp .hero{padding:1.2rem 1.2rem 1.5rem;}
  .stApp .hero h1{font-size:1.9rem;}
  .ticket{grid-template-columns:1fr;}
  .ticket .body{border-left:none;border-top:2px dashed var(--line);}
  .ticket .stub-word{font-size:1.7rem;}
}
</style>
"""


def esc(text):
    return html_lib.escape(str(text), quote=True)


def render_html(markup):
    """Send HTML to Streamlit: flatten to one line (stops markdown code blocks) and escape $."""
    flat = re.sub(r"\s*\n\s*", " ", markup.strip()).replace("$", "&#36;")
    st.markdown(flat, unsafe_allow_html=True)


def section(title, subtitle=""):
    sub = f'<p class="sec-sub">{esc(subtitle)}</p>' if subtitle else ""
    render_html(f'<h2 class="sec">{esc(title)}</h2>{sub}')


def kpi_tiles(items):
    """items: list of (value, label, accent_colour)."""
    tiles = "".join(
        f'<div class="kpi" style="--kc:{c}"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div></div>'
        for v, l, c in items
    )
    render_html(f'<div class="kpis">{tiles}</div>')


def full_width(func):
    """Kwargs that make a Streamlit element fill its column on any Streamlit version.
    Newer versions use width="stretch"; older ones use use_container_width=True."""
    if "width" in inspect.signature(func).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def show_chart(fig, key):
    st.plotly_chart(fig, key=key, **full_width(st.plotly_chart))


def show_table(data, **kwargs):
    st.dataframe(data, **full_width(st.dataframe), **kwargs)


def style_fig(fig, height=300, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=INK, size=13),
        showlegend=legend,
    )
    return fig


def prob_chart(classes, probs):
    order = np.argsort(probs)                          # ascending, so the biggest bar ends up on top
    fig = go.Figure(
        go.Bar(
            x=[probs[i] * 100 for i in order],
            y=[classes[i].capitalize() for i in order],
            orientation="h",
            marker_color=[color_for(classes[i]) for i in order],
            text=[f"{probs[i] * 100:.1f}%" for i in order],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 118], visible=False)
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=190)


def donut_chart(counts):
    labels = list(counts.keys())
    fig = go.Figure(
        go.Pie(
            labels=[l.capitalize() for l in labels],
            values=[counts[l] for l in labels],
            hole=0.6,
            marker=dict(colors=[color_for(l) for l in labels], line=dict(color="#fff", width=2)),
            textinfo="percent",
            sort=False,
        )
    )
    return style_fig(fig, height=280, legend=True)


def confusion_chart(cm, labels):
    names = [l.capitalize() for l in labels]
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=names, y=names, showscale=False,
            colorscale=[[0, "#F3F5FA"], [1, "#2E4A7D"]],
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
            xgap=3, ygap=3,
        )
    )
    top = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=names[j], y=names[i], text=str(int(cm[i, j])), showarrow=False,
                font=dict(size=20, color="#fff" if cm[i, j] > top / 2 else INK),
            )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return style_fig(fig, height=340)


# ----------------------------------------------------------------------------
# 5. UI sections
# ----------------------------------------------------------------------------
def hero(bundle, df):
    acc = bundle["metrics"]["accuracy"] * 100
    render_html(
        f"""
        <div class="hero">
          <div class="bulbs"></div>
          <div class="brand">{esc(APP_NAME)}</div>
          <h1>How does this review feel?</h1>
          <p>Paste a movie or show review. A text classifier trained on {len(df):,} labelled reviews
          reads it and tells you whether it sounds positive, negative or neutral.</p>
          <span class="tag">TF-IDF text features</span>
          <span class="tag">Logistic Regression</span>
          <span class="tag">{acc:.1f}% test accuracy</span>
        </div>
        """
    )


def set_example(text):
    st.session_state["review_text"] = text


def tab_analyze(bundle):
    st.session_state.setdefault("review_text", "")
    section("Analyse a review", "Write your own review, or start from an example and press Analyse review.")

    left, right = st.columns([5, 4], gap="large")

    with left:
        example_cols = st.columns(len(EXAMPLES))
        for i, (col, (label, text)) in enumerate(zip(example_cols, EXAMPLES)):
            col.button(label, key=f"example_{i}", on_click=set_example, args=(text,))

        st.text_area(
            "Review text",
            key="review_text",
            height=170,
            max_chars=MAX_CHARS,
            placeholder="For example: The story dragged a little, but the cast was brilliant.",
            label_visibility="collapsed",
        )
        clicked = st.button("Analyse review", type="primary", key="analyse_button")

    if clicked:
        text = st.session_state.get("review_text", "")
        problem = validate_review(text)
        if problem:
            st.session_state["result"] = None
            st.session_state["problem"] = problem
        else:
            try:
                st.session_state["result"] = analyse_one(bundle, text.strip())
                st.session_state["problem"] = None
            except Exception as exc:                   # never show a raw traceback to visitors
                st.session_state["result"] = None
                st.session_state["problem"] = f"Something went wrong while analysing this review: {exc}"

    result = st.session_state.get("result")
    problem = st.session_state.get("problem")

    with right:
        if problem:
            st.warning(problem)
        elif result is None:
            render_html(
                """
                <div class="empty"><b>Your result appears here</b>
                Enter a review on the left and press Analyse review. You will see the predicted
                sentiment, how confident the model is, and the words that drove the decision.</div>
                """
            )
        else:
            short = " ".join(result["text"].split())
            if len(short) > 220:
                short = short[:217] + "..."
            render_html(
                f"""
                <div class="ticket" style="--accent:{color_for(result['label'])}">
                  <div class="stub">
                    <div class="stub-cap">Predicted sentiment</div>
                    <div class="stub-word">{esc(result['label'].capitalize())}</div>
                  </div>
                  <div class="body">
                    <div class="conf-head"><span>Model confidence</span><b>{result['confidence']:.1f}%</b></div>
                    <div class="track"><div class="fill" style="width:{result['confidence']:.1f}%"></div></div>
                    <p class="quote">“{esc(short)}”</p>
                  </div>
                </div>
                """
            )
            if result["known"] == 0:
                st.warning(
                    "None of the words in this review appear in the model’s vocabulary, so this "
                    "result is not reliable. The model only knows words from its training reviews."
                )
            elif result["confidence"] < 50:
                st.info("The model is unsure here. The probabilities are close, so treat this result with care.")
            show_chart(prob_chart(result["classes"], result["probs"]), "prob_chart")

    if result is not None and not problem and result["words"]:
        section(
            "Why the model said this",
            "These words from your review moved the prediction the most. Common words such as “the” and “was” are ignored.",
        )
        chips = "".join(
            f'<span class="chip {"sup" if w >= 0 else "opp"}">{esc(word)}<small>{w:+.2f}</small></span>'
            for word, w in result["words"]
        )
        label = esc(result["label"])
        render_html(
            f"""{chips}
            <div class="legend">Green words support “{label}”. Red words pull toward a different sentiment.</div>"""
        )


def tab_batch(bundle):
    section(
        "Analyse many reviews at once",
        "Paste one review per line, or upload a CSV file that has a column named “review”.",
    )
    mode = st.radio(
        "Input method", ["Paste reviews", "Upload CSV"],
        horizontal=True, label_visibility="collapsed", key="batch_mode",
    )

    texts, error = None, None
    if mode == "Paste reviews":
        st.text_area(
            "Reviews", key="batch_text", height=190,
            placeholder="Great movie, loved every minute\nThe show was boring and far too long",
            label_visibility="collapsed",
        )
        if st.button("Analyse all", type="primary", key="batch_paste_button"):
            texts = [ln.strip() for ln in st.session_state.get("batch_text", "").splitlines() if ln.strip()]
            if not texts:
                texts, error = None, "Add at least one review, one per line."
    else:
        upload = st.file_uploader("CSV file", type=["csv"], key="batch_file", label_visibility="collapsed")
        if st.button("Analyse all", type="primary", key="batch_csv_button"):
            if upload is None:
                error = "Choose a CSV file first."
            else:
                texts, error = read_reviews_csv(upload)

    if error:
        st.warning(error)

    if texts:
        note = None
        if len(texts) > MAX_BATCH:
            note = f"Your file has {len(texts):,} reviews. Only the first {MAX_BATCH:,} were analysed."
            texts = texts[:MAX_BATCH]
        try:
            labels, probs, known = predict_many(bundle, texts)
            classes = [str(c) for c in bundle["model"].classes_]
            conf = probs.max(axis=1) * 100
            st.session_state["batch_df"] = pd.DataFrame(
                {
                    "Review": texts,
                    "Sentiment": [str(l).capitalize() for l in labels],
                    "Confidence": np.round(conf, 1),
                    "Known words": known,
                }
            )
            st.session_state["batch_note"] = note
        except Exception as exc:
            st.session_state["batch_df"] = None
            st.error(f"Something went wrong while analysing these reviews: {exc}")

    out = st.session_state.get("batch_df")
    if out is None or out.empty:
        return

    if st.session_state.get("batch_note"):
        st.info(st.session_state["batch_note"])

    counts = out["Sentiment"].str.lower().value_counts()
    tiles = [(f"{len(out):,}", "Reviews analysed", INK)]
    for cls in bundle["labels"]:
        tiles.append((f"{int(counts.get(cls, 0)):,}", f"{cls.capitalize()} reviews", color_for(cls)))
    kpi_tiles(tiles)

    chart_col, table_col = st.columns([2, 3], gap="large")
    with chart_col:
        present = {c: int(counts[c]) for c in counts.index}
        show_chart(donut_chart(present), "batch_donut")
    with table_col:
        show_table(
            out,
            hide_index=True,
            height=300,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )
        if (out["Known words"] == 0).any():
            st.caption("Rows with 0 known words contain no vocabulary the model recognises, so treat them as unreliable.")
        st.download_button(
            "Download results as CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="reelfeel_results.csv",
            mime="text/csv",
            key="batch_download",
        )


def tab_performance(bundle):
    m = bundle["metrics"]
    section("How well does the model perform?", "Scores come from the 20% of reviews the model never saw while training.")
    kpi_tiles(
        [
            (f"{m['accuracy'] * 100:.1f}%", "Accuracy", INK),
            (f"{m['precision'] * 100:.1f}%", "Precision (macro average)", INK),
            (f"{m['recall'] * 100:.1f}%", "Recall (macro average)", INK),
            (f"{m['f1'] * 100:.1f}%", "F1 score (macro average)", INK),
            (f"{bundle['train_size']:,} / {bundle['test_size']:,}", "Training / test reviews", GOLD),
        ]
    )

    if m["accuracy"] >= 0.98:
        render_html(
            """<div class="note"><b>Why is the score so high?</b> Near-perfect scores usually mean the data is simple
            or repetitive. This demo dataset is built from a small set of sentence templates, so the model learns the
            templates rather than language in general. Expect lower scores on real-world reviews.</div>"""
        )

    left, right = st.columns(2, gap="large")
    with left:
        section("Confusion matrix", "Rows are the true sentiment, columns are what the model predicted.")
        show_chart(confusion_chart(bundle["confusion"], bundle["labels"]), "confusion_chart")
    with right:
        section("Per-class report")
        rep = pd.DataFrame(bundle["report"]).T
        rep = rep[rep.index != "accuracy"].copy()
        rep.index = [str(i).capitalize() for i in rep.index]
        rep.columns = ["Precision", "Recall", "F1 score", "Support"]
        rep[["Precision", "Recall", "F1 score"]] = rep[["Precision", "Recall", "F1 score"]].round(3)
        rep["Support"] = rep["Support"].astype(int)
        show_table(rep, height=250)
        st.caption("Support is the number of test reviews in each class.")

    section("Words the model links to each sentiment", "The strongest word signals the model learned from the training reviews.")
    names = np.asarray(bundle["tfidf"].get_feature_names_out())
    weights = class_weights(bundle["model"])
    classes = [str(c) for c in bundle["model"].classes_]
    for start in range(0, len(classes), 3):
        cols = st.columns(3, gap="large")
        for col, idx in zip(cols, range(start, min(start + 3, len(classes)))):
            order = np.argsort(weights[idx])[::-1][:8][::-1]
            fig = go.Figure(
                go.Bar(
                    x=weights[idx][order], y=names[order], orientation="h",
                    marker_color=color_for(classes[idx]),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                )
            )
            fig.update_layout(title=dict(text=classes[idx].capitalize(), font=dict(size=15)))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(showgrid=False)
            with col:
                show_chart(style_fig(fig, height=300), f"words_{classes[idx]}")


def tab_dataset(df, bundle):
    section("The training data", "Every review the model learned from, with its sentiment label.")
    lengths = df["review"].astype(str).str.split().str.len()
    kpi_tiles(
        [
            (f"{len(df):,}", "Reviews in the dataset", INK),
            (f"{df['sentiment'].nunique()}", "Sentiment classes", INK),
            (f"{lengths.mean():.1f}", "Average words per review", INK),
            (f"{bundle['n_features']:,}", "Words in the model vocabulary", GOLD),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        counts = df["sentiment"].value_counts().reindex(bundle["labels"]).fillna(0)
        fig = go.Figure(
            go.Bar(
                x=[c.capitalize() for c in counts.index], y=counts.values,
                marker_color=[color_for(c) for c in counts.index],
                text=[int(v) for v in counts.values], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        fig.update_layout(title=dict(text="Reviews per sentiment", font=dict(size=15)))
        fig.update_yaxes(showgrid=True, gridcolor="#E3E8F1", rangemode="tozero")
        show_chart(style_fig(fig, height=300), "class_balance")
    with right:
        fig = go.Figure(go.Histogram(x=lengths, marker_color=INK, nbinsx=15, hovertemplate="%{x} words: %{y}<extra></extra>"))
        fig.update_layout(title=dict(text="Review length (words)", font=dict(size=15)), bargap=0.08)
        fig.update_yaxes(showgrid=True, gridcolor="#E3E8F1")
        show_chart(style_fig(fig, height=300), "length_hist")

    section("Browse the reviews")
    f1, f2 = st.columns([1, 2])
    with f1:
        choice = st.selectbox("Sentiment", ["All"] + [c.capitalize() for c in bundle["labels"]], key="data_filter")
    with f2:
        query = st.text_input("Search", key="data_search", placeholder="Search for a word, for example “acting”")
    view = df
    if choice != "All":
        view = view[view["sentiment"] == choice.lower()]
    if query.strip():
        view = view[view["review"].astype(str).str.contains(query.strip(), case=False, regex=False)]
    shown = view.head(300).rename(columns={"review": "Review", "sentiment": "Sentiment"})
    show_table(shown, hide_index=True, height=320)
    st.caption(f"Showing {len(shown):,} of {len(view):,} matching reviews.")


def tab_about(bundle, df, data_source, model_source):
    section("How it works", "From a raw sentence to a sentiment label in four steps.")
    render_html(
        """
        <div class="steps">
          <div class="step"><div class="n">1</div><h4>Clean the data</h4>
            <p>Column names are standardised, rows with missing values are dropped, and up to 2,000 reviews are kept.</p></div>
          <div class="step"><div class="n">2</div><h4>Turn words into numbers</h4>
            <p>TF-IDF scores every word by how useful it is for telling reviews apart. English filler words are removed and the top 10,000 words are kept.</p></div>
          <div class="step"><div class="n">3</div><h4>Learn from examples</h4>
            <p>Logistic Regression trains on 80% of the reviews and learns which words point to which sentiment.</p></div>
          <div class="step"><div class="n">4</div><h4>Predict</h4>
            <p>Your review goes through the same TF-IDF step. The model returns a probability for each sentiment and picks the highest.</p></div>
        </div>
        """
    )

    render_html(
        """<div class="note"><b>Know the limits.</b> This model was trained on a small set of template-based reviews.
        It recognises the words in those reviews, so unusual wording, slang, sarcasm, other languages and long
        real-world reviews can produce weak or wrong results. Treat it as a learning project, not a production tool.</div>"""
    )

    section("Technical details")
    st.markdown(
        f"""
- **Data source:** {"`reviews.csv` in the project folder" if data_source == "reviews.csv" else "generated with the notebook’s review generator"} ({len(df):,} reviews)
- **Model status:** {model_source}
- **Split:** 80% training, 20% testing, `random_state=42`, stratified by sentiment
- **Vectoriser:** `TfidfVectorizer(stop_words="english", max_features=10000)`
- **Classifier:** `LogisticRegression(max_iter=1000)`
- **scikit-learn version:** {sklearn.__version__}
"""
    )
    with st.expander("How do I retrain the model?"):
        st.markdown(
            "Replace `reviews.csv` with your own data (columns `review` and `sentiment`), or delete `model.pkl`. "
            "On the next start the app trains a fresh model and saves it again. "
            "The app also retrains by itself when `reviews.csv` changes or the scikit-learn version differs."
        )


# ----------------------------------------------------------------------------
# 6. App entry point
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=f"{APP_NAME} | Review sentiment analyzer", page_icon="🎬", layout="wide")
    render_html(CSS)

    try:
        with st.spinner("Preparing the model. The first start can take a few seconds..."):
            df, data_source = load_dataset()
            bundle, model_source = get_bundle()
    except Exception as exc:
        st.error(f"The app could not start: {exc}")
        st.stop()
        return

    hero(bundle, df)

    tabs = st.tabs(["Analyse", "Batch", "Performance", "Dataset", "About"])
    with tabs[0]:
        tab_analyze(bundle)
    with tabs[1]:
        tab_batch(bundle)
    with tabs[2]:
        tab_performance(bundle)
    with tabs[3]:
        tab_dataset(df, bundle)
    with tabs[4]:
        tab_about(bundle, df, data_source, model_source)

    render_html(f'<div class="foot">{esc(APP_NAME)} is a machine learning mini project built with scikit-learn and Streamlit.</div>')


if __name__ == "__main__":
    main()
