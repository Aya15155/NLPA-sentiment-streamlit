# Streamlit Sentiment Analysis Web App
# Uses the saved CNN model, tokenizer.pkl, and config.pkl from the notebook.
# The app reuses the saved tokenizer. It does NOT fit a new tokenizer.

from pathlib import Path
import html
import pickle
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Page setup + comfort styling
# -----------------------------
st.set_page_config(
    page_title="CNN Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
)

PALETTE = {
    "cream": "#FFF8F0",
    "soft_peach": "#FFF1E8",
    "peach": "#FFB7A8",
    "coral": "#FF765F",
    "mint": "#EAF7F6",
    "soft_mint": "#F3FBFA",
    "dark": "#21304F",
    "muted": "#5B6B84",
    "lavender": "#D8C6F3",
}

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        color: {PALETTE['dark']} !important;
    }}

    .stApp {{
        background: linear-gradient(135deg, {PALETTE['cream']} 0%, #FFF3EA 48%, #F3FBFA 100%) !important;
        color: {PALETTE['dark']} !important;
    }}

    .main .block-container {{
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}

    header[data-testid="stHeader"] {{
        background: rgba(255, 248, 240, 0.96) !important;
        border-bottom: 1px solid rgba(51, 65, 85, 0.08) !important;
    }}

    [data-testid="stToolbar"],
    [data-testid="stToolbar"] *,
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] * {{
        color: {PALETTE['dark']} !important;
        fill: {PALETTE['dark']} !important;
    }}

    [data-testid="stDecoration"] {{
        background: linear-gradient(90deg, {PALETTE['mint']}, {PALETTE['peach']}) !important;
    }}

    .main-title {{
    padding: 1.2rem 1.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFF1E8 55%, #EAF7F6 100%);
    box-shadow: 0 10px 30px rgba(51, 65, 85, 0.08);
    margin-bottom: 1.2rem;
    }}

    .main-title h1 {{
        margin: 0;
        color: {PALETTE['dark']} !important;
        font-size: 2.2rem;
    }}

    .main-title p {{
        margin: .35rem 0 0 0;
        color: {PALETTE['muted']} !important;
        font-size: 1rem;
    }}

    .soft-card {{
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(51, 65, 85, 0.08);
        border-radius: 22px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 10px 24px rgba(51, 65, 85, 0.06);
        margin-bottom: 1rem;
        color: {PALETTE['dark']} !important;
    }}

    .metric-card {{
        background: rgba(255, 255, 255, 0.96);
        border-radius: 20px;
        border: 1px solid rgba(51, 65, 85, 0.08);
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(51, 65, 85, 0.06);
        min-height: 110px;
        color: {PALETTE['dark']} !important;
    }}

    .metric-label {{
        font-size: .85rem;
        color: {PALETTE['muted']} !important;
        margin-bottom: .35rem;
    }}

    .metric-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {PALETTE['dark']} !important;
    }}

    .metric-note {{
        font-size: .78rem;
        color: {PALETTE['muted']} !important;
        margin-top: .25rem;
    }}

    .positive-pill {{
        display:inline-block;
        background: {PALETTE['mint']};
        color: #245B5C !important;
        border-radius: 999px;
        padding: .45rem .9rem;
        font-weight: 800;
    }}

    .negative-pill {{
        display:inline-block;
        background: #FFE1DC;
        color: #8C3D35 !important;
        border-radius: 999px;
        padding: .45rem .9rem;
        font-weight: 800;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,.96) !important;
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(51, 65, 85, 0.08);
        box-shadow: 0 8px 20px rgba(51, 65, 85, 0.05);
        color: {PALETTE['dark']} !important;
    }}

    div[data-testid="stMetric"] * {{
        color: {PALETTE['dark']} !important;
    }}

    .stTabs [aria-selected="true"] {{
    background: #FFF1E8 !important;
    border-bottom: 3px solid #FF765F !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        background: rgba(255,255,255,.92) !important;
        padding: .65rem 1.05rem;
        border: 1px solid rgba(51, 65, 85, 0.08);
    }}

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {{
        color: {PALETTE['dark']} !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: #FFFFFF !important;
    }}

    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {{
        color: #1F2937 !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {PALETTE['mint']} !important;
        border-bottom: 3px solid #FF765F !important;
    }}

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {{
        color: {PALETTE['dark']} !important;
        font-weight: 800 !important;
    }}

    label,
    .stMarkdown,
    .stText,
    p,
    span,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {{
        color: {PALETTE['dark']} !important;
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: {PALETTE['dark']} !important;
    }}

    textarea,
    input,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {{
        color: {PALETTE['dark']} !important;
        background-color: rgba(255,255,255,.98) !important;
    }}

    textarea::placeholder,
    input::placeholder {{
        color: #7A8A9B !important;
        opacity: 1 !important;
    }}

    button {{
        color: {PALETTE['dark']} !important;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #FFB7A8 0%, #FFD8CF 100%) !important;
        color: #21304F !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #FFA493 0%, #FFCABE 100%) !important;
        color: #21304F !important;
    }}
    /* Altair line charts only. Bar charts use custom responsive HTML below. */
    div[data-testid="stVegaLiteChart"] {{
        background: #FFFFFF !important;
        border: 1px solid #EFE7DD !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 22px rgba(54, 44, 35, 0.06) !important;
        padding: 16px 18px 18px 18px !important;
        margin: 0 auto 1.25rem auto !important;
        width: calc(100% - 36px) !important;
        max-width: 1280px !important;
        box-sizing: border-box !important;
        overflow-x: auto !important;
        color: {PALETTE['dark']} !important;
    }}

    div[data-testid="stVegaLiteChart"] > div {{
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }}

    .html-chart-box {{
        width: calc(100% - 36px);
        max-width: 1280px;
        margin: 0 auto 1.25rem auto;
        background: #FFFFFF;
        border: 1px solid #EFE7DD;
        border-radius: 20px;
        box-shadow: 0 8px 22px rgba(54, 44, 35, 0.06);
        padding: 24px 28px 28px 28px;
        box-sizing: border-box;
        color: {PALETTE['dark']} !important;
    }}

    .html-chart-title {{
        color: {PALETTE['dark']} !important;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 1.35rem;
    }}

    .html-chart-row {{
        display: grid;
        grid-template-columns: minmax(145px, 230px) minmax(0, 1fr) minmax(70px, 90px);
        gap: 14px;
        align-items: center;
        margin: 1rem 0;
    }}

    .html-chart-label {{
        color: {PALETTE['dark']} !important;
        font-size: .88rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }}

    .html-chart-track {{
        height: 34px;
        background: #F4F6F8;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }}

    .html-chart-fill {{
        height: 100%;
        border-radius: 999px;
        min-width: 3px;
    }}

    .html-chart-value {{
        color: {PALETTE['dark']} !important;
        font-size: .85rem;
        font-weight: 700;
        text-align: right;
        white-space: nowrap;
    }}

    .html-chart-axis {{
        display: grid;
        grid-template-columns: minmax(145px, 230px) minmax(0, 1fr) minmax(70px, 90px);
        gap: 14px;
        align-items: center;
        margin-top: .8rem;
        color: {PALETTE['muted']} !important;
        font-size: .78rem;
    }}

    .html-chart-axis-line {{
        display: flex;
        justify-content: space-between;
        border-top: 1px solid #E5E7EB;
        padding-top: .45rem;
    }}

    /* Force normal Streamlit tables to light theme */
    div[data-testid="stTable"],
    div[data-testid="stDataFrame"] {{
        background: #FFFFFF !important;
        color: {PALETTE['dark']} !important;
        border-radius: 16px !important;
    }}

    div[data-testid="stTable"] table {{
        background: #FFFFFF !important;
        color: {PALETTE['dark']} !important;
        border-collapse: collapse !important;
    }}

    div[data-testid="stTable"] thead tr th {{
        background: #EAF7F7 !important;
        color: {PALETTE['dark']} !important;
        font-weight: 800 !important;
        border-bottom: 1px solid #D9E6E6 !important;
    }}

    div[data-testid="stTable"] tbody tr td {{
        background: #FFFFFF !important;
        color: {PALETTE['dark']} !important;
        border-bottom: 1px solid #EFE7DD !important;
    }}

    div[data-testid="stTable"] tbody tr:nth-child(even) td {{
        background: #FFF8F0 !important;
    }}

    .stDataFrame,
    .stDataFrame *,
    div[data-testid="stDataFrame"] *,
    div[data-testid="stTable"] * {{
        color: {PALETTE['dark']} !important;
    }}

    @media (max-width: 780px) {{
        .main-title h1 {{
            font-size: 1.65rem;
        }}

        .html-chart-box {{
            width: calc(100% - 8px);
            padding: 18px 16px 20px 16px;
        }}

        .html-chart-row {{
            grid-template-columns: 1fr;
            gap: 8px;
            margin: 1.1rem 0;
        }}

        .html-chart-value {{
            text-align: left;
        }}

        .html-chart-axis {{
            display: none;
        }}
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-title">
      <h1>💬 Sentiment Analysis Web App</h1>
      <p>CNN sentiment classifier using the saved tokenizer and model from the NLP lab notebook.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "models"

# The app loads the model files automatically from a local "models" folder.
# Expected structure:
# app.py
# models/
#   original_CNN_model.keras
#   tokenizer.pkl
#   config.pkl
model_dir = DEFAULT_MODEL_DIR


# -----------------------------
# Notebook results embedded from the final notebook
# -----------------------------
NOTEBOOK_RESULTS = {
    "workflow": [
        {"Step": "1", "Stage": "Install dependencies", "Result": "Keras, Gensim, SpaCy installed; en_core_web_sm downloaded in Colab"},
        {"Step": "2", "Stage": "Load original dataset", "Result": "IMDB labelled TSV loaded with Text and Label columns"},
        {"Step": "3", "Stage": "Preprocessing", "Result": "SpaCy tokenization, punctuation/stopword removal, Text_Final created"},
        {"Step": "4", "Stage": "Train/test split", "Result": "668 training samples and 75 test samples; MAX_SEQUENCE_LENGTH = 50"},
        {"Step": "5–8", "Stage": "Embedding + tokenization", "Result": "GloVe 100d selected; Keras tokenizer fitted only on training data"},
        {"Step": "9–10", "Stage": "Original CNN", "Result": "CNN trained with early stopping; best model evaluated on original test set"},
        {"Step": "Bonus", "Stage": "ELMo CNN", "Result": "ELMo sentence vectors used to train a second CNN model"},
        {"Step": "Final", "Stage": "External dataset test", "Result": "Amazon product reviews tested to check domain shift and performance change"},
    ],
    "dataset_summary": [
        {"Item": "Original dataset", "Value": "IMDB movie reviews"},
        {"Item": "Original train samples", "Value": "668"},
        {"Item": "Original test samples", "Value": "75"},
        {"Item": "External dataset", "Value": "Amazon product reviews"},
        {"Item": "External samples", "Value": "1000"},
        {"Item": "External class balance", "Value": "500 positive / 500 negative"},
        {"Item": "Sequence length", "Value": "50"},
        {"Item": "Embedding", "Value": "GloVe 100d for original CNN"},
        {"Item": "ELMo vector size", "Value": "1024"},
    ],
    "model_scores": [
        {"Model / Dataset": "Original CNN on IMDB test", "Accuracy": 0.7200, "Loss": 0.571852},
        {"Model / Dataset": "ELMo CNN on IMDB test", "Accuracy": 0.8133, "Loss": 0.4175},
        {"Model / Dataset": "Original CNN on Amazon reviews", "Accuracy": 0.6480, "Loss": 0.642919},
    ],
    "original_training": [
        {"Epoch": 1, "Training Accuracy": 0.5491, "Validation Accuracy": 0.6119, "Training Loss": 0.6860, "Validation Loss": 0.6676},
        {"Epoch": 2, "Training Accuracy": 0.7105, "Validation Accuracy": 0.7313, "Training Loss": 0.5982, "Validation Loss": 0.5503},
        {"Epoch": 3, "Training Accuracy": 0.7854, "Validation Accuracy": 0.7015, "Training Loss": 0.4470, "Validation Loss": 0.5628},
        {"Epoch": 4, "Training Accuracy": 0.8735, "Validation Accuracy": 0.6716, "Training Loss": 0.3118, "Validation Loss": 0.5556},
        {"Epoch": 5, "Training Accuracy": 0.9185, "Validation Accuracy": 0.6716, "Training Loss": 0.2119, "Validation Loss": 0.6520},
        {"Epoch": 6, "Training Accuracy": 0.9551, "Validation Accuracy": 0.6418, "Training Loss": 0.1394, "Validation Loss": 0.7366},
        {"Epoch": 7, "Training Accuracy": 0.9601, "Validation Accuracy": 0.6567, "Training Loss": 0.1094, "Validation Loss": 0.8986},
    ],
    "elmo_training": [
        {"Epoch": 1, "Training Accuracy": 0.6938, "Validation Accuracy": 0.7612, "Training Loss": 0.6104, "Validation Loss": 0.5049},
        {"Epoch": 2, "Training Accuracy": 0.8170, "Validation Accuracy": 0.7761, "Training Loss": 0.3979, "Validation Loss": 0.5020},
        {"Epoch": 3, "Training Accuracy": 0.8702, "Validation Accuracy": 0.7313, "Training Loss": 0.3123, "Validation Loss": 0.6299},
        {"Epoch": 4, "Training Accuracy": 0.9285, "Validation Accuracy": 0.8060, "Training Loss": 0.1905, "Validation Loss": 0.6786},
        {"Epoch": 5, "Training Accuracy": 0.9534, "Validation Accuracy": 0.8060, "Training Loss": 0.1252, "Validation Loss": 0.7784},
        {"Epoch": 6, "Training Accuracy": 0.9834, "Validation Accuracy": 0.7910, "Training Loss": 0.0650, "Validation Loss": 0.9983},
        {"Epoch": 7, "Training Accuracy": 0.9884, "Validation Accuracy": 0.8060, "Training Loss": 0.0558, "Validation Loss": 1.0675},
    ],
    "external_report": [
        {"Class": "Positive", "Precision": 0.75, "Recall": 0.45, "F1-score": 0.56, "Support": 500},
        {"Class": "Negative", "Precision": 0.61, "Recall": 0.85, "F1-score": 0.71, "Support": 500},
        {"Class": "Macro avg", "Precision": 0.68, "Recall": 0.65, "F1-score": 0.63, "Support": 1000},
        {"Class": "Weighted avg", "Precision": 0.68, "Recall": 0.65, "F1-score": 0.63, "Support": 1000},
    ],
    "external_oov_rate": 0.4998,
}


# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource(show_spinner="Loading optional SpaCy model...")
def load_spacy_model():
    """Load the same SpaCy model used in the notebook if available."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading CNN model and tokenizer...")
def load_artifacts(model_folder: str):
    model_folder = Path(model_folder)

    model_path = model_folder / "original_CNN_model.keras"
    tokenizer_path = model_folder / "tokenizer.pkl"
    config_path = model_folder / "config.pkl"

    missing = [str(p) for p in [model_path, tokenizer_path, config_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    # compile=False is enough for prediction and avoids optimizer-version issues.
    model = tf.keras.models.load_model(model_path, compile=False)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    max_sequence_length = int(config.get("MAX_SEQUENCE_LENGTH", 50))
    return model, tokenizer, max_sequence_length


nlp = load_spacy_model()

try:
    model, tokenizer, MAX_SEQUENCE_LENGTH = load_artifacts(str(model_dir))
except Exception as e:
    st.error("Could not load the model files. Make sure the models folder is beside app.py and contains original_CNN_model.keras, tokenizer.pkl, and config.pkl.")
    st.exception(e)
    st.stop()


# -----------------------------
# Preprocessing and prediction
# -----------------------------
def preprocess_text(text: str) -> str:
    """
    Same idea as the notebook's spacy_tokenize:
    - lowercase
    - remove punctuation
    - remove stopwords if SpaCy model is available
    """
    text = "" if pd.isna(text) else str(text)

    if nlp is not None:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
    else:
        # Fallback only if en_core_web_sm is not installed.
        tokens = re.findall(r"[a-zA-Z']+", text.lower())

    return " ".join(tokens)


def vectorize_texts(texts):
    processed = [preprocess_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return processed, sequences, padded


def calculate_oov_rate(processed_texts):
    tokens = []
    for text in processed_texts:
        tokens.extend(str(text).split())

    if not tokens:
        return 0.0

    known = sum(1 for tok in tokens if tok in tokenizer.word_index)
    unknown = len(tokens) - known
    return unknown / len(tokens)


def predict_sentiment(texts):
    processed, sequences, padded = vectorize_texts(texts)
    probs = model.predict(padded, verbose=0)

    results = []
    for raw_text, clean_text, prob in zip(texts, processed, probs):
        prob = np.array(prob).flatten()

        # Notebook model uses 2 outputs: [Positive, Negative]
        if len(prob) >= 2:
            positive_conf = float(prob[0])
            negative_conf = float(prob[1])
        else:
            # Fallback for a sigmoid model, if ever used.
            positive_conf = float(prob[0])
            negative_conf = 1.0 - positive_conf

        if positive_conf >= negative_conf:
            sentiment = "Positive"
            confidence = positive_conf
        else:
            sentiment = "Negative"
            confidence = negative_conf

        results.append(
            {
                "Text": raw_text,
                "Text_Final": clean_text,
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Positive_Confidence": positive_conf,
                "Negative_Confidence": negative_conf,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df, calculate_oov_rate(processed)


# -----------------------------
# Visualization helpers
# -----------------------------
def chart_title(text):
    return alt.TitleParams(
        text=text or "",
        anchor="start",
        fontSize=18,
        fontWeight="bold",
        color=PALETTE["dark"],
        offset=16,
    )


def chart_padding():
    # Vega/Altair padding keeps the title, labels, bars, and axes inside the white chart card.
    return {"top": 24, "left": 18, "right": 58, "bottom": 28}


def numeric_domain(values, percent=False):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vals.empty:
        return [0, 1]
    max_val = float(vals.max())
    if percent:
        # Accuracy, F1-score, precision, recall are stored as 0–1 values.
        # Keeping a 0–1 domain prevents cropped bars and makes percentages consistent.
        return [0, 1]
    if max_val <= 0:
        return [0, 1]
    return [0, max_val * 1.15]


def soft_bar_chart(df, x, y, title=None, y_format=None, color_field=None, height=330):
    # Render a responsive centered HTML bar chart using components.html.
    # This avoids Streamlit displaying HTML as code and prevents chart overflow/cropping.
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("No chart data available.")
        return

    plot_df = df[[x, y]].dropna().copy()
    if plot_df.empty:
        st.info("No chart data available.")
        return

    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])
    if plot_df.empty:
        st.info("No numeric data available for this chart.")
        return

    is_percent = y_format == "%"
    values = plot_df[y].astype(float)

    if is_percent:
        domain_max = 1.0
        axis_max_label = "100%"
    else:
        max_value = float(values.max())
        domain_max = max_value * 1.18 if max_value > 0 else 1.0
        axis_max_label = f"{domain_max:.2f}".rstrip("0").rstrip(".")

    colors = [PALETTE["teal"], PALETTE["lavender"], PALETTE["rose"], PALETTE["sage"], PALETTE["soft_blue"], PALETTE["peach"]]

    def fmt_value(v):
        if is_percent:
            return f"{v * 100:.2f}%"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.4f}".rstrip("0").rstrip(".")

    plot_df = plot_df.sort_values(y, ascending=False).reset_index(drop=True)

    rows_html = []
    for i, row in plot_df.iterrows():
        label = html.escape(str(row[x]))
        val = float(row[y])
        width_pct = 0 if domain_max <= 0 else max(0, min(100, (val / domain_max) * 100))
        color = colors[i % len(colors)]
        rows_html.append(
            f'''
            <div class="chart-row">
                <div class="chart-label" title="{label}">{label}</div>
                <div class="chart-track">
                    <div class="chart-fill" style="width:{width_pct:.3f}%; background:{color};"></div>
                </div>
                <div class="chart-value">{fmt_value(val)}</div>
            </div>
            '''
        )

    chart_height = max(275, 135 + len(plot_df) * 72)

    chart_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: {PALETTE['dark']};
            overflow: hidden;
        }}
        * {{ box-sizing: border-box; }}
        .chart-box {{
            width: min(100%, 1120px);
            margin: 0 auto;
            background: #FFFFFF;
            border: 1px solid #EFE7DD;
            border-radius: 18px;
            padding: 24px 28px 26px 28px;
            box-shadow: 0 8px 22px rgba(54, 44, 35, 0.06);
        }}
        .chart-title {{
            font-weight: 800;
            font-size: 17px;
            color: {PALETTE['dark']};
            margin: 0 0 20px 0;
            line-height: 1.35;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: minmax(150px, 220px) minmax(0, 1fr) minmax(70px, 90px);
            gap: 16px;
            align-items: center;
            margin: 18px 0;
        }}
        .chart-label {{
            font-size: 13px;
            color: {PALETTE['dark']};
            line-height: 1.25;
            white-space: normal;
            overflow-wrap: anywhere;
            text-align: right;
        }}
        .chart-track {{
            width: 100%;
            height: 32px;
            border-radius: 10px;
            background: #F5F7FB;
            border: 1px solid #E8EDF5;
            overflow: hidden;
        }}
        .chart-fill {{
            height: 100%;
            border-radius: 9px;
            min-width: 2px;
        }}
        .chart-value {{
            font-size: 13px;
            font-weight: 700;
            color: {PALETTE['dark']};
            text-align: left;
            white-space: nowrap;
        }}
        .chart-axis {{
            display: grid;
            grid-template-columns: minmax(150px, 220px) minmax(0, 1fr) minmax(70px, 90px);
            gap: 16px;
            align-items: center;
            margin-top: 12px;
        }}
        .chart-axis-line {{
            border-top: 1px solid #D8DEE9;
            display: flex;
            justify-content: space-between;
            padding-top: 8px;
            color: {PALETTE['muted']};
            font-size: 12px;
        }}
        @media (max-width: 760px) {{
            .chart-box {{ padding: 18px 16px 20px 16px; }}
            .chart-row {{
                grid-template-columns: 1fr;
                gap: 8px;
                margin: 20px 0;
            }}
            .chart-label {{ text-align: left; }}
            .chart-value {{ text-align: right; }}
            .chart-axis {{ display: none; }}
        }}
    </style>
    </head>
    <body>
        <div class="chart-box">
            <div class="chart-title">{html.escape(title or y)}</div>
            {''.join(rows_html)}
            <div class="chart-axis">
                <div></div>
                <div class="chart-axis-line"><span>0</span><span>{html.escape(axis_max_label)}</span></div>
                <div></div>
            </div>
        </div>
    </body>
    </html>
    '''

    components.html(chart_html, height=chart_height, scrolling=False)

def soft_line_chart(df, y_columns, title=None, height=340):
    long_df = df.melt("Epoch", value_vars=y_columns, var_name="Metric", value_name="Value")
    is_accuracy = any("Accuracy" in col for col in y_columns)
    chart = (
        alt.Chart(long_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "Epoch:O",
                title="Epoch",
                axis=alt.Axis(labelAngle=0, labelColor=PALETTE["dark"], titleColor=PALETTE["dark"], labelPadding=8),
            ),
            y=alt.Y(
                "Value:Q",
                title="Value",
                scale=alt.Scale(domain=[0, 1], nice=False) if is_accuracy else alt.Scale(zero=False, nice=True),
                axis=alt.Axis(
                    labelColor=PALETTE["dark"],
                    titleColor=PALETTE["dark"],
                    gridColor="#E5E7EB",
                    domainColor="#CBD5E1",
                    tickColor="#CBD5E1",
                    labelPadding=8,
                ),
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=[PALETTE["teal"], PALETTE["rose"], PALETTE["lavender"], PALETTE["sage"]]),
                legend=alt.Legend(labelColor=PALETTE["dark"], titleColor=PALETTE["dark"], orient="bottom"),
            ),
            tooltip=["Epoch", "Metric", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(
            width="container",
            height=height,
            title=chart_title(title),
            background="#FFFFFF",
            padding=chart_padding(),
            autosize={"type": "fit", "contains": "padding", "resize": True},
        )
        .configure_axis(labelColor=PALETTE["dark"], titleColor=PALETTE["dark"], gridColor="#E5E7EB", domainColor="#CBD5E1", tickColor="#CBD5E1")
        .configure_view(stroke=None, fill="#FFFFFF")
    )
    st.altair_chart(chart, use_container_width=True)

def show_metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sentiment_summary(results_df: pd.DataFrame):
    counts = results_df["Sentiment"].value_counts().reindex(["Positive", "Negative"]).fillna(0).astype(int)
    col1, col2, col3 = st.columns(3)
    with col1:
        show_metric_card("Positive", int(counts.get("Positive", 0)), "Predicted positive rows")
    with col2:
        show_metric_card("Negative", int(counts.get("Negative", 0)), "Predicted negative rows")
    with col3:
        show_metric_card("Average confidence", f"{results_df['Confidence'].mean() * 100:.2f}%", "Mean prediction confidence")

    chart_df = counts.reset_index()
    chart_df.columns = ["Sentiment", "Count"]
    soft_bar_chart(chart_df, "Sentiment", "Count", title="Sentiment distribution", color_field="Sentiment", height=300)


def load_optional_notebook_comparison(model_folder: Path) -> pd.DataFrame:
    """If the user saved a comparison CSV, load it; otherwise use final notebook values."""
    possible_files = [
        model_folder / "notebook_comparison_results.csv",
        model_folder / "comparison_df.csv",
        APP_DIR / "notebook_comparison_results.csv",
        APP_DIR / "comparison_df.csv",
    ]

    for path in possible_files:
        if path.exists():
            df = pd.read_csv(path)
            if {"Dataset", "Accuracy", "Loss"}.issubset(df.columns):
                return df[["Dataset", "Accuracy", "Loss"]]

    return pd.DataFrame(
        {
            "Dataset": ["Original IMDB test set", "External product reviews"],
            "Accuracy": [0.7200, 0.6480],
            "Loss": [0.571852, 0.642919],
        }
    )


# -----------------------------
# Tabs
# -----------------------------
tab_sentence, tab_csv, tab_results = st.tabs(
    ["📝 Single Sentence", "📂 CSV Upload", "📊 Full Notebook Results"]
)


with tab_sentence:
    st.subheader("Predict one sentence")
    st.write("Enter one review or sentence. The app will return Positive/Negative sentiment and the confidence score.")

    user_sentence = st.text_area(
        "Enter a sentence",
        placeholder="Example: This product is amazing and I really liked it.",
        height=120,
    )

    if st.button("Predict sentiment", type="primary", use_container_width=True):
        if not user_sentence.strip():
            st.warning("Please enter a sentence first.")
        else:
            prediction_df, oov_rate = predict_sentiment([user_sentence])
            row = prediction_df.iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                pill_class = "positive-pill" if row["Sentiment"] == "Positive" else "negative-pill"
                st.markdown(
                    f"<div class='metric-label'>Predicted sentiment</div><span class='{pill_class}'>{row['Sentiment']}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                show_metric_card("Confidence", f"{row['Confidence'] * 100:.2f}%", "Highest class confidence")
            with col3:
                show_metric_card("OOV rate", f"{oov_rate * 100:.2f}%", "Unknown words compared with tokenizer")

            st.markdown("#### Confidence scores")
            confidence_chart = pd.DataFrame(
                {
                    "Sentiment": ["Positive", "Negative"],
                    "Confidence": [row["Positive_Confidence"], row["Negative_Confidence"]],
                }
            )
            soft_bar_chart(confidence_chart, "Sentiment", "Confidence", title="Positive vs Negative confidence", y_format="%", color_field="Sentiment", height=300)

            with st.expander("Show processed text"):
                st.write(row["Text_Final"])


with tab_csv:
    st.subheader("Upload a CSV file with multiple texts")
    st.write("Upload a CSV file, choose the text column, then predict sentiment for all rows.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Could not read the CSV file.")
            st.exception(e)
            st.stop()

        st.markdown("#### Uploaded data preview")
        st.dataframe(input_df.head(), use_container_width=True)

        default_index = 0
        for candidate in ["Text", "text", "review", "Review", "sentence", "Sentence", "comment", "Comment"]:
            if candidate in input_df.columns:
                default_index = list(input_df.columns).index(candidate)
                break

        text_column = st.selectbox(
            "Select the column that contains the text",
            options=list(input_df.columns),
            index=default_index,
        )

        if st.button("Predict CSV sentiments", type="primary", use_container_width=True):
            texts = input_df[text_column].fillna("").astype(str).tolist()
            results_df, oov_rate = predict_sentiment(texts)

            output_df = input_df.copy()
            output_df["Predicted_Sentiment"] = results_df["Sentiment"]
            output_df["Confidence"] = results_df["Confidence"]
            output_df["Positive_Confidence"] = results_df["Positive_Confidence"]
            output_df["Negative_Confidence"] = results_df["Negative_Confidence"]
            output_df["Text_Final"] = results_df["Text_Final"]

            st.success(f"Prediction finished. CSV OOV rate: {oov_rate * 100:.2f}%")
            show_sentiment_summary(output_df.rename(columns={"Predicted_Sentiment": "Sentiment"}))

            st.markdown("#### Prediction results")
            st.dataframe(output_df, use_container_width=True)

            csv_bytes = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


with tab_results:
    st.subheader("Full notebook results dashboard")
    st.write("This tab summarizes the lab notebook from the first preprocessing step to the final external dataset test.")

    summary_df = pd.DataFrame(NOTEBOOK_RESULTS["model_scores"])
    comparison_df = load_optional_notebook_comparison(model_dir)
    original_external_df = comparison_df.rename(columns={"Dataset": "Model / Dataset"})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_metric_card("Original CNN accuracy", "72.00%", "IMDB test set")
    with col2:
        show_metric_card("ELMo CNN accuracy", "81.33%", "Best notebook result")
    with col3:
        show_metric_card("External accuracy", "64.80%", "Amazon reviews")
    with col4:
        show_metric_card("External OOV rate", "49.98%", "Domain vocabulary shift")

    st.markdown("### 1) Notebook workflow from beginning to end")
    workflow_df = pd.DataFrame(NOTEBOOK_RESULTS["workflow"])
    st.dataframe(workflow_df, use_container_width=True, hide_index=True)

    st.markdown("### 2) Dataset and configuration summary")
    dataset_summary_df = pd.DataFrame(NOTEBOOK_RESULTS["dataset_summary"])
    st.dataframe(dataset_summary_df, use_container_width=True, hide_index=True)

    st.markdown("### 3) Main result comparison")
    soft_bar_chart(summary_df, "Model / Dataset", "Accuracy", title="Accuracy across notebook experiments", y_format="%", color_field="Model / Dataset")
    soft_bar_chart(summary_df, "Model / Dataset", "Loss", title="Loss across notebook experiments", color_field="Model / Dataset")
    st.dataframe(summary_df.style.format({"Accuracy": "{:.4f}", "Loss": "{:.4f}"}), use_container_width=True)

    st.markdown("### 4) Original CNN training curves")
    original_training_df = pd.DataFrame(NOTEBOOK_RESULTS["original_training"])
    col_acc, col_loss = st.columns(2)
    with col_acc:
        soft_line_chart(original_training_df, ["Training Accuracy", "Validation Accuracy"], title="Original CNN accuracy curve")
    with col_loss:
        soft_line_chart(original_training_df, ["Training Loss", "Validation Loss"], title="Original CNN loss curve")

    st.markdown("### 5) ELMo CNN training curves")
    elmo_training_df = pd.DataFrame(NOTEBOOK_RESULTS["elmo_training"])
    col_acc2, col_loss2 = st.columns(2)
    with col_acc2:
        soft_line_chart(elmo_training_df, ["Training Accuracy", "Validation Accuracy"], title="ELMo CNN accuracy curve")
    with col_loss2:
        soft_line_chart(elmo_training_df, ["Training Loss", "Validation Loss"], title="ELMo CNN loss curve")

    st.markdown("### 6) Original vs external dataset comparison")
    soft_bar_chart(original_external_df, "Model / Dataset", "Accuracy", title="Original IMDB test vs external Amazon reviews", y_format="%", color_field="Model / Dataset")
    st.dataframe(comparison_df.style.format({"Accuracy": "{:.4f}", "Loss": "{:.4f}"}), use_container_width=True)

    st.markdown("### 7) External dataset classification report")
    report_df = pd.DataFrame(NOTEBOOK_RESULTS["external_report"])
    st.dataframe(report_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}), use_container_width=True, hide_index=True)
    soft_bar_chart(report_df[report_df["Class"].isin(["Positive", "Negative"])], "Class", "F1-score", title="External dataset F1-score by class", y_format="%", color_field="Class")

    st.markdown("### 8) Final explanation: does performance change?")
    st.markdown(
        """
        <div class="soft-card">
        Yes. The original CNN achieved <b>72.00%</b> accuracy on the original IMDB test set, but dropped to <b>64.80%</b> on the external Amazon product review dataset. This happened mainly because of <b>domain shift</b>: the model was trained on movie-review language but tested on product-review language. The external dataset also had a high <b>OOV rate of 49.98%</b>, meaning almost half of the external tokens were not known by the original training tokenizer. This reduces the model's ability to represent the new reviews correctly. The ELMo CNN performed best on the original test set at <b>81.33%</b> because contextual embeddings capture richer sentence meaning than fixed GloVe/tokenizer features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Optional: notebook code to export exact comparison results"):
        st.code(
            "# Run this at the end of the notebook after comparison_df is created\n"
            "SAVE_DIR = '/content/drive/MyDrive/NLPA Lab/models'\n"
            "comparison_df.to_csv(SAVE_DIR + '/notebook_comparison_results.csv', index=False)\n"
            "print('Saved notebook comparison results for Streamlit app')",
            language="python",
        )
