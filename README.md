# Streamlit Sentiment Analysis App - Final Version

This final version fixes the visualization layout:

- White chart cards
- Centered visualizations
- Charts resize to the block width
- No cropped bars, titles, or axis labels
- Long labels use horizontal bars
- Light Streamlit header/theme
- No settings sidebar

## Required folder structure

Place the app beside your saved `models` folder:

```text
NLPA Lab/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── models/
    ├── original_CNN_model.keras
    ├── tokenizer.pkl
    └── config.pkl
```

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

The app reuses your saved tokenizer and model. It does not fit a new tokenizer.
