"""
Streamlit UI (human-friendly).
Run: streamlit run src_2/streamlit_app.py
"""

import sys
from pathlib import Path
PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import streamlit as st
from src_2.predict import predict_texts
from src_2.paths import MODELS_DIR, DISTILBERT_DIR, DEFAULT_SKLEARN_PIPELINE

st.set_page_config(page_title="Twitter Airline Sentiment", layout="centered")

st.title("Twitter Airline Sentiment – Demo")
st.caption("")

with st.sidebar:
    st.header("Settings")
    found_hf = DISTILBERT_DIR.exists()
    found_skl = DEFAULT_SKLEARN_PIPELINE is not None
    if found_hf and found_skl:
        default_ix = 0
        options = ["distilbert", "sklearn"]
    elif found_hf:
        default_ix = 0
        options = ["distilbert"]
    elif found_skl:
        default_ix = 0
        options = ["sklearn"]
    else:
        options = ["distilbert", "sklearn"]
        default_ix = 0

    model_name = st.radio("Model", options=options, index=default_ix, horizontal=True)
    batch_mode = st.checkbox("Batch mode (one text per line)", value=False)
    show_all = st.checkbox("Return all class scores", value=False)

    st.write("**Detected paths**")
    st.code(
        f"MODELS_DIR = {MODELS_DIR}\n"
        f"DISTILBERT_DIR = {DISTILBERT_DIR}  [{'OK' if found_hf else 'MISSING'}]\n"
        f"SKLEARN_PIPELINE = {DEFAULT_SKLEARN_PIPELINE}  [{'OK' if found_skl else 'MISSING'}]"
    )

if batch_mode:
    raw = st.text_area("Texts (one per line)", height=220,
                       placeholder="I love this airline!\nThe flight was delayed.\nmeh")
    inputs = [ln for ln in (raw.splitlines() if raw else []) if ln.strip()]
else:
    raw = st.text_area("Text", height=160,
                       placeholder="Type something like: The service was amazing!")
    inputs = [raw] if raw and raw.strip() else []

if st.button("Predict"):
    if not inputs:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running inference…"):
            preds = predict_texts(inputs, model_name=model_name, return_all_scores=show_all)

        st.success(f"Done. {len(preds)} prediction(s).")
        for i, p in enumerate(preds):
            st.subheader(f"Example {i+1}")
            if "top" in p:  # full score mode
                st.write(f"**Label:** {p['top']['label']}  |  **Score:** {p['top']['score']:.4f}")
                st.json(p["all_scores"])
            else:
                label = p.get("label", "?")
                score = p.get("score", None)
                if score is None:
                    st.write(f"**Label:** {label}")
                else:
                    st.write(f"**Label:** {label}  |  **Score:** {score:.4f}")
