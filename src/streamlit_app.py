# Streamlit ≥1.35  ▸  M10 dashboard
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# ─── Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="M10 – Text‑Cls Dashboard", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")

ARTIFACTS_DIR = Path("artifacts")
MODEL = load(ARTIFACTS_DIR / "logreg_tfidf.joblib")
TEST_DF = pd.read_parquet(ARTIFACTS_DIR / "test_preds.parquet")  # y_true, y_prob

# ─── Header metrics ─────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("AUC", f"{auc(*roc_curve(TEST_DF.y_true, TEST_DF.y_prob)[:2]):.3f}")
with c2:
    st.metric("F1", f"{TEST_DF.f1.mean():.3f}")
with c3:
    st.metric("Recall", f"{TEST_DF.recall.mean():.3f}")

# ─── Tabs ───────────────────────────────────────────────────────────────
tab_over, tab_perf, tab_feat, tab_err, tab_live = st.tabs(
    ["Story", "Performance", "Features", "Errors", "Try it"]
)

with tab_over:
    st.subheader("Target balance")
    fig, ax = plt.subplots()
    sns.countplot(x="y_true", data=TEST_DF, ax=ax)
    st.pyplot(fig)

# ─── Performance tab ────────────────────────────────────────────────────
with tab_perf:
    st.subheader("ROC & PR curves with live threshold tuning")

    # ── Cache expensive metric computation ──────────────────────────────
    @st.cache_data(show_spinner=False)
    def _curves(y_true, y_prob):
        fpr, tpr, roc_th = roc_curve(y_true, y_prob)
        prec, rec, pr_th = precision_recall_curve(y_true, y_prob)
        return fpr, tpr, roc_th, prec, rec, pr_th

    fpr, tpr, roc_th, prec, rec, pr_th = _curves(TEST_DF.y_true, TEST_DF.y_prob)

    # ── Side‑by‑side ROC / PR plots ─────────────────────────────────────
    c_roc, c_pr = st.columns(2, gap="large")

    with c_roc:
        fig_r, ax_r = plt.subplots()
        ax_r.plot(fpr, tpr, lw=2)
        ax_r.plot([0, 1], [0, 1], ls="--", lw=1)
        ax_r.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC – AUC {auc(fpr, tpr):.3f}",
        )
        st.pyplot(fig_r, clear_figure=True)

    with c_pr:
        fig_p, ax_p = plt.subplots()
        ax_p.plot(rec, prec, lw=2)
        ax_p.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve")
        st.pyplot(fig_p, clear_figure=True)

    st.divider()

    # ── Threshold slider + dynamic confusion matrix ─────────────────────
    thr = st.slider("Decision threshold", 0.00, 1.00, 0.50, 0.01, key="thr_slider")
    y_pred_thr = (TEST_DF.y_prob >= thr).astype(int)
    cm = confusion_matrix(TEST_DF.y_true, y_pred_thr, labels=[0, 1])

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm,
        square=True,
        linewidths=0.5,
    )
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title(f"Confusion matrix @ threshold {thr:.2f}")
    st.pyplot(fig_cm, clear_figure=True)

    # Derived metrics (optional, but often useful for reviewers)
    tn, fp, fn, tp = cm.ravel()
    st.write(
        f"""
        **Derived metrics @ {thr:.2f}**

        | Metric | Value |
        |--------|-------|
        | Accuracy | {(tp + tn) / cm.sum():.3f} |
        | Precision | {tp / (tp + fp + 1e-12):.3f} |
        | Recall | {tp / (tp + fn + 1e-12):.3f} |
        | Specificity | {tn / (tn + fp + 1e-12):.3f} |
        """
    )


# ─── Feature‑insight tab ────────────────────────────────────────────────
with tab_feat:
    st.subheader("Top coefficients (|β|)")

    # ── 1. Build one tidy DataFrame of term‑coefficient pairs ───────────
    @st.cache_data(show_spinner=False)
    def _coef_frame(model):
        vec = model.named_steps["tfidf"]
        clf = model.named_steps["logreg"]
        names = vec.get_feature_names_out()
        coefs = clf.coef_.ravel()
        return pd.DataFrame({"term": names, "coef": coefs})

    coef_df = _coef_frame(MODEL)

    # ── 2. User chooses how many features per polarity to display ───────
    top_n = st.select_slider("Top‑N per class", [5, 10, 15, 20, 30], value=10)

    pos_terms = coef_df.nlargest(top_n, "coef")  # strong positive weight
    neg_terms = coef_df.nsmallest(top_n, "coef")  # strong negative weight

    # ── 3. Side‑by‑side barplots ────────────────────────────────────────
    fig, (ax_pos, ax_neg) = plt.subplots(
        1,
        2,
        figsize=(10, max(6, top_n / 1.5)),
        sharey=True,
        gridspec_kw={"wspace": 0.25},
    )

    sns.barplot(x="coef", y="term", data=pos_terms, palette="Greens_r", ax=ax_pos).set(
        title="Pushes toward class 1 ➜"
    )

    sns.barplot(
        x="coef",
        y="term",
        data=neg_terms.assign(coef=lambda d: d.coef.abs()),  # show magnitude
        palette="Reds_r",
        ax=ax_neg,
    ).set(title="Pushes toward class 0 ➜")

    ax_pos.set_xlabel("Coefficient")
    ax_neg.set_xlabel("|Coefficient|")
    ax_pos.invert_xaxis()  # positive chart grows to the right
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # ── 4. Contextual examples via expander ─────────────────────────────
    all_terms = pd.concat([pos_terms.term, neg_terms.term]).unique().tolist()
    term_sel = st.selectbox("Inspect term in context", all_terms)

    # pull ≤5 examples from TEST_DF containing the chosen term
    mask = TEST_DF.text.str.contains(rf"\b{term_sel}\b", case=False, regex=True)
    samples = TEST_DF.loc[mask, ["text", "y_true", "y_prob"]].head(5)

    with st.expander(f"Examples containing **{term_sel}** ({samples.shape[0]} shown)"):
        for _, row in samples.iterrows():
            st.markdown(
                f"> {row.text[:300]}{' …' if len(row.text) > 300 else ''}"
                f"\n\n*True label :* **{row.y_true}**   *Pred prob :* **{row.y_prob:.2f}**"
            )


with tab_err:
    st.subheader("Explore misclassifications")
    wrong = TEST_DF.query("y_true != y_pred")
    edited = st.data_editor(wrong[["text", "y_true", "y_prob"]])
    st.caption("Double‑click any row to inspect.")

# ─── Try‑it‑live tab ────────────────────────────────────────────────────
with tab_live:
    st.subheader("Paste text for inference")

    raw_text = st.text_area("Input text", height=180)
    run_btn = st.button("Predict", disabled=not raw_text.strip())

    # ── Helper: highlight influential terms ─────────────────────────────
    import re

    @st.cache_data(show_spinner=False)
    def _highlight_terms(text: str, model, top_k: int = 8) -> str:
        """Return HTML string with top‑K contributing n‑grams highlighted."""
        vec, clf = model.named_steps["tfidf"], model.named_steps["logreg"]
        terms = vec.get_feature_names_out()
        coefs = clf.coef_.ravel()

        X_doc = vec.transform([text]).tocsr()
        idxs, tfidf = X_doc.indices, X_doc.data
        contrib = {terms[i]: tfidf[j] * coefs[i] for j, i in enumerate(idxs)}

        # pick top‑K by absolute contribution
        top = sorted(contrib.items(), key=lambda t: abs(t[1]), reverse=True)[:top_k]

        # highlight in original text (case‑insensitive whole‑word match)
        highlighted = text
        for term, weight in top:
            color = "#C6F6D5" if weight > 0 else "#FEB2B2"  # green / red pastel
            repl = (
                rf'<span style="background:{color};padding:1px 2px;'
                rf'border-radius:3px;"><b>\1</b></span>'
            )
            pattern = re.compile(rf"(?i)\b({re.escape(term)})\b")
            highlighted = pattern.sub(repl, highlighted)

        return f"<p style='line-height:1.5'>{highlighted}</p>"

    if run_btn:
        prob = float(MODEL.predict_proba([raw_text])[0, 1])
        thr = st.session_state.get(
            "thr_slider", 0.50
        )  # fall back to 0.5 if slider not used
        pred = int(prob >= thr)

        # ── Display core metrics ─────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Probability (class 1)", f"{prob:.2%}")
        col2.metric("Predicted class", pred)

        # ── Show highlighted text ───────────────────────────────────────
        st.markdown("###### Influential n‑grams")
        st.markdown(_highlight_terms(raw_text, MODEL), unsafe_allow_html=True)

        # ── JSON payload preview ────────────────────────────────────────
        st.markdown("###### JSON payload")
        st.json(
            {
                "text": raw_text,
                "threshold": thr,
                "probability": prob,
                "predicted_class": pred,
            }
        )
