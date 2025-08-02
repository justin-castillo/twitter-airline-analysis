# Twitter sentiment analysis

## Project overview
This repository delivers an end‑to‑end natural‑language‑processing pipeline for the **Twitter US Airline Sentiment** dataset (≈14 600 tweets from Feb 2015). It moves from raw CSV to a production‑style inference script, demonstrating professional data‑science workflow: data cleaning, exploratory analysis, feature engineering, model training, evaluation, and deployment‑ready packaging.

## Objectives
1. Quantify overall positive, neutral, and negative sentiment toward major U.S. airlines.  
2. Identify the top drivers of negative sentiment (e.g., late flight, rude service).  
3. Benchmark classical ML (logistic regression, linear SVM, gradient boosting) against a simple LSTM baseline.  
4. Package the best model behind a lightweight REST endpoint for real‑time scoring.

## Dataset
* **Source** Kaggle → *Twitter US Airline Sentiment*  
* **Size**  ≈14 600 English‑language tweets  
* **Labels** `positive`, `neutral`, `negative` + 11 negative‑reason categories  
* **Timeframe** Feb – Mar 2015  

## Folder structure
```
twitter-sentiment-analysis/
│
├─ data/                # Raw and processed datasets
├─ notebooks/           # EDA and modelling notebooks (clear, modular)
├─ src/
│   ├─ data_prep.py     # Reproducible cleaning + feature pipeline
│   ├─ train.py         # Model training + hyper‑parameter search
│   └─ predict.py       # Single‑tweet inference script / REST hook
├─ reports/             # Generated figures and model cards
└─ environment.yml      # Locked, reproducible Conda environment
```

## Tools used
| Tool | Why it was chosen |
|------|-------------------|
| **Python 3.11** | Widely adopted language for modern NLP and data pipelines. |
| **pandas** | Efficient tabular manipulation and cleaning. |
| **scikit‑learn** | Fast prototyping of classical models and grid/RandomizedSearchCV. |
| **NLTK / spaCy** | Robust tokenization, lemmatization, and stop‑word management. |
| **TensorFlow / Keras** | Quick LSTM baseline without heavy infrastructure. |
| **Optuna** | More efficient hyper‑parameter tuning than manual grid search. |
| **matplotlib / seaborn** | Publication‑quality EDA and error‑analysis visuals. |
| **FastAPI** | Minimal‑overhead REST interface for model inference. |
| **pre‑commit + black + ruff** | Enforce consistent code style and linting on every commit. |

## Quick‑start
```bash
# 1.  Clone repo and set working dir
git clone https://github.com/<your‑user>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# 2.  Create reproducible environment
conda env create -f environment.yml
conda activate twitter-sentiment-env

# 3.  Download data (Kaggle API required)
kaggle datasets download -d crowdflower/twitter-airline-sentiment -p data/raw --unzip

# 4.  Run pipeline end‑to‑end
python src/train.py         # trains models + logs metrics
python src/predict.py -t "Flight was delayed but crew handled it well."
```

## Results
| Model | Macro‑F1 | Accuracy |
|-------|----------|----------|
| Logistic regression (TF‑IDF) | 0.83 | 0.84 |
| SVM (char‑ngrams) | 0.85 | 0.86 |
| Bi‑LSTM (GloVe 100 d) | 0.87 | 0.88 |

*(Full confusion matrices and SHAP‑style feature attributions are in `reports/`.)*

## Roadmap
- [ ] Deploy inference endpoint to Render/Heroku with CI/CD.  
- [ ] Add multilingual transfer‑learning experiment with XLM‑R.  
- [ ] Replace LSTM with lightweight DistilBERT for higher recall.  

## License
Code released under MIT; dataset governed by **CC BY‑NC‑SA 4.0** per original source.
