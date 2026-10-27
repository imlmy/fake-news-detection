# Fake News Detection

A machine learning project that classifies news articles as **real** or **fake** using NLP preprocessing and TF-IDF features, comparing **Logistic Regression** and **Support Vector Machine (LinearSVC)**. Built following the **CRISP-DM** methodology.

## Dataset

WELFake (Verma et al., 2021): 72,134 labelled news articles combined from four sources (Kaggle, McIntire, Reuters, BuzzFeed Political).

- Labels: `0` = Fake, `1` = Real
- The dataset is **not included** in this repository. Download `WELFake_Dataset.csv` from [Zenodo](https://zenodo.org/record/4561253) and place it in the project folder before running the notebook.

## Approach

**1. Data cleaning** (72,134 → 62,100 articles)
- Removed records with missing title or text
- Removed short or degenerate entries (≤ 3 words), which were 99.7% labelled Real and could cause label leakage
- Removed duplicate article text, keeping the first occurrence

**2. Text preprocessing (NLTK)**
- Text normalisation and tokenisation (`word_tokenize`)
- Stop-word removal
- POS-aware lemmatisation (`WordNetLemmatizer` + `pos_tag`), which fixed errors from noun-only lemmatisation (e.g. "uses" → "us")

**3. Feature extraction**
- TF-IDF vectorisation (`max_features=5000`)
- 80/20 stratified train-test split (`random_state=42`)

**4. Modelling and evaluation**
- Logistic Regression vs. LinearSVC
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC (Fake treated as the positive class)
- Confusion matrices and top TF-IDF terms per class for interpretability

## Results

| Metric    | Logistic Regression | SVM (LinearSVC) |
|-----------|--------------------:|----------------:|
| Accuracy  | 0.9507 | **0.9522** |
| Precision | 0.9569 | **0.9583** |
| Recall    | 0.9547 | **0.9558** |
| F1-Score  | 0.9558 | **0.9570** |
| ROC-AUC   | 0.9502 | **0.9517** |

SVM performed slightly better on every metric, while Logistic Regression offers similar performance with more interpretable coefficients.

## Key Finding

Both models ranked **"reuters"** and **"breitbart"** among the strongest Fake indicators. This suggests the models partly learn source or publication references rather than purely general linguistic patterns, a known risk with WELFake. Removing source references during preprocessing is a direction for future work.

## Tech Stack

Python, pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells. NLTK resources (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) are downloaded inside the notebook.

## Reference

Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word embedding over linguistic features for fake news detection. *IEEE Transactions on Computational Social Systems*.
