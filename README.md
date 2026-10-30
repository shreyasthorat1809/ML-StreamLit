# ReelFeel: Movie & Show Review Sentiment Analyzer

A Streamlit web app for a machine learning mini project.
It classifies a review as **positive**, **negative** or **neutral** using
**TF-IDF** features and **Logistic Regression** (scikit-learn).

## Features
- Analyse a single review with predicted sentiment, confidence and the words that drove the decision
- Batch mode: paste many reviews or upload a CSV, then download the results
- Performance page: accuracy, precision, recall, F1, confusion matrix, per-class report
- Dataset page: class balance, review lengths, searchable table

## Project structure
```
MyProject/
├── app.py                  # the whole app: data, model, prediction and UI
├── requirements.txt        # Python dependencies
├── reviews.csv             # training dataset (columns: review, sentiment)
├── model.pkl               # created automatically on first run, then reused
├── .streamlit/config.toml  # theme colours
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repository and branch, and set **Main file path** to `app.py`.
4. Click **Deploy** and wait for the build. Your public URL appears at the top.

## Notes
- The model is trained once and saved to `model.pkl`. It retrains automatically if `reviews.csv`
  changes or the scikit-learn version differs from the one that saved the file.
- The demo dataset is generated from sentence templates, so test scores are near 100%.
  Replace `reviews.csv` with real reviews (columns `review` and `sentiment`) for realistic results.
