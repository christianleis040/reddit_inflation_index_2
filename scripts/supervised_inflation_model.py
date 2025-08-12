import os
import json
import argparse
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Subreddit-Name (z. B. food)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch-Größe für Training und Evaluation")
parser.add_argument("--num_epochs", type=int, default=10, help="Anzahl der Trainingsepochen")
args = parser.parse_args()
SUBREDDIT = args.subreddit
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

# Pfade
RAW_DIR = f"data/raw/{SUBREDDIT}"
INFLATION_CSV = "data/inflation/usa_inflation.csv"
OUTPUT_DIR = f"data/model_3_1/{SUBREDDIT}"
MODEL_NAME = "distilbert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Funktion zur Text-Extraktion aus einem Post
def extract_text(post):
    return (post.get("title", "") + "\n\n" + post.get("selftext", "")).strip()

# VADER-Sentiment-Analysator
analyzer = SentimentIntensityAnalyzer()

# Laden der Reddit-Daten
def load_reddit_data():
    texts, dates, sentiments, n_posts_list = [], [], [], []
    for file in sorted(os.listdir(RAW_DIR)):
        if not file.endswith(".jsonl"): continue
        try:
            dt = datetime.strptime(file.replace(".jsonl", ""), "%Y-%m-%d")
        except ValueError:
            continue
        with open(os.path.join(RAW_DIR, file), "r") as f:
            monthly_texts = []
            monthly_sentiments = []
            n_posts = 0
            for line in f:
                try:
                    post = json.loads(line)
                    txt = extract_text(post)
                    if txt and len(txt.split()) >= 3:
                        monthly_texts.append(txt)
                        sentiment = analyzer.polarity_scores(txt)["compound"]
                        monthly_sentiments.append(sentiment)
                        n_posts += 1
                except:
                    continue
            if monthly_texts:
                texts.append(" ".join(monthly_texts))
                dates.append(dt)
                sentiments.append(np.mean(monthly_sentiments) if monthly_sentiments else 0.0)
                n_posts_list.append(n_posts)
    return pd.DataFrame({"date": dates, "text": texts, "vader_sentiment": sentiments, "n_posts": n_posts_list})

# Laden der Inflationsdaten (CPI)
def load_inflation():
    print("🔍 Überprüfe Inflations-CSV-Datei:")
    with open(INFLATION_CSV, 'r') as f:
        print(f.read().splitlines()[:5])
    df = pd.read_csv(INFLATION_CSV, skiprows=1, names=["date", "cpi"])
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        print(f"❌ Fehler beim Parsen der Datumsspalte: {e}")
        raise
    return df

# Hauptfunktion
def main():
    print("📥 Lade Daten...")
    df_reddit = load_reddit_data()
    df_cpi = load_inflation()

    # Sicherstellen, dass Daten als datetime vorliegen
    df_reddit["date"] = pd.to_datetime(df_reddit["date"])
    df_cpi["date"] = pd.to_datetime(df_cpi["date"], errors="coerce")

    # Daten zusammenführen (merge) basierend auf Datum
    df = pd.merge(df_reddit, df_cpi, on="date", how="inner")
    if df.empty:
        print("❌ Keine übereinstimmenden Daten nach dem Merge. Überprüfe die Datumsformate.")
        return

    print(f"✅ Gemergte Daten: {len(df)} Zeilen")

    # Z-Normalisierung der CPI-Werte
    scaler = StandardScaler()
    df["cpi_normalized"] = scaler.fit_transform(df[["cpi"]])

    # Sortieren nach Datum für chronologischen Split
    df = df.sort_values("date")

    # Chronologischer Split (80% Training, 20% Test)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Baseline-Modell: Ridge Regression mit TF-IDF und VADER-Sentiment
    print("🚀 Trainiere Baseline-Modell (Ridge Regression)...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(train_df["text"])
    X_test_tfidf = vectorizer.transform(test_df["text"])
    X_train = np.hstack([X_train_tfidf.toarray(), train_df[["vader_sentiment"]].values])
    X_test = np.hstack([X_test_tfidf.toarray(), test_df[["vader_sentiment"]].values])
    y_train = train_df["cpi_normalized"]
    y_test = test_df["cpi_normalized"]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_test_pred = ridge.predict(X_test)
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
    print(f"Baseline Ridge RMSE: {ridge_test_rmse:.4f}")

    # Vorhersagen für alle Daten (Baseline)
    X_full = np.hstack([vectorizer.transform(df["text"]).toarray(), df[["vader_sentiment"]].values])
    ridge_full_pred = scaler.inverse_transform(ridge.predict(X_full).reshape(-1, 1)).flatten()

    # Tokenizer laden für BERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenisierungs-Funktion
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Trainings-, Test- und Gesamt-Dataset erstellen
    train_dataset = Dataset.from_pandas(train_df[["text", "cpi_normalized", "date", "n_posts", "vader_sentiment"]].rename(columns={"cpi_normalized": "labels"}))
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type="numpy")

    test_dataset = Dataset.from_pandas(test_df[["text", "cpi_normalized", "date", "n_posts", "vader_sentiment"]].rename(columns={"cpi_normalized": "labels"}))
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type="numpy")

    full_dataset = Dataset.from_pandas(df[["text", "cpi_normalized", "date", "n_posts", "vader_sentiment"]].rename(columns={"cpi_normalized": "labels"}))
    full_dataset = full_dataset.map(tokenize_function, batched=True)
    full_dataset.set_format(type="numpy")

    # Modell laden (DistilBERT für Regression)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    # Metriken für Regression definieren
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}

    # Training-Argumente
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        learning_rate=2e-5,
    )

    # Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Manuelles Early Stopping
    print("🚀 Starte BERT-Training...")
    best_rmse = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        trainer.train(resume_from_checkpoint=(epoch > 0))
        eval_results = trainer.evaluate()
        current_rmse = eval_results["eval_rmse"]
        print(f"Epoche {epoch+1} - RMSE: {current_rmse:.4f}")
        if current_rmse < best_rmse - 0.01:
            best_rmse = current_rmse
            patience_counter = 0
            trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Early Stopping ausgelöst!")
            break

    # Bestes Modell laden
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Testen des Modells
    print("🔍 Evaluiere BERT-Modell...")
    eval_results = trainer.evaluate()
    print("BERT Evaluationsergebnisse:", eval_results)

    # Vorhersagen auf Testdaten (BERT)
    test_predictions = trainer.predict(test_dataset)
    test_pred_df = pd.DataFrame({
        "subreddit": [SUBREDDIT] * len(test_dataset),
        "date": list(test_dataset["date"]),
        "n_posts": list(test_dataset["n_posts"]),
        "avg_sentiment": scaler.inverse_transform(test_predictions.predictions).flatten(),
        "true_cpi": scaler.inverse_transform(np.array(test_dataset["labels"]).reshape(-1, 1)).flatten(),
        "vader_sentiment": list(test_dataset["vader_sentiment"])
    })
    test_pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print("✅ Test-Vorhersagen gespeichert in:", os.path.join(OUTPUT_DIR, "test_predictions.csv"))

    # Vorhersagen auf dem gesamten Datensatz (BERT)
    print("🔍 Generiere Vorhersagen für alle Daten (BERT)...")
    full_predictions = trainer.predict(full_dataset)
    pred_df = pd.DataFrame({
        "subreddit": [SUBREDDIT] * len(full_dataset),
        "date": list(full_dataset["date"]),
        "n_posts": list(full_dataset["n_posts"]),
        "avg_sentiment": scaler.inverse_transform(np.array(full_predictions.predictions).reshape(-1, 1)).flatten(),
        "vader_sentiment": list(full_dataset["vader_sentiment"])
    })

    # In data/sentiment speichern
    sentiment_output_dir = "data/sentiment"
    os.makedirs(sentiment_output_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(sentiment_output_dir, f"model_3_{SUBREDDIT}.csv"), index=False)
    print("✅ Vorhersagen gespeichert in:", os.path.join(sentiment_output_dir, f"model_3_{SUBREDDIT}.csv"))
    print("✅ Training abgeschlossen. Modell gespeichert in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()