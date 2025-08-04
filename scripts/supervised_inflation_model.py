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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=True, help="Subreddit-Name (z. B. povertyfinance)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch-Größe für Training und Evaluation")
parser.add_argument("--num_epochs", type=int, default=3, help="Anzahl der Trainingsepochen")
args = parser.parse_args()
SUBREDDIT = args.subreddit
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

# Pfade
RAW_DIR = f"data/raw/{SUBREDDIT}"
INFLATION_CSV = "data/inflation/usa_inflation.csv"  # Typo korrigiert
OUTPUT_DIR = f"data/model_3/{SUBREDDIT}"
MODEL_NAME = "distilbert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Funktion zur Text-Extraktion aus einem Post
def extract_text(post):
    return (post.get("title", "") + "\n\n" + post.get("selftext", "")).strip()

# Laden der Reddit-Daten
def load_reddit_data():
    texts, dates = [], []
    for file in sorted(os.listdir(RAW_DIR)):
        if not file.endswith(".jsonl"): continue
        try:
            # Annahme: Dateiname ist im Format YYYY-MM-DD
            dt = datetime.strptime(file.replace(".jsonl", ""), "%Y-%m-%d")
        except ValueError:
            continue
        with open(os.path.join(RAW_DIR, file), "r") as f:
            monthly_texts = []
            for line in f:
                try:
                    post = json.loads(line)
                    txt = extract_text(post)
                    if txt and len(txt.split()) >= 3:
                        monthly_texts.append(txt)
                except:
                    continue
        if monthly_texts:
            texts.append(" ".join(monthly_texts))
            dates.append(dt)
    return pd.DataFrame({"date": dates, "text": texts})

# Laden der Inflationsdaten (CPI)
def load_inflation():
    return pd.read_csv(INFLATION_CSV, names=["date", "cpi"], header=None, parse_dates=["date"])

# Hauptfunktion
def main():
    print("📥 Lade Daten...")
    df_reddit = load_reddit_data()
    df_cpi = load_inflation()

    # Sicherstellen, dass Daten als datetime vorliegen
    df_reddit["date"] = pd.to_datetime(df_reddit["date"])
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])

    # Daten zusammenführen (merge) basierend auf Datum
    df = pd.merge(df_reddit, df_cpi, on="date", how="inner")
    if df.empty:
        print("❌ Keine übereinstimmenden Daten nach dem Merge. Überprüfe die Datumsformate.")
        return

    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenisierungs-Funktion
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Dataset erstellen
    dataset = Dataset.from_pandas(df[["text", "cpi"]].rename(columns={"cpi": "labels"}))
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

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
    )

    # Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Modell trainieren
    print("🚀 Starte Training...")
    trainer.train()

    # Bestes Modell speichern
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Testen des Modells
    print("🔍 Evaluiere Modell...")
    eval_results = trainer.evaluate()
    print("Evaluationsergebnisse:", eval_results)

    # Vorhersagen auf Testdaten
    predictions = trainer.predict(eval_dataset)
    pred_df = pd.DataFrame({
        "date": eval_dataset["date"],
        "true_cpi": eval_dataset["labels"],
        "predicted_cpi": predictions.predictions.flatten()
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("✅ Vorhersagen gespeichert in:", os.path.join(OUTPUT_DIR, "predictions.csv"))
    print("✅ Training abgeschlossen. Modell gespeichert in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()