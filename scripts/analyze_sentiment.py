import pandas as pd
from transformers import pipeline
import torch
import re

# --- Configuration ---
INPUT_CSV = 'data/raw_text_data.csv'
OUTPUT_CSV = 'data/sentiment_results.csv'
# --- ALTERNATIVE MULTILINGUAL MODEL ---
MODEL_NAME = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'

# --- Text Cleaning Function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main analysis function ---
def analyze_sentiment():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: The input file was not found at {INPUT_CSV}")
        return

    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'].str.strip() != '']
    
    if df.empty:
        print("The CSV file is empty or contains no valid text after cleaning. Exiting.")
        return

    print(f"Loading ALTERNATIVE MULTILINGUAL model: {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline('zero-shot-classification', model=MODEL_NAME, device=device)
    
    candidate_labels = ['positive', 'negative', 'neutral']
    
    results = []
    total_rows = len(df)
    
    print(f"Starting multilingual sentiment analysis on {total_rows} text entries...")
    
    for i, row in df.iterrows():
        text_to_classify = row['cleaned_text']
        
        # This model has a different max length, but truncating is still safe
        if len(text_to_classify) > 512:
            text_to_classify = text_to_classify[:512]

        try:
            output = classifier(text_to_classify, candidate_labels)
            
            results.append({
                'brand': row['brand'],
                'original_text': row['text'],
                'sentiment': output['labels'][0],
                'confidence_score': output['scores'][0]
            })
        except Exception as e:
            print(f"Could not process row {i+1}. Error: {e}")
        
        print(f"Processed {i + 1}/{total_rows}...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nAnalysis complete! Multilingual results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    analyze_sentiment()
