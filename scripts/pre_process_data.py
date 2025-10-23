import pandas as pd
import sys
import os

# Import the necessary components from your logic script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from brandict_logic import sentiment_classifier, clean_text

# --- Configuration ---
INPUT_FILE = 'data/raw_text_data.csv'
OUTPUT_FILE = 'data/analyzed_data.csv'

def pre_process_and_save():
    """
    Reads the raw text data, performs sentiment analysis on every row,
    and saves the final, analyzed data to a new CSV file.
    This is the slow part that we only need to run once.
    """
    print(f"--- Starting one-time data pre-processing ---")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{INPUT_FILE}'. Please run the collection script first.")
        return

    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.dropna(subset=['cleaned_text', 'country'], inplace=True)
    df = df[df['cleaned_text'].str.strip() != '']

    print("Analyzing sentiment for each comment... (This will take a while)")
    sentiment_labels = ['positive', 'negative', 'neutral']
    sentiments = []
    total_rows = len(df)

    for i, text in enumerate(df['cleaned_text']):
        if not text:
            sentiments.append("neutral")
            continue
        try:
            result = sentiment_classifier(text, sentiment_labels)
            sentiments.append(result['labels'][0])
            print(f"Processed {i + 1}/{total_rows}...")
        except Exception as e:
            print(f"Error processing row {i + 1}: {e}")
            sentiments.append("neutral") # Default to neutral on error

    df['sentiment'] = sentiments

    # Save only the necessary columns to the new file
    final_df = df[['brand', 'country', 'sentiment']]
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n--- Pre-processing complete! ---")
    print(f"Analyzed data has been saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    pre_process_and_save()