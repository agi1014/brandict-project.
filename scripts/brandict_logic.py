from ultralytics import YOLO
import easyocr
from transformers import pipeline
import torch
import cv2
import os
import re

# --- Configuration ---
LOGO_MODEL_PATH = 'runs/train/brand_detector_run1/weights/best.pt'
SENTIMENT_MODEL_NAME = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
CONF_THRESHOLD = 0.2

# --- 1. Initialize All Models (Load them once on startup to save time) ---
print("Initializing Brandict AI models... (This may take a moment)")
try:
    # Load Logo Detector
    logo_detector = YOLO(LOGO_MODEL_PATH)
    # Load Text Reader
    text_reader = easyocr.Reader(['en'])
    # Load Sentiment Classifier
    device = 0 if torch.cuda.is_available() else -1
    sentiment_classifier = pipeline('zero-shot-classification', model=SENTIMENT_MODEL_NAME, device=device)
    print("Models initialized successfully!")
except Exception as e:
    print(f"Error initializing models: {e}")
    # In a real app, you might want to exit or handle this more gracefully
    logo_detector = text_reader = sentiment_classifier = None

def clean_text(text):
    """A simple text cleaning function."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_content(image_path, post_text):
    """
    The main function that processes an image and its associated text post.
    """
    if not all([logo_detector, text_reader, sentiment_classifier]):
        return {"error": "Models are not initialized."}

    print(f"\n--- Processing new content ---")
    print(f"Image: {image_path}")
    print(f"Post Text: '{post_text}'")

    if not os.path.exists(image_path):
        return {"error": "Image not found."}

    # --- 2. Analyze the Sentiment of the Post Text ---
    cleaned_post_text = clean_text(post_text)
    sentiment_labels = ['positive', 'negative', 'neutral']
    sentiment_result = sentiment_classifier(cleaned_post_text, sentiment_labels)
    
    post_sentiment = sentiment_result['labels'][0]
    sentiment_score = sentiment_result['scores'][0]
    
    print(f"Step 1: Sentiment Analysis -> {post_sentiment} (Score: {sentiment_score:.2f})")

    # --- 3. Detect Logos in the Image ---
    logo_results = logo_detector(image_path, conf=CONF_THRESHOLD)
    image = cv2.imread(image_path)
    
    final_outputs = []
    
    if len(logo_results[0].boxes) == 0:
        print("Step 2: Logo Detection -> No logos found.")
        # Even if no logo is found, we still have the text sentiment
        return [{
            "detected_brand": "None",
            "post_sentiment": post_sentiment,
            "sentiment_confidence": round(sentiment_score, 2),
            "text_from_image_ocr": "None",
            "original_post_text": post_text
        }]

    print(f"Step 2: Logo Detection -> Found {len(logo_results[0].boxes)} logo(s).")

    # --- 4. Process Each Detected Logo ---
    for i, box in enumerate(logo_results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        brand_name = logo_detector.names[int(box.cls[0])]
        
        logo_crop = image[y1:y2, x1:x2]
        ocr_result = text_reader.readtext(logo_crop)
        
        detected_text_in_image = " ".join([res[1] for res in ocr_result]) if ocr_result else "None"

        # --- 5. Combine all data into a structured result ---
        output_data = {
            "detected_brand": brand_name,
            "post_sentiment": post_sentiment,
            "sentiment_confidence": round(sentiment_score, 2),
            "text_from_image_ocr": detected_text_in_image,
            "original_post_text": post_text
        }
        final_outputs.append(output_data)
        
    print("\n--- Final Structured Output ---")
    for output in final_outputs:
        print(output)
        
    return final_outputs
