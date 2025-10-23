import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import time
from contextlib import asynccontextmanager
import asyncio
import random

# Import necessary components from your logic script
import sys
sys.path.append('./scripts')
from brandict_logic import logo_detector

# --- Global variables ---
ANALYZED_DATA = pd.DataFrame()
COUNTRY_SENTIMENT_DATA = {}

def load_preprocessed_data():
    """Loads and aggregates the pre-analyzed data."""
    global ANALYZED_DATA, COUNTRY_SENTIMENT_DATA
    analyzed_data_path = 'data/analyzed_data.csv'
    if not os.path.exists(analyzed_data_path):
        print(f"CRITICAL ERROR: '{analyzed_data_path}' not found.")
        return

    print("Loading pre-processed sentiment data...")
    ANALYZED_DATA = pd.read_csv(analyzed_data_path)
    
    for brand in ANALYZED_DATA['brand'].unique():
        brand_key = str(brand).lower()
        COUNTRY_SENTIMENT_DATA[brand_key] = {}
        brand_df = ANALYZED_DATA[ANALYZED_DATA['brand'] == brand]
        for country in brand_df['country'].unique():
            country_df = brand_df[brand_df['country'] == country]
            sentiment_counts = country_df['sentiment'].value_counts()
            total = len(country_df)
            if total > 0:
                pos_pct = int((sentiment_counts.get("positive", 0) / total) * 100)
                neg_pct = int((sentiment_counts.get("negative", 0) / total) * 100)
                neu_pct = 100 - pos_pct - neg_pct
                COUNTRY_SENTIMENT_DATA[brand_key][str(country)] = {
                    "positive": pos_pct, "negative": neg_pct, "neutral": neu_pct
                }
    print("Data loaded successfully. Server is ready.")

async def call_generative_api_simulator(prompt: str) -> dict:
    """
    IMPROVED SIMULATOR: Generates more realistic and dynamic suggestions.
    """
    print("\n--- SIMULATING Improved Generative AI ---")
    await asyncio.sleep(1) # Simulate network delay

    country_name = "Global"
    try:
        # A better way to find the country name in the prompt
        if "in " in prompt:
            country_name = prompt.split("in ")[1].split(":")[0].strip().split("'")[0]
    except IndexError:
        pass

    print(f"DEBUG: Generating simulated suggestions for: {country_name}")

    # A bank of realistic suggestion templates
    suggestion_templates = {
        "positive": [
            f"- Double down on what's working in {country_name} by launching a user-generated content campaign celebrating local brand stories.",
            f"- Amplify positive sentiment by collaborating with high-profile influencers from {country_name} for an authentic endorsement.",
            f"- Introduce a loyalty program in {country_name} to reward dedicated customers and further strengthen brand affinity."
        ],
        "negative": [
            f"- Launch a targeted PR campaign in {country_name} to address the specific criticisms and rebuild trust.",
            f"- Engage directly with dissatisfied customers on social media in {country_name} with a dedicated support team to resolve issues publicly.",
            f"- Partner with a respected local charity or organization in {country_name} to demonstrate community commitment and improve public image."
        ],
        "neutral": [
            f"- Run a series of exciting, high-energy pop-up events in major cities within {country_name} to create buzz and generate conversation.",
            f"- Launch a limited-edition product exclusive to {country_name}, designed with local cultural tastes in mind.",
            f"- Collaborate with an unexpected but popular local brand in {country_name} to create a surprising and shareable product."
        ]
    }

    # Decide which suggestions to show based on the prompt
    sentiment_focus = "neutral"
    if "highly negative" in prompt or "Negative:" in prompt:
        sentiment_focus = "negative"
    elif "Positive:" in prompt:
        sentiment_focus = "positive"

    # Pick two random suggestions from the appropriate list
    final_suggestions = random.sample(suggestion_templates[sentiment_focus], 2)
    
    return {"suggestions": "\n".join(final_suggestions)}


# --- API Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_preprocessed_data()
    yield

app = FastAPI(title="Brandict API", version="4.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/detect_brand_from_camera/")
async def detect_brand_from_camera(image_file: UploadFile = File(...)):
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, "camera_frame.jpg")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        # IMPROVEMENT: Lowered confidence for more sensitive detection
        results = logo_detector(temp_file_path, conf=0.20)
        
        detections = []
        for box in results[0].boxes:
            brand_name = logo_detector.names[int(box.cls[0])]
            coordinates = box.xyxy[0].tolist()
            detections.append({"brand": brand_name, "box": coordinates})
            
        return {"detections": detections}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ... (other endpoints remain the same)
@app.get("/get_brand_sentiment/{brand_name}")
def get_brand_sentiment(brand_name: str):
    brand_key = brand_name.lower()
    if ANALYZED_DATA.empty: return {"error": "No data available."}
    brand_df = ANALYZED_DATA[ANALYZED_DATA['brand'].str.lower() == brand_key]
    if brand_df.empty: return {"error": f"No data found for brand: {brand_name}"}
    sentiment_counts = brand_df['sentiment'].value_counts().to_dict()
    return {
        "brand": brand_name,
        "overall_sentiment": {
            "positive": int(sentiment_counts.get("positive", 0)),
            "negative": int(sentiment_counts.get("negative", 0)),
            "neutral": int(sentiment_counts.get("neutral", 0))
        },
        "regional_sentiment": COUNTRY_SENTIMENT_DATA.get(brand_key, {})
    }

@app.post("/get_global_suggestions/")
async def get_global_suggestions(sentiment_data: dict):
    brand = sentiment_data.get("brand", "the brand")
    prompt = f"GLOBAL suggestions for {brand} based on {sentiment_data.get('overall_sentiment', {})}"
    return await call_generative_api_simulator(prompt)

@app.post("/get_country_suggestion/{brand_name}/{country_name}")
async def get_country_suggestion(brand_name: str, country_name: str):
    brand_key = brand_name.lower()
    country_data = COUNTRY_SENTIMENT_DATA.get(brand_key, {}).get(country_name)
    if not country_data:
        return {"suggestions": f"Error: No sentiment data found for {country_name}."}
    prompt = (f"Suggestions for {brand_name} in {country_name}: Positive: {country_data['positive']}%, Negative: {country_data['negative']}%")
    return await call_generative_api_simulator(prompt)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

