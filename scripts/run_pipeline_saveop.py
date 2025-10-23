import cv2
import easyocr
from ultralytics import YOLO
import os

# --- Configuration ---
MODEL_PATH = 'runs/train/brand_detector_run1/weights/best.pt' #<-- Verify this path
TEST_IMAGE_PATH = 'data/test_images/shoebox.jpg' #<-- Use an image with some text
OUTPUT_DIR = 'runs/pipeline_results' #<-- Results will be saved here
CONF_THRESHOLD = 0.2

def run_brandict_pipeline():
    print("--- Starting Brandict Pipeline (Saving Output) ---")

    # --- 1. Load Models ---
    print("Loading YOLOv8 logo detector...")
    logo_detector = YOLO(MODEL_PATH)
    
    print("Loading EasyOCR text reader...")
    text_reader = easyocr.Reader(['en'])

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
        
    # --- 2. Detect Logos ---
    print(f"\nRunning logo detection on {TEST_IMAGE_PATH}...")
    results = logo_detector(TEST_IMAGE_PATH, conf=CONF_THRESHOLD)
    
    original_image = cv2.imread(TEST_IMAGE_PATH)
    
    if len(results[0].boxes) == 0:
        print("No logos were detected in the image.")
        return

    # --- 3. Process Detections ---
    for i, box in enumerate(results[0].boxes):
        print(f"\n--- Processing Detection #{i+1} ---")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        brand_name = logo_detector.names[int(box.cls[0])]
        
        print(f"Detected Brand: {brand_name}")
        
        logo_crop = original_image[y1:y2, x1:x2]
        
        # --- 4. Extract Text ---
        print("Running OCR on the detected logo area...")
        ocr_result = text_reader.readtext(logo_crop)
        
        if not ocr_result:
            print("OCR Result: No text found in this area.")
        else:
            detected_text = " ".join([text for _, text, _ in ocr_result])
            print(f"OCR Result: Found text -> '{detected_text}'")

        # Draw the box on the image for visualization
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, brand_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- 5. Save the Final Image ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'result_' + os.path.basename(TEST_IMAGE_PATH))
    cv2.imwrite(output_path, original_image)
    
    print(f"\nPipeline finished. Result image saved to: {output_path}")

if __name__ == "__main__":
    run_brandict_pipeline()