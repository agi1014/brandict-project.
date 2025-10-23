import cv2
import easyocr
from ultralytics import YOLO
import os

# --- Configuration ---
MODEL_PATH = 'runs/train/brand_detector_run1/weights/best.pt' #<-- Verify this path is correct
TEST_IMAGE_PATH = 'data/test_images/nikestreet.jpg' #<-- Use an image with text for a good test
CONF_THRESHOLD = 0.2 # We can raise this a bit now that we know the model works

def run_brandict_pipeline():
    """
    Runs the first two steps of the Brandict pipeline:
    1. Detects a logo using YOLOv8.
    2. Extracts text from the logo's bounding box using EasyOCR.
    """
    print("--- Starting Brandict Pipeline ---")

    # --- 1. Load Models ---
    print("Loading YOLOv8 logo detector...")
    logo_detector = YOLO(MODEL_PATH)
    
    print("Loading EasyOCR text reader (this may take a moment on first run)...")
    # Initialize EasyOCR with English as the language
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

    # --- 3. Process Each Detection ---
    for i, box in enumerate(results[0].boxes):
        print(f"\n--- Processing Detection #{i+1} ---")
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        brand_name = logo_detector.names[cls_id]
        
        print(f"Detected Brand: {brand_name}")
        
        # Crop the image to the detected bounding box
        logo_crop = original_image[y1:y2, x1:x2]
        
        # --- 4. Extract Text with EasyOCR ---
        print("Running OCR on the detected logo area...")
        ocr_result = text_reader.readtext(logo_crop)
        
        if not ocr_result:
            print("OCR Result: No text found in this area.")
        else:
            detected_text = " ".join([text for _, text, _ in ocr_result])
            print(f"OCR Result: Found text -> '{detected_text}'")

        # Draw the box on the original image for visualization
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, brand_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- 5. Show Final Image ---
    cv2.imshow("Pipeline Result", original_image)
    print("\nPipeline finished. Press any key to close the result window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_brandict_pipeline()