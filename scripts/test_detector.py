from ultralytics import YOLO
import cv2
import os

# --- Configuration ---
# 1. VERIFY THIS PATH: Check your 'runs/train/' folder. Is this the latest run? 
#    If you have a 'brand_detector_run2' folder, change this path!
MODEL_PATH = 'runs/train/brand_detector_run1/weights/best.pt' 

# 2. Path to your test image
TEST_IMAGE_PATH = 'data/test_images/test1.jpg' 

# 3. NEW: Confidence Threshold (a value between 0.0 and 1.0)
#    We are setting it very low (10%) to see all potential detections.
CONF_THRESHOLD = 0.1

def test_detector():
    print("--- Testing Logo Detector with Low Confidence Threshold ---")

    test_dir = os.path.dirname(TEST_IMAGE_PATH)
    os.makedirs(test_dir, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please check the folder name in 'runs/train/' and update the path.")
        return

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
        return

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # --- UPDATED: We now pass our confidence threshold to the model ---
    print(f"Running detection with a confidence threshold of {CONF_THRESHOLD}")
    results = model(TEST_IMAGE_PATH, conf=CONF_THRESHOLD)

    # --- Process and display the results ---
    img = cv2.imread(TEST_IMAGE_PATH)
    detections_found = False
    
    for r in results:
        for box in r.boxes:
            detections_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            label = f'{model.names[cls_id]} {conf:.2f}'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    if not detections_found:
        print("\n--- No detections found even with the low threshold. ---")
    
    cv2.imshow('Logo Detection Result', img)
    print("\nAn image window has opened. Press any key to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detector()
