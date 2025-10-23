from ultralytics import YOLO
import torch

def train_model():
    """
    Trains a YOLOv8 model on the custom brand dataset.
    """
    print("--- Starting YOLOv8 Model Training ---")

    # Check if a GPU is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pre-trained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest.
    # For higher accuracy, you could use 'yolov8s.pt' or 'yolov8m.pt'.
    model = YOLO('yolov8n.pt')
    model.to(device) # Move the model to the selected device

    # --- Start the training process ---
    try:
        results = model.train(
            data='data.yaml',      # Path to your dataset configuration file
            epochs=50,             # Number of times to loop through the dataset. 50 is a good start.
            imgsz=640,             # Resize images to 640x640 pixels for training
            batch=8,               # Number of images to process at once. Decrease if you run out of memory.
            project='runs/train',  # Directory to save training runs
            name='brand_detector_run1', # Name for this specific training run
            exist_ok=True          # Allow overwriting of a previous run with the same name
        )
        print("\n--- Training successfully completed! ---")
        print("Your trained model and results are saved in the 'runs/train/brand_detector_run1' directory.")
        # The best model is saved as 'best.pt' inside the 'weights' subfolder.

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Please check your 'data.yaml' file paths and ensure your dataset is correctly structured.")

if __name__ == '__main__':
    train_model()
