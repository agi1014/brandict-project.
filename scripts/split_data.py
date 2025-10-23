import os
import random
import shutil

# --- Configuration ---
# The directory where you saved all your images and .txt annotation files
RAW_DATA_DIR = 'data/raw images'
# The ratio of data to use for training (e.g., 0.8 means 80% train, 20% validation)
TRAIN_RATIO = 0.8

# --- Define output paths ---
images_train_path = 'data/images/train'
images_val_path = 'data/images/val'
labels_train_path = 'data/labels/train'
labels_val_path = 'data/labels/val'

def split_dataset():
    """
    Splits the raw annotated data into training and validation sets.
    """
    print("--- Starting Dataset Split ---")

    # --- Create directories if they don't exist ---
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)

    # --- Get all unique image file names from the raw data directory ---
    all_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_files) # Shuffle for a random split

    if not all_files:
        print(f"Error: No image files found in '{RAW_DATA_DIR}'. Please check the path.")
        return

    # --- Calculate the split point ---
    split_point = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split_point]
    val_files = all_files[split_point:]

    # --- Function to move image and its corresponding label file ---
    def move_files(file_list, img_dest_folder, lbl_dest_folder):
        moved_count = 0
        for filename in file_list:
            basename, _ = os.path.splitext(filename)
            
            # Define source paths for the image and its label
            img_src_path = os.path.join(RAW_DATA_DIR, filename)
            lbl_src_path = os.path.join(RAW_DATA_DIR, basename + '.txt')

            # Check if both image and label exist before moving
            if os.path.exists(img_src_path) and os.path.exists(lbl_src_path):
                shutil.move(img_src_path, os.path.join(img_dest_folder, filename))
                shutil.move(lbl_src_path, os.path.join(lbl_dest_folder, basename + '.txt'))
                moved_count += 1
        return moved_count

    # --- Move the files to their new homes ---
    print(f"Moving {len(train_files)} files to the training set...")
    train_moved = move_files(train_files, images_train_path, labels_train_path)
    print(f"Successfully moved {train_moved} training pairs.")

    print(f"Moving {len(val_files)} files to the validation set...")
    val_moved = move_files(val_files, images_val_path, labels_val_path)
    print(f"Successfully moved {val_moved} validation pairs.")

    print("\n--- Data Splitting Complete! ---")

if __name__ == '__main__':
    split_dataset()
