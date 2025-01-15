import os
import cv2
import numpy as np

# Preprocessing function
def preprocess_image(img):
    # Resize to a fixed size (e.g., 224x224)
    img_resized = cv2.resize(img, (224, 224))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur for noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalize the image pixel values
    norm_img = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Enhance contrast using histogram equalization
    enhanced_img = cv2.equalizeHist(norm_img)

    # Apply random rotation for augmentation
    rows, cols = enhanced_img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle=np.random.uniform(-10, 10), scale=1)
    rotated_img = cv2.warpAffine(enhanced_img, rotation_matrix, (cols, rows))

    #image segmentation
    

    return rotated_img

# Function to read images from subfolders, preprocess, and overwrite them
def process_images_in_place(input_dir):
    # Traverse the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for valid image extensions
                img_path = os.path.join(root, file)
                
                # Read the image
                img = cv2.imread(img_path)

                if img is not None:
                    # Preprocess the image
                    processed_img = preprocess_image(img)

                    # Overwrite the original image with the processed one
                    cv2.imwrite(img_path, processed_img)

                    print(f"Processed and replaced: {img_path}")

# Accept input path from user
input_dir = input("Enter the path to the image dataset: ")

# Check if the provided directory exists
if os.path.isdir(input_dir):
    # Process the images in place
    process_images_in_place(input_dir)
else:
    print(f"The directory {input_dir} does not exist. Please provide a valid path.")
