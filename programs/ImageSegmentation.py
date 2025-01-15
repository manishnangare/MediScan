
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Take the input image path from the user
image_path = input("Please enter the path to your image: ")

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding (128 is the threshold, and 255 is the max value for pixels above the threshold)
    _, segmented_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Display the segmented image using matplotlib
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')  # Hide the axes
    plt.show()
