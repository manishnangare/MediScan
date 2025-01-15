import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load the VGG16 model pre-trained on ImageNet without the top layers (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Create a new model that outputs the features from the 'block5_pool' layer
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Take the input image path from the user
image_path = input("Please enter the path to your image: ")

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Resize the image to match the input size required by VGG16 (256x256)
    resized_image = cv2.resize(image, (256, 256))

    # Normalize the image to the range expected by VGG16 (e.g., pixel values between -1 and 1)
    normalized_image = preprocess_input(resized_image.astype(np.float32))

    # Add a batch dimension since the model expects input shape to be (batch_size, height, width, channels)
    image_batch = np.expand_dims(normalized_image, axis=0)

    # Extract features using the feature extractor model
    features = feature_extractor.predict(image_batch)

    # Output the shape of the extracted features
    print("Extracted features shape:", features.shape)

    # Displaying the features
    # The features will be of shape (1, 8, 8, 512). We'll visualize a few of these feature maps.
    feature_map_count = features.shape[-1]  # Number of feature maps (512 in block5_pool)
    
    # Display some feature maps
    num_feature_maps_to_display = 6  # Number of feature maps to display (you can adjust this number)

    plt.figure(figsize=(15, 15))
    for i in range(num_feature_maps_to_display):
        plt.subplot(1, num_feature_maps_to_display, i + 1)
        plt.imshow(features[0, :, :, i], cmap='viridis')  # Extract and display the ith feature map
        plt.title(f"Feature Map {i+1}")
        plt.axis('off')

    plt.show()
