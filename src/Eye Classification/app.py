import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (update the path to your model)
model = load_model('model/model.keras')

# Image preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path)  # Open image using PIL
    img = img.resize((256, 256))  # Resize image to the size expected by the model
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize the image to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the route for the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image to a temporary directory
    image_path = os.path.join('static/uploads', file.filename)
    file.save(image_path)

    # Preprocess the image
    img = preprocess_image(image_path)

    # Perform inference (prediction) using the trained model
    prediction = model.predict(img)

    # Map the prediction to a disease category (adjust according to your model's output)
    classes = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataracts']  # Modify according to your model
    predicted_class = classes[np.argmax(prediction)]

    # Return the prediction result as a JSON response
    return jsonify({
        'prediction': predicted_class,
        'confidence': str(np.max(prediction))  # Confidence score
    })

# Route to the main page (UI)
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    # Run the Flask app
    app.run(debug=True)
