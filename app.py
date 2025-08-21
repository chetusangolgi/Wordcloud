# app.py
# To run this:
# 1. Install libraries: pip install Flask Pillow numpy wordcloud opencv-python flask-cors
# 2. Run from terminal: python -m flask run

import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from wordcloud import WordCloud
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow the frontend to call the backend
CORS(app)

def create_threshold_mask(image_bytes, threshold):
    """
    Creates an extremely detailed mask by using high resolution and minimal processing.
    """
    try:
        # 1. Load Image at VERY HIGH RESOLUTION
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        max_dimension = 2000
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # 2. Apply Threshold to get the raw shape
        mask = np.where(img_np < threshold, 255, 0).astype(np.uint8)

        # 3. Apply STRATEGIC MICRO-CLEANING
        # Removes isolated single-pixel noise without affecting real details.
        kernel = np.ones((2, 2), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned_mask

    except Exception as e:
        print(f"Error creating threshold mask: {e}")
        return None

@app.route('/generate', methods=['POST'])
def generate_word_cloud():
    """ API endpoint to generate the word cloud. """
    data = request.json
    if not all(k in data for k in ['image', 'text', 'threshold']):
        return jsonify({"error": "Missing required data: image, text, and threshold"}), 400

    image_data = data['image']
    text = data['text']
    threshold = int(data['threshold'])

    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    # --- THIS BLOCK WAS MISSING AND IS NOW RESTORED ---
    # It decodes the base64 image data sent from the frontend.
    try:
        image_header, image_encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(image_encoded)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400
    # ---------------------------------------------------

    # --- 2. Create Extremely Detailed Mask of the Subject ---
    # This line will now work because 'image_bytes' is defined.
    detailed_subject_mask = create_threshold_mask(image_bytes, threshold)
    if detailed_subject_mask is None:
        return jsonify({"error": "Could not create a mask from the image with the current threshold."}), 500
    
    # Invert the mask to draw words on the background
    target_mask = cv2.bitwise_not(detailed_subject_mask)

    if int(np.sum(target_mask)) == 0:
        return jsonify({"error": "No drawable background area found. Try raising the threshold."}), 400

    # --- 3. Generate the Word Cloud Image with DENSE FILL settings ---
    try:
        wc = WordCloud(
            background_color="white",
            mode="RGBA",
            mask=target_mask,
            color_func=lambda *args, **kwargs: "black",
            repeat=True,
            max_words=2000,
            relative_scaling=0.1,
            font_step=1,
        )
        wc.generate(text)
        word_image = wc.to_image()

    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return jsonify({"error": "Failed to generate word cloud."}), 500


    # --- 4. Send Image Back to Frontend ---
    img_io = io.BytesIO()
    word_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)