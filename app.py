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
    Creates the most accurate mask possible with zero cleaning.
    """
    try:
        # 1. Load Image at MAXIMUM practical RESOLUTION
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        # MODIFIED: Pushed to a very high resolution for final accuracy.
        max_dimension = 2500
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # 2. Apply Threshold to get the raw shape
        mask = np.where(img_np < threshold, 255, 0).astype(np.uint8)

        # 3. NO CLEANING
        # For absolute, raw accuracy, all cleaning operations are removed.
        # The mask is a 1:1 representation of the thresholded pixels.
        return mask

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

    try:
        image_header, image_encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(image_encoded)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    detailed_subject_mask = create_threshold_mask(image_bytes, threshold)
    if detailed_subject_mask is None:
        return jsonify({"error": "Could not create a mask from the image with the current threshold."}), 500
    
    target_mask = cv2.bitwise_not(detailed_subject_mask)

    if int(np.sum(target_mask)) == 0:
        return jsonify({"error": "No drawable background area found. Try raising the threshold."}), 400

    # --- Generate Word Cloud with ABSOLUTE MAX DENSITY settings ---
    try:
        wc = WordCloud(
            background_color="white",
            mode="RGBA",
            mask=target_mask,
            color_func=lambda *args, **kwargs: "black",
            repeat=True,
            # MODIFIED: Increased word limit for denser packing.
            max_words=4000,
            relative_scaling=0.1,
            font_step=1,
            # MODIFIED: Allow the use of extremely small fonts to fill tiny gaps.
            min_font_size=2,
        )
        wc.generate(text)
        word_image = wc.to_image()

    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return jsonify({"error": "Failed to generate word cloud."}), 500

    img_io = io.BytesIO()
    word_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)