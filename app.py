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
    Creates a clean mask based on a brightness threshold using a robust cleaning process.

    Args:
        image_bytes (bytes): The raw bytes of the uploaded image.
        threshold (int): The brightness threshold (0-255).

    Returns:
        numpy.ndarray or None: A mask with the subject in white (255), or None on failure.
    """
    try:
        # 1. Load Image and Resize
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        max_dimension = 800
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # 2. Apply Threshold
        # Pixels darker than the threshold become the subject (white in the mask)
        mask = np.where(img_np < threshold, 255, 0).astype(np.uint8)

        # 3. Clean the Mask with Morphological Operations
        kernel_small = np.ones((3, 3), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

        kernel_large = np.ones((10, 10), np.uint8)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        # 4. Final Cleanup: Isolate the Largest Shape
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(closed_mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

        return clean_mask

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

    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    try:
        image_header, image_encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(image_encoded)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    # --- 2. Create Mask ---
    # display_mask: subject = white (255), background = black (0)
    display_mask = create_threshold_mask(image_bytes, threshold)
    if display_mask is None:
        return jsonify({"error": "Could not create a mask from the image with the current threshold."}), 500

    # >>> KEY CHANGE <<<
    # We want words on the "black" part of the original picture.
    # Since WordCloud places words where mask > 0 (white), invert the mask.
    target_mask = cv2.bitwise_not(display_mask)  # now black region becomes white for WordCloud

    if int(np.sum(target_mask) == 0):
        return jsonify({"error": "No drawable area found on the black region. Try lowering the threshold."}), 400

    # --- 3. Generate the Word Cloud Image ---
    # --- 3. Generate the Word Cloud Image ---
    try:
        wc = WordCloud(
            background_color="white",
            mode="RGB",
            mask=target_mask,                  # Use the INVERTED mask
            color_func=lambda *args, **kwargs: "black",
            relative_scaling=0.5,
            # Removed contour_width and contour_color
        )
        wc.generate(text)
        word_image = wc.to_image()
    
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return jsonify({"error": "Failed to generate word cloud."}), 500


    # --- 4. Final Compositing ---
    final_image = word_image.resize(Image.fromarray(target_mask).size, Image.Resampling.LANCZOS)

    # --- 5. Send Image Back to Frontend ---
    img_io = io.BytesIO()
    final_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
