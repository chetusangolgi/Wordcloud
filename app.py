# app.py
# To run this:
# 1. Install libraries: pip install Flask Pillow numpy wordcloud opencv-python
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
        numpy.ndarray or None: A mask with the subject in white, or None on failure.
    """
    try:
        # 1. Load Image and Resize
        img = Image.open(io.BytesIO(image_bytes)).convert("L") # Convert to grayscale
        max_dimension = 800
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # 2. Apply Threshold
        # Pixels darker than the threshold become the subject (white in the mask)
        mask = np.where(img_np < threshold, 255, 0).astype(np.uint8)

        # 3. Clean the Mask with Morphological Operations
        # This is the key improvement to create a solid shape.
        
        # 'Opening' removes small noise objects (like salt grains) from the background.
        kernel_small = np.ones((3, 3), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # 'Closing' fills small holes within the main subject (like pepper grains).
        kernel_large = np.ones((10, 10), np.uint8)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        # 4. Final Cleanup: Isolate the Largest Shape
        # This step is now much more reliable because the mask is clean.
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
    # We now only need one mask, which will be used for both generation and display.
    display_mask = create_threshold_mask(image_bytes, threshold)
    if display_mask is None:
        return jsonify({"error": "Could not create a mask from the image with the current threshold."}), 500

    # --- 3. Generate the Word Cloud Image ---
    try:
        if np.sum(display_mask) == 0:
            return jsonify({"error": "The detected subject is too thin to fit words. Try increasing the threshold."}), 400

        # FIX: Removed the cv2.erode step and are now using the built-in contour feature.
        wc = WordCloud(
            background_color="white",
            mode="RGB",
            mask=display_mask, # Use the clean mask directly
            color_func=lambda *args, **kwargs: "black",
            relative_scaling=0.5,
            contour_width=2, # Draw a contour around the shape
            contour_color='black' # Color of the contour
        )
        wc.generate(text)
        word_image = wc.to_image()

    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return jsonify({"error": "Failed to generate word cloud."}), 500

    # --- 4. Final Compositing ---
    # The word cloud image is now the final image, as it includes the contour.
    final_image = word_image.resize(Image.fromarray(display_mask).size, Image.Resampling.LANCZOS)
    
    # --- 5. Send Image Back to Frontend ---
    img_io = io.BytesIO()
    final_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
