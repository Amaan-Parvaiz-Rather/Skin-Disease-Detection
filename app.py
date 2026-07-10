import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import time

# Import our custom modules
from predict import run_prediction
from gemini_api import get_disease_info

# ============================================================================
# FLASK SETUP
# ============================================================================
app = Flask(__name__)

# Configure upload and static folders
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'heatmaps'), exist_ok=True)

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def home():
    """Renders the premium landing page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, runs inference, gets API info, and returns JSON.
    We return JSON so the frontend JavaScript can handle the skeleton loading
    and smooth transition to results without a page reload.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        # Save the uploaded file safely
        filename = secure_filename(file.filename)
        # Add timestamp to prevent caching issues if same filename uploaded twice
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Run AI Inference (Prediction, Grad-CAM, Severity)
            prediction_results = run_prediction(filepath)
            
            # 2. Get Disease Info from Gemini API (skip for healthy results)
            if prediction_results.get('healthy'):
                disease_info = {
                    "overview": "No significant skin condition was detected in this image. Your skin appears to be in good health.",
                    "symptoms": "No concerning symptoms detected.",
                    "precautions": [
                        "Maintain a regular skincare routine.",
                        "Apply sunscreen daily to protect against UV damage.",
                        "Stay hydrated and eat a balanced diet.",
                        "If you notice any changes in your skin, consult a dermatologist."
                    ],
                    "when_to_see_doctor": "While no disease was detected, always consult a dermatologist if you notice persistent redness, unusual spots, itching, or any other changes in your skin."
                }
            else:
                disease_info = get_disease_info(prediction_results['disease'])
            
            # 3. Combine results
            response_data = {
                'status': 'success',
                'original_image': filepath.replace('\\', '/'),
                'heatmap_image': prediction_results['heatmap_path'],
                'prediction': prediction_results,
                'info': disease_info
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
