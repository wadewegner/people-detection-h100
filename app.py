import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import load_model, process_batch
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))

print("Starting model loading process...")
model = load_model()
if model is None:
    print("WARNING: Model failed to load. The demo will run without processing.")
else:
    print("Model loaded successfully in app.py")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(frame_bytes))
    
    # Process the frame
    processed_images, inference_time = process_batch(model, [img])
    
    # Convert back to base64
    buffered = io.BytesIO()
    processed_images[0].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'processed_frame': f'data:image/jpeg;base64,{img_str}',
        'inference_time': inference_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
