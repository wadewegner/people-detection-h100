import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response
from model import load_model, process_batch
from PIL import Image
import time

app = Flask(__name__)

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

BATCH_SIZE = 4  # Adjust based on your GPU memory

def gen_frames():
    video = cv2.VideoCapture('face-demographics-walking.mp4')
    batch = []
    total_frames = 0
    total_time = 0
    
    while True:
        batch.clear()
        for _ in range(BATCH_SIZE):
            success, frame = video.read()
            if not success:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = video.read()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            batch.append(pil_image)
        
        try:
            results, inference_time = process_batch(model, batch)
            total_frames += BATCH_SIZE
            total_time += inference_time
            avg_fps = total_frames / total_time if total_time > 0 else 0
            
            for result in results:
                result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                cv2.putText(result, f"Avg FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', result)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return original frames if processing fails
            for pil_image in batch:
                result_np = np.array(pil_image)
                ret, buffer = cv2.imencode('.jpg', result_np[:, :, ::-1])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
