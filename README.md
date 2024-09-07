# Real-time Object Detection with H100 GPU

This project demonstrates real-time object detection using a webcam feed processed by an NVIDIA H100 GPU. It uses a Flask server to handle the backend processing and a web interface for user interaction.

## Description

The system captures video frames from the user's webcam in the browser, sends them to a Flask server running on an H100 GPU, processes the frames using a pre-trained Faster R-CNN model for object detection, and sends the results back to the browser. The processed frames are displayed with bounding boxes around detected objects, along with real-time performance metrics like frames per second and inference time.

## Prerequisites

- Python 3.8+
- NVIDIA H100 GPU
- CUDA 11.8+
- Web browser with webcam access

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/people-detection-h100.git
cd people-detection-h100
```

2. Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```
pip install flask flask-cors pillow torch torchvision numpy opencv-python
```

## Usage

Start the Flask server:

```
python3 app.py
```

Note that camera access requires HTTPS. Use a tool like ngrok to proxy the calls if you don't otherwise have a certificate.

Select your camera from the dropdown menu and click "Start Capture" to begin the object detection process.

## Files

`app.py`: Flask server setup and main application logic
`model.py`: Object detection model loading and processing functions
`templates/index.html`: Web interface for the application
