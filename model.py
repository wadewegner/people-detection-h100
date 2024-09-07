import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))

def load_model():
    try:
        print("Attempting to load Faster R-CNN model...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully and moved to {device}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def process_batch(model, images, confidence_threshold=0.5):
    if model is None:
        print("Model not loaded. Cannot process batch.")
        return images, 0

    device = next(model.parameters()).device
    transform = transforms.Compose([transforms.ToTensor()])
    
    batch = [transform(img).to(device) for img in images]
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(batch)
    inference_time = time.time() - start_time
    
    processed_images = []
    for i, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        boxes = outputs[i]['boxes'].cpu().numpy()
        scores = outputs[i]['scores'].cpu().numpy()
        labels = outputs[i]['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > confidence_threshold and label == 1:  
                x1, y1, x2, y2 = box.astype(int)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"Person {score:.2f}", fill="red")
        
        processed_images.append(img)
    
    return processed_images, inference_time
