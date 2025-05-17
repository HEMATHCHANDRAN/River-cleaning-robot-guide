from flask import Flask, request, jsonify
import os
from PIL import Image
import io
import torch
from torchvision import models, transforms

app = Flask(__name__)

# 📦 Load pre-trained model (can replace with your custom model)
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 🚀 Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🚮 Define labels manually or use custom labels
labels = ["plastic", "paper", "leaf", "metal", "glass"]

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Simulate mapping to your label list (dummy map)
    result = labels[predicted.item() % len(labels)]

    return jsonify({"waste_type": result})
