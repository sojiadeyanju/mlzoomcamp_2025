import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# 1. Load the Model
# It's in the current working directory of the container
model_path = "/var/task/hair_classifier_empty.onnx"
session = ort.InferenceSession(model_path)

# Get input name (usually 'input.1' or similar)
input_name = session.get_inputs()[0].name

# 2. Load and Preprocess Image
# We read from the mounted volume '/input_data'
image_path = "/input_data/yf_dokzqy3vcritme8ggnzqlvwa.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

img = Image.open(image_path).convert('RGB')

# RESIZE to 200x200
img = img.resize((200, 200), resample=Image.BILINEAR)

# Normalize and Transpose
# Convert to Numpy, Scale to 0-1
img_np = np.array(img, dtype=np.float32) / 255.0

# Transpose to Channels First (C, H, W)
img_chw = np.transpose(img_np, (2, 0, 1))

# Normalize (ImageNet stats)
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_normalized = (img_chw - mean) / std

# Add Batch Dimension (1, 3, 200, 200)
input_tensor = np.expand_dims(img_normalized, axis=0)

# 3. Run Prediction
outputs = session.run(None, {input_name: input_tensor})
probability = outputs[0][0][0] # Adjust index based on specific model output shape

print(f"Prediction Probability: {probability:.4f}")
print("Result: Straight Hair" if probability > 0.5 else "Result: Curly Hair")