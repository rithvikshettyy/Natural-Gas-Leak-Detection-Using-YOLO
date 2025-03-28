import os
import torch
from ultralytics import YOLO

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda:0')
else:
    print("CUDA not available, using CPU")
    device = torch.device('cpu')

# Define model path with proper path formatting
model_path = os.path.join("runs", "train", "exp", "weights", "best.pt")

# Verify model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the path is correct.")

try:
    # Load trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Move model to appropriate device
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Evaluate the model
    print("Starting evaluation...")
    metrics = model.val()
    print("\nEvaluation Results:")
    print(metrics)
    
except Exception as e:
    print(f"An error occurred during evaluation: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")

