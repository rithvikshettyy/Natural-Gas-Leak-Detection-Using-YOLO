import os
import torch
import numpy as np
from ultralytics import YOLO

# Force CUDA device selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check CUDA availability and print detailed information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda:0')
    # Enable CUDA optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
else:
    print("CUDA not available, using CPU")
    device = torch.device('cpu')

print(f"Using device: {device}")

# Define paths with proper formatting
model_path = os.path.join("runs", "train", "exp", "weights", "best.pt")
data_yaml_path = os.path.join("pipeline-leak-prediction", "data.yaml")

# Verify files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the path is correct.")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Please ensure the path is correct.")

try:
    # Load trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Move model to appropriate device
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Run validation to get accurate metrics
    print("\nRunning validation to calculate model performance...")
    metrics = model.val(
        data=data_yaml_path,
        device=device,
        verbose=True,
        save_json=True,  # Save predictions as JSON
        save_hybrid=False,  # Don't save hybrid labels
        conf=0.25,  # Confidence threshold
        iou=0.45,   # IoU threshold
        max_det=300,  # Maximum number of detections
        half=True,  # Use half precision
        dnn=False,  # Don't use OpenCV DNN
        plots=True,  # Generate plots
        save_txt=True,  # Save labels as txt
        save_conf=True,  # Save confidence scores
        save_crop=True,  # Save cropped predictions
        show_labels=True,  # Show labels
        show_conf=True,  # Show confidence
        show_boxes=True,  # Show boxes
        line_width=2,  # Line width for boxes
        source=None,  # No source for inference
        vid_stride=1,  # Video stride
        stream_buffer=False,  # Don't buffer video stream
        visualize=False,  # Don't visualize predictions
        augment=False,  # Don't augment validation
        agnostic_nms=False,  # Don't use agnostic NMS
        classes=None,  # Use all classes
        retina_masks=False,  # Don't use retina masks
        embed=None,  # Don't use embeddings
        save_frames=False,  # Don't save frames
        format="torchscript",  # Save model in TorchScript format
        keras=False,  # Don't save as Keras model
        optimize=False,  # Don't optimize model
    )
    
    # Print detailed metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")  # Mean Average Precision at IoU=0.50
    print(f"mAP50-95: {metrics.box.map:.4f}")  # Mean Average Precision at IoU=0.50:0.95
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    print(f"F1-Score: {metrics.box.f1:.4f}")
    print(f"Confidence: {metrics.box.conf:.4f}")
    print(f"Speed: {metrics.speed['inference']:.2f}ms")
    
    # Print class-wise metrics
    print("\nClass-wise metrics:")
    for i, name in enumerate(metrics.box.names):
        print(f"\nClass {name}:")
        print(f"  Precision: {metrics.box.precision[i]:.4f}")
        print(f"  Recall: {metrics.box.recall[i]:.4f}")
        print(f"  mAP50: {metrics.box.map50[i]:.4f}")
        print(f"  mAP50-95: {metrics.box.map[i]:.4f}")
    
    print("\nValidation completed successfully!")
    print(f"Results saved in: {os.path.join('runs', 'detect', 'val')}")
    
except Exception as e:
    print(f"An error occurred during validation: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
