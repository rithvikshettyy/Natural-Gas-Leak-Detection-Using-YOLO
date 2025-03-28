import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

# Force CUDA device selection for GTX 1650
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU (GTX 1650)

# Check CUDA availability and print detailed information
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    # Get all available CUDA devices
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Print information about all available devices
    for i in range(device_count):
        print(f"\nDevice {i}:")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Properties: {torch.cuda.get_device_properties(i)}")
    
    # Set CUDA device explicitly to GTX 1650
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    
    # Enable CUDA optimizations for GTX 1650
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
    torch.backends.cudnn.allow_tf32 = True
    
    # Print current device information
    print(f"\nUsing CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

print(f"Using device: {device}")

# Verify data.yaml path exists
data_yaml_path = "pipeline-leak-prediction/data.yaml"
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Please ensure the path is correct.")

try:
    # Initialize and train the model
    model = YOLO("yolov8s.pt")
    print("Starting training...")
    
    # Force model to use GPU and verify device placement
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Configure training parameters optimized for GTX 1650
    results = model.train(
        data=data_yaml_path,
        epochs=5,
        batch=4,
        imgsz=640,
        device=0,  # Use first GPU (GTX 1650)
        workers=2,  # Reduce workers to avoid potential issues
        amp=True,   # Enable automatic mixed precision
        half=True,  # Use half precision for better GPU memory usage
        cache=True, # Cache images in GPU memory
        rect=False, # Disable rectangular training
        cos_lr=True, # Use cosine learning rate
        optimizer="AdamW", # Use AdamW optimizer
        verbose=True, # Show detailed training information
        deterministic=True, # Ensure deterministic results
        seed=42, # Set random seed
        patience=50, # Early stopping patience
        save=True, # Save best model
        save_period=1, # Save every epoch
        project="runs/train", # Save results to runs/train
        name="exp", # Experiment name
        exist_ok=True, # Overwrite existing experiment
        pretrained=True, # Use pretrained weights
        resume=False, # Don't resume from checkpoint
        close_mosaic=10, # Disable mosaic augmentation in last 10 epochs
        overlap_mask=True, # Allow mask overlap
        mask_ratio=4, # Mask ratio for segmentation
        dropout=0.0, # No dropout
        val=True, # Validate during training
        split="val", # Use validation split
        save_json=False, # Don't save predictions as JSON
        save_hybrid=False, # Don't save hybrid labels
        conf=0.001, # Confidence threshold
        iou=0.6, # IoU threshold
        max_det=300, # Maximum number of detections
        dnn=False, # Don't use OpenCV DNN
        plots=True, # Generate plots
        source=None, # No source for inference
        vid_stride=1, # Video stride
        stream_buffer=False, # Don't buffer video stream
        visualize=False, # Don't visualize predictions
        augment=False, # Don't augment validation
        agnostic_nms=False, # Don't use agnostic NMS
        classes=None, # Use all classes
        retina_masks=False, # Don't use retina masks
        embed=None, # Don't use embeddings
        show=False, # Don't show predictions
        save_frames=False, # Don't save frames
        save_txt=False, # Don't save labels as txt
        save_conf=False, # Don't save confidence scores
        save_crop=False, # Don't save cropped predictions
        show_labels=True, # Show labels in plots
        show_conf=True, # Show confidence in plots
        show_boxes=True, # Show boxes in plots
        line_width=2, # Default line width
        format="torchscript", # Save model in TorchScript format
        keras=False, # Don't save as Keras model
        optimize=False, # Don't optimize model
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


