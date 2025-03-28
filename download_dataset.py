from roboflow import Roboflow

# Replace with your actual API key
rf = Roboflow(api_key="8ZdQRo1ECTSNHpcNPZdD")  

# Load the dataset project
project = rf.workspace().project("pipeline-leak-prediction")  # Use your exact dataset name on Roboflow

# Get the latest version of the dataset
dataset = project.version(1).download("yolov8")  
