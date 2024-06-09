"""
I denne fil træner vi modellen med input fra config fil
Input er 100 annoterede herbarium ark
"""

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=100)  # train the model