import os
import re
import cv2
import torch
import yolov7
import numpy as np
import numpy as np
import tensorflow as tf 
from fruit_id import fruit_id
import matplotlib.pyplot as plt
from train import create_train_generator, validate_model
import matplotlib.patches as patches
from train import create_train_generator, validate_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cnn_model():
    return tf.keras.models.load_model('fruits_trained.h5')

def load_yolo_model():
    return yolov7.load('yolov7.pt')


def load_image(image_path):
    # Read the image from the specified image path
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Error loading image at path: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop_fruit_region(image, bbox):
    # Extract the bounding box coordinates
    x, y, w, h = bbox

    return image[y:y+h, x:x+w]



def display_results(image, bbox, detection_class, classification_result):
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(image)
    
    # Create a Rectangle patch for the bounding box
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    
    # Add the bounding box to the plot
    ax.add_patch(rect)
    
    # Display the detection class and classification result as text
    plt.text(x, y-10, f'Detection Class: {detection_class}', color='red')
    plt.text(x, y-30, f'Classification Result: {classification_result}', color='blue')
    
    # Set axis off and show the plot
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    # Load YOLO and CNN models
    yolo_model = load_yolo_model()
    cnn_model = load_cnn_model()
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_dir = './fruits-360/Training/'
    val_dir = './fruits-360/Test/'
    train_generator = create_train_generator(train_dir)
    cnn_model.fit(train_generator, epochs=10, validation_data=validate_model(val_dir))

    yolo_model.conf = 0.25
    yolo_model.iou = 0.45
    yolo_model.classes = 80

    image = load_image('./fruits-360/test-multiple_fruits/apple.jpg')
    # Perform object detection using YOLO
    detections = yolo_model(image)
    
    predictions = detections.pred[0]
    boxes = predictions[:, 4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]
    
    # Filter out low confidence detections
    detections.show()