import os
import re
import cv2
import numpy as np
import numpy as np
import tensorflow as tf
from fruit_id import fruit_id
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train import create_train_generator, validate_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cnn_model():
    return tf.keras.models.load_model("fruits_trained.h5").compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_yolo_model():
    return tf.keras.models.load_model("yolov7.p7")


def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)
        else:
            print(f"Error loading image at path: {image_path}")
    
    return images

def detect_objects(yolo_model, image):
    # Perform object detection using the YOLO model
    detections = yolo_model.detect_objects(image)
    
    # Process YOLO detections and extract relevant information
    detected_objects = []
    for detection in detections:
        class_label = detection.class_label
        confidence = detection.confidence
        bbox = detection.bbox
        detected_objects.append({'class_label': class_label, 'confidence': confidence, 'bbox': bbox})
    
    return detected_objects

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

    # Load an image
    # Iterate over the loaded images for YOLO detection and CNN classification
    images_paths = [os.path.join(os.getcwd(), path) for path in os.listdir("./fruits-360/test-multiple_fruits") if re.match(r".*\.(jpg|jpeg|png)", path)]
    images = load_images(images_paths)
    for image in images:
        # Perform object detection using YOLO
        detections = detect_objects(yolo_model, image)
        
        # Process each detection and classification
        for detection in detections:
            fruit_region = crop_fruit_region(image, detection.bbox)
            predicted_class = cnn_model.predict(fruit_region)
            
            # Display results for each image and detection
            display_results(image, detection.bbox, detection.class_label, predicted_class)

    # Perform object detection using YOLO
    detections = detect_objects(yolo_model, image)

    # Process each detection
    for detection in detections:
        fruit_region = crop_fruit_region(image, detection.bbox)
        
        # Perform fruit classification using your CNN model
        predicted_class = cnn_model.predict(fruit_region)
        
        # Combine detection and classification results
        display_results(image, detection.bbox, detection.class_label, predicted_class)