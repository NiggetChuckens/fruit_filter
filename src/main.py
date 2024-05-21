import tensorflow as tf, os, cv2, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# Variable declaration
directory = r'./fruits-360_dataset/fruits-360/Training/'
test_directory = r'./fruits-360_dataset/fruits-360/test-multiple_fruits/'
num_classes = sum(1 for _ in os.walk(directory))
batch_size = 32

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

def calculate_epochs(training_directory, batch_size):
    return sum(len(files) for _, _, files in os.walk(training_directory))

def load_images_from_dir(directory):
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            print(file)
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                label = root.split('/')[-1]  # Extract label from directory name
                
                # Load and preprocess the image
                image = cv2.imread(image_path)
                image = cv2.resize(image, (100, 100))  # Resize image
                image = image / 255.0  # Normalize pixel values
                
                images.append(image)
                labels.append(label)
    
    return np.array(images), np.array(labels)


num_epochs = calculate_epochs(directory, batch_size)
print(f"Number of epochs: {num_epochs}")
train_images, train_labels = load_images_from_dir(directory)
val_images, val_labels = load_images_from_dir(test_directory)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels_encoded, epochs=num_epochs, validation_data=(val_images, val_labels_encoded))
print(history.history)
# Plot the accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()