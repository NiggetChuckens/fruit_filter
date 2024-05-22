import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def load_model():
    """
    Loads a Keras model from the specified file "fruits.h5".

    Returns:
        tf.keras.Model: The loaded Keras model.
    """

    return tf.keras.models.load_model("fruits.h5")

def prediction(model, image_set):
    """
    Generates predictions using the specified model on the given image set and displays the top predicted classes.

    Args:
        model: The model used for making predictions.
        image_set: The set of images on which predictions are to be made.

    Returns:
        None
    """

    probs = model.predict(image_set)
    clase = np.argmax(probs, -1)   

    for i in range(10):
        plt.imshow(image_set[i]/255.)
        plt.axis('off')
        plt.show()
        print("Predicción:", "perro" if clase[i] else "gato")

def validate_model(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return train_datagen.flow_from_directory(
        data_dir,
        target_size=(21, 28),  # Match the expected input shape of your model
        batch_size=32,
        class_mode='categorical',
    )

def create_train_generator(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return train_datagen.flow_from_directory(
        data_dir,
        target_size=(21, 28),  # Match the expected input shape of your model
        batch_size=32,
        class_mode='categorical',
    )

if __name__ == "__main__":
    model = load_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_dir = './fruits-360/Training/'
    val_dir = './fruits-360/Test/'
    train_generator = create_train_generator(train_dir)
    history = model.fit(train_generator, epochs=10, validation_data=validate_model(val_dir))
    
    model.save("fruits_trained.h5")
    
    # Assuming 'history' is the object returned by model.fit()
    history_dict = history.history
    
    # Plotting the training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'])
    plt.plot(history_dict['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    # Plotting the training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    
    plt.tight_layout()
    plt.show()