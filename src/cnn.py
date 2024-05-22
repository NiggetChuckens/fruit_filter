import os
import keras
from img_p import load_images
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model


dirname = os.path.join(os.getcwd(), r'.\\fruits-360\\Training\\')
imgpath = dirname + os.sep 


def train_model():
    INIT_LR = 1e-3
    epochs = 6
    batch_size = 64

    train_X, train_label, valid_X, valid_label, nClasses = load_images(imgpath)

    # Define the input shape using an Input layer
    inputs = Input(shape=(21, 28, 3))

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)  # Flatten the output of the convolutional layers
    x = Dense(128, activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(nClasses, activation='softmax')(x)

    # Create the model
    fruit_model = Model(inputs=inputs, outputs=outputs)

    fruit_model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    fruit_model.save("fruits.h5")

    fruit_train_dropout = fruit_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

    return fruit_train_dropout