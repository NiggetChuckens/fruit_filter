import os
import keras
from img_p import load_images
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dirname = os.path.join(os.getcwd(), r'.\\fruits-360\\Training\\')
imgpath = dirname + os.sep 

def train_model():
    INIT_LR = 1e-3
    epochs = 6
    batch_size = 64

    train_X,train_label,valid_X,valid_label,nClasses = load_images(imgpath)


    print(f"train_X shape: {train_X.shape}")
    print(f"train_label shape: {train_label.shape}")
    print(f"valid_X shape: {valid_X.shape}")
    print(f"valid_label shape: {valid_label.shape}")
    print(f"nClasses: {nClasses}")

    
    fruit_model = Sequential()
    fruit_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))
    fruit_model.add(LeakyReLU(alpha=0.1))
    fruit_model.add(MaxPooling2D((2, 2),padding='same'))
    fruit_model.add(Dropout(0.5))

    fruit_model.add(Flatten())
    fruit_model.add(Dense(32, activation='linear'))
    fruit_model.add(LeakyReLU(alpha=0.1))
    fruit_model.add(Dropout(0.5)) 
    fruit_model.add(Dense(nClasses, activation='softmax'))

    fruit_model.summary()

    fruit_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(decay=INIT_LR / 100),metrics=['accuracy'])
    if len(train_X) != len(train_label):
        raise ValueError("Data arrays must contain the same number of samples")
    if len(valid_X) != len(valid_label):
        raise ValueError("Data arrays must contain the same number of samples")
    
    fruit_train_dropout = fruit_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,  validation_data=(valid_X, valid_label))
    fruit_model.save("fruits.h5")
    
    # guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
    return fruit_train_dropout
    # en cnn.py en la parte del fit, en el validation data dice que no son iguales los arrays ns si podi runearlo en esta wea, se demora como 5 mi
    #donde se rompe el código
    #demora caleta
    # ni tanto, antes se demoraba ma