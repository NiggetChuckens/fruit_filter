import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def iterate_folders(imgpath):
    folders = []
    for root, dirs, files in os.walk(imgpath):
        for folder in dirs:
            folders.append(os.path.join(root, folder))
    return folders
            
def load_images(imgpath):
    
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    
    folders = iterate_folders(imgpath)
    for folder in folders:
        print(folder)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$",file):
                    cant += 1
                    filepath = os.path.join(root, file)
                    image = plt.imread(filepath)
                    images.append(image)
                    b = "Leyendo..." + str(cant)
                    print(b, end="\r")
                    if prevRoot != root:
                        print(root, cant)
                        prevRoot = root
                        directories.append(root)
                        dircount.append(cant)
                        cant = 0
    dircount.append(cant)

    dircount = dircount[1:]
    if dircount:  # Check if dircount is not empty before accessing elements
        dircount[0] = dircount[0] + 1
    print('Directorios leidos:', len(directories))
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:', sum(dircount))
    return set_index(images, directories, dircount)
                
def set_index(images, directories, dircount):
    labels=[]
    indice=0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice=indice+1
    print("Cantidad etiquetas creadas: ",len(labels))
    
    fruits=[]
    indice=0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice , name[len(name)-1])
        fruits.append(name[len(name)-1])
        indice=indice+1
    
    y = np.array(labels)
    X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
    
    # Find the unique numbers from the train labels
    classes = np.unique(y)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)
    

#Mezclar todo y crear los grupos de entrenamiento y testing
    train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
    print('Training data shape : ', train_X.shape, train_Y.shape)
    print('Testing data shape : ', test_X.shape, test_Y.shape)
    
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.
    
    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)
    
    # Display the change for category label using one-hot encoding
    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])
    
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    
    print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
    
    return train_X,valid_X,train_label,valid_label,nClasses
                    
if __name__ == '__main__':
    dirname = os.path.join(os.getcwd(), 'fruits-360')
    dirname = os.path.join(dirname, 'Training')
    imgpath = dirname + os.sep
    a=load_images(imgpath)
    print(a)