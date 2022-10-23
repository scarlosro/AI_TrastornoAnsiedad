import glob
from json import load
import random
import shutil
from scipy.io import loadmat
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import h5py
from keras.layers import Reshape

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def write_h5file():

    matfiles = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/*.mat')

    # Train, Valid, Test sets
    setX = []
    setY = []

    # Explore the dataset
    for file in matfiles:
        arc_mat = loadmat(file, struct_as_record=False)
        
        # Split the song data and labels
        data = arc_mat['data']# .flatten()
        label = arc_mat['label']

        print(data)
        print(label)


        # Generate train, valid and test set
        setX.append(data)
        setY.append(label)


    
    # Stack the data vertically
    #setX = np.vstack(setX)
    #setY = np.array(setY)

    print(len(setX))
    print(len(setY))
    # Create the dataset file
    file = h5py.File('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5', 'w')

    # Add the data to the dataset
    file.create_dataset('setX', data=setX)
    file.create_dataset('setY', data=setY)
   

    # Close the file
    file.close()


def load_Dataset_from_h5file(h5file_path):

    # Read the H5File
    h5_file = h5py.File(h5file_path, 'r')

    # Separate the file values into train, valid and test set
    setX = h5_file['setX'][:]
    setY = h5_file['setY'][:]


    # Close the H5File
    h5_file.close()

    # Return the train, valid and test set
    return setX, setY


def normalise(x_mean, x_std, x_data):
    return (x_data - x_mean) / x_std


def red_neuro ():   

    # train_x, train_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_1/h5.h5')
    train_x, train_y = load_Dataset_from_h5file('./lote_1/h5.h5')
    #test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
    test_x, test_y = load_Dataset_from_h5file('./lote_26/h5.h5')



#preguntar lo de las capaz
#preguntar meter por mini batch
    model = models.Sequential()
    model.add(Reshape((1,14,128), input_shape=(14, 128)))
    model.add(layers.Conv2D(2,(1, 2), activation='relu'))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Conv2D(2,(1, 2), activation='relu'))
    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Conv2D(2,(1, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=10, 
                        validation_data=(test_x, test_y))

    test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

    print(test_acc)


#write_h5file()

red_neuro()

'''
dire_trainin_ligeros = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/training/ligera/*.mat')
dire_trainin_normales = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/training/normal/*.mat')
dire_trainin_moderados = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/training/moderada/*.mat')
dire_trainin_severos = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/training/severa/*.mat')

dire_testin_ligeros = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testing/ligera/*.mat')
dire_testin_normales = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testing/normal/*.mat')
dire_testin_moderados = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testing/moderada/*.mat')
dire_testin_severos = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testing/severa/*.mat')

def crearLote(lista,no_lote):
    path = os.getcwd()
    path_create = path + '/lote_' + str(no_lote)
    try:
        os.mkdir(path_create)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    for i in range(5):
        no_arch = random.randint(0,len(lista)-1)
        archivo = lista[no_arch]
        shutil.copy2(archivo,path_create, follow_symlinks=True)
        lista.remove(archivo)

for no_lote in range(25):
    crearLote(dire_trainin_ligeros,no_lote)
    crearLote(dire_trainin_normales,no_lote)
    crearLote(dire_trainin_moderados,no_lote)
    crearLote(dire_trainin_severos,no_lote)

for no_lote in range(5):
    crearLote(dire_testin_ligeros,no_lote + 26)
    crearLote(dire_testin_normales,no_lote + 26)
    crearLote(dire_testin_moderados,no_lote + 26)
    crearLote(dire_testin_severos,no_lote + 26)





'''

'''
    for archivo in list_archivos:
        arc_mat = loadmat(archivo, struct_as_record=False)
        eeg = arc_mat['data']
        eeg = eeg.flatten()
        eegs.append(eeg)
    print(eegs)
    return eegs

ligeros_eeg = obtenerEEGs(dire_ligeros)
normales_eeg = obtenerEEGs(dire_normales)
moderados_eeg = obtenerEEGs(dire_moderados)
severos_eeg = obtenerEEGs(dire_severos)

SAMPLES = 600

def separar_lista(l: list) -> np.array:
    tamañoDiv = SAMPLES

#print(ligeros_eeg)
'''