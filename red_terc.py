# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

# Librerias de ayuda
import numpy as np
import glob
import h5py
from scipy.io import loadmat

direc = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote*')


def write_h5file(carpeta,nameF):
    
    files = glob.glob(carpeta + '/*.mat')

    # Train, Valid, Test sets
    setX = []
    setY = []


    #print(' La carpeta ' +  carpeta + ' contiene ', len(files))
    for file in files:
        arc_mat = loadmat(file, struct_as_record=False)
        
        # Split the song data and labels
        data = arc_mat['data'].flatten()
        label = arc_mat['label']


        #print(data)
        #print(label)


        # Generate train, valid and test set
        setX.append(data)
        setY.append(label)


    # Stack the data vertically
    #setX = np.vstack(setX)
    #setY = np.array(setY)

    print(len(setX))
    print(len(setY))
    # Create the dataset file
    file = h5py.File('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/' + nameF +'.h5', 'w')

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

#print(len(direc))
#write_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/trainA','trainA')

#write_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testA','testA')



# train_x, train_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_1/h5.h5')
train_x, train_y = load_Dataset_from_h5file('./trainA.h5')
print(train_x.shape)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
#test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
test_x, test_y = load_Dataset_from_h5file('./testA.h5')


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1792,1)),
    keras.layers.Dense(1792, activation='relu'),
    keras.layers.Dense(1500, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#plot_model(model, to_file='model2.png')


model.fit(train_x, train_y, epochs=50)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
