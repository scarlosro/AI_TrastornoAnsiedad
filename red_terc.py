# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import glob
import h5py

direc = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote*')


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
#write_h5file(25,'train')

#write_h5file(5,'test')



# train_x, train_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_1/h5.h5')
train_x, train_y = load_Dataset_from_h5file('./train.h5')
print(train_x[0].shape)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
#test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
test_x, test_y = load_Dataset_from_h5file('./test.h5')


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1792,1)),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_x, train_y, epochs=100)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
