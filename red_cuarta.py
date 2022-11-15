from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
from keras.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model
import random


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
#print(train_x[0].shape)
#test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
test_x, test_y = load_Dataset_from_h5file('./test.h5')

#print(train_x.shape)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
#print(train_x.shape)


model = Sequential()
#model.add(Conv1D(1000, 2, activation="relu", input_shape=(1792,1)))
model.add(Conv1D(600, 2, activation="relu", input_shape=(1792,1)))
model.add(Dense(500, activation="relu"))
model.add(MaxPooling1D())
model.add(Dense(50, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(4, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()
#model.fit(train_x, train_y, batch_size=15,epochs=100)
model.fit(train_x, train_y,epochs=3)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)
#
plot_model(model, to_file='model1b.png',show_shapes=True,
    show_dtype=True,
    show_layer_names=True,)



test_y = np.squeeze(np.asarray(test_y))

#print(len(train_x))

y_prediction = model.predict(test_x)
#y_prediction = y_prediction.flatten()

#y_prediction = np.where(y_prediction > 0.5, 1, 0)
n=random.randint(0,len(test_x))
#print(y_prediction[n])
#print(test_y[n])
pred = np.argmax(y_prediction[n])
y_true = test_y[n]


print('\n\n ----------- PREDICCIÓN -----------------\n')

frase = ' Se predijo que el paciente tiene ansiedad nivel:' 
if pred == 0:
    print(frase + ' ligera')
elif pred == 1:
    print(frase + ' moderada')
elif pred == 2:
    print(frase + ' normal')
elif pred == 3:
    print(frase + ' severa')

if pred == y_true:
    print("La predicción es correcta")
else:
    print("La predicción es incorrecta")
