from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
from keras.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay
import os
import random
import shutil
from scipy.io import loadmat

# Librerias de ayuda
import numpy as np
import glob
import h5py



listaAnsiosos =  glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/ansiosoA/*.mat')
listaNoAnsiosos  =  glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/noansiosoA/*.mat')

def dividirTestAndTrain(lista,size_tr):
    path = os.getcwd()
    size_lista = len(lista)
    for i in range(size_lista):
        no_arch = random.randint(0,len(lista)-1)
        archivo = lista[no_arch]
        if i < size_lista * size_tr:
            shutil.copy2(archivo,path + '/trainBA', follow_symlinks=True)
            lista.remove(archivo)
        else:
            shutil.copy2(archivo,path + '/testBA', follow_symlinks=True)
            lista.remove(archivo)



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


dividirTestAndTrain(listaAnsiosos, .70)
dividirTestAndTrain(listaNoAnsiosos, .70)


size_ansioso = len(glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/trainB/*.mat'))
size_nansioso = len(glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testB/*.mat'))

print('Ansiosos ', size_ansioso, ' no ', size_nansioso)

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
write_h5file('./trainBA','trainBinA')
write_h5file('./testBA','testBinA')



# train_x, train_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_1/h5.h5')
train_x, train_y = load_Dataset_from_h5file('./trainBinA.h5')
#print(train_x[0].shape)
#test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
test_x, test_y = load_Dataset_from_h5file('./testBinA.h5')

#print(train_x.shape)
#train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
print("X ", train_x.shape)
print("Y ", train_y.shape)


model = Sequential()
model.add(Conv1D(100, 2, activation="relu", input_shape=(1792,1)))
model.add(Dense(100, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()
#model.fit(train_x, train_y, batch_size=15,epochs=100)
model.fit(train_x, train_y,epochs=100)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)

test_y = np.squeeze(np.asarray(test_y))
y_prediction = model.predict(test_x)
y_prediction = y_prediction.flatten()


y_prediction = np.where(y_prediction > 0.5, 1, 0)
print(test_y)

tn, fp, fn, tp = confusion_matrix(test_y, y_prediction).ravel()
print('Confusion Matrix')
#print(pred)
print(' TN ', tn, ' fp ',fp, ' fn ', fn ,' tp ', tp)

print(classification_report(test_y, y_prediction))
'''
pred = model.predict(test_x)
pred_y = pred.argmax(axis=1)

cm = confusion_matrix(test_y, pred_y)
print(cm)

'''