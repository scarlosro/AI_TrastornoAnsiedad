from random import randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import random
import shutil
# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report


import h5py

import glob
from scipy.io import loadmat



def write_h5file(carpeta,nameF):
    
    files = glob.glob(carpeta + '/*.mat')

    # Train, Valid, Test sets
    setX = []
    setY = []


    print(' La carpeta ' +  carpeta + ' contiene ', len(files))
    for file in files:
        arc_mat = loadmat(file, struct_as_record=False)
        
        # Split the song data and labels
        data = arc_mat['data'] .flatten()
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



#crearCarpeta('/trainA',.70,True)
#crearCarpeta('/testA',.70,False)

#write_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/trainA','trainA')

#write_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/testA','testA')



# train_x, train_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_1/h5.h5')
train_x, train_y = load_Dataset_from_h5file('./trainBin.h5')
#test_x, test_y = load_Dataset_from_h5file('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/lote_26/h5.h5')
test_x, test_y = load_Dataset_from_h5file('./testBin.h5')


def normalise(x_mean, x_std, x_data):
    return (x_data - x_mean) / x_std


x_mean = train_x.mean()
x_std = train_x.std()

x_train = normalise(x_mean, x_std, train_x)
#x_val = normalise(x_mean, x_std, x_val)
x_test = normalise(x_mean, x_std, test_x)

x_train.mean(), x_train.std()


#print("El tamaño de train  es ", x_train.shape)
#print("El tamaño de test  es ", x_test.shape)
#print(train_y.shape)


def create_minibatches(x, y, mb_size, shuffle = True):

    assert x.shape[0] == y.shape[0], 'Error en cantidad de muestras'
    total_data = x.shape[0]
    #print(total_data)
    if shuffle: 
        idxs = np.arange(total_data)
        #print('El total de datos es ', total_data, 'y idxs es ', idxs)
        np.random.shuffle(idxs)
        #print(idxs)
        x = x[idxs]
        y = y[idxs]  
        
        
    return ((x[i:i+mb_size], y[i:i+mb_size]) for i in range(0, total_data, mb_size))


x_train_tensor = torch.tensor(x_train.copy())
y_train_tensor = torch.tensor(train_y.copy())

#x_val_tensor = torch.tensor(x_val.copy())
#y_val_tensor = torch.tensor(y_val.copy())

validation_x = x_test[:int(len(test_x)*.50)]
validation_y = test_y[:int(len(test_y)*.50)]
x_test = x_test[:int(len(test_x)*.50):]
test_y = test_y[:int(len(test_y)*.50):]

x_test_tensor = torch.tensor(x_test.copy())
y_test_tensor = torch.tensor(test_y.copy())
x_validation_tensor = torch.tensor(validation_x.copy())



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Estammos usando: {device}')

def accuracy(model, x, y, mb_size):
    num_correct = 0
    num_total = 0
    model.eval()
    model = model.to(device=device)
    with torch.no_grad():
        for (xi, yi) in create_minibatches(x, y, mb_size):
            xi = xi.to(device=device, dtype = torch.float32)
            yi = yi.to(device=device, dtype = torch.long)
            scores = model(xi) # mb_size, 10
            _, pred = scores.max(dim=1) #pred shape (mb_size )
            num_correct += (pred == yi.squeeze()).sum() # pred shape (mb_size), yi shape (mb_size, 1)
            num_total += pred.size(0)

            return float(num_correct)/num_total     


def train(model, optimiser, mb_size, epochs=10):
    model = model.to(device=device)
    for epoch in range(epochs):
        for (xi, yi) in create_minibatches(x_train_tensor, y_train_tensor, mb_size):
            #print('El xi es ', xi, ' y el yi es ', yi, 'Con tamaño ', xi.shape, ' y ', yi.shape)
            model.train()
            xi = xi.to(device=device, dtype=torch.float32)
            yi = yi.to(device=device, dtype=torch.long)
            scores = model(xi)
            #funcion cost
            cost = F.cross_entropy(input= scores, target=yi.squeeze())
            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
        
        #print(f'Epoch: {epoch}, accuracy: {accuracy(model, x_test_tensor, y_test_tensor, mb_size)}')

        print(f'Epoch: {epoch}, costo: {cost.item()}, accuracy: {accuracy(model, x_test_tensor, y_test_tensor, mb_size)}')



#Instanciar modelo
hidden1 = 1000 
hidden = 1000
lr = .001
epochs = 50
mb_size = 50
model1 = nn.Sequential(nn.Linear(in_features=1792, out_features=hidden1), nn.ReLU(),
                       nn.Linear(in_features=hidden1, out_features=hidden), nn.ReLU(),
                       nn.Linear(in_features=hidden, out_features=2))
optimiser = torch.optim.SGD(model1.parameters(), lr=lr)

train(model1, optimiser, mb_size, epochs)

print(accuracy(model1, x_test_tensor,  y_test_tensor, mb_size))

y_pred = []

# iterate over test data
for n, inputs in enumerate(x_validation_tensor):
        output = model1(inputs) # Feed Network
        #print(output)
        _, pred = output.max(dim=0) 
        #print(pred.item())
        y_pred.append(pred.item())

        
   

validation_y = np.squeeze(validation_y)
#print(validation_y)
#print(y_pred)
# Build confusion matrix
print('Confusion Matrix')
print(confusion_matrix(validation_y, y_pred))


print(classification_report(validation_y, y_pred))
