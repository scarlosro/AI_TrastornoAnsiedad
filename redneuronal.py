from ctypes import sizeof
import numpy as np
import h5py
import scipy.io 
import glob
import tensorflow as tf

#obten todos los archivos matlab de la carpeta
matfiles = glob.glob('/Users/carlossanchez/Desktop/DASPS_Database/Preprocesseddata.mat/*.mat')


#diccionario de los matlabs ordenados para la lectura
data = {}
data = sorted(matfiles)

#Obtener las labels para la clasificación de los archivos.
labels_file = open("labels.txt","r")
labels=[]
labels =labels_file.read()


for i in range(len(data)):
    print(i)


f = h5py.File('S22preprocessed.mat','r')
#f = h5py.File('matlab_matrix.mat','r')
#mat = scipy.io.loadmat('matlab_matrix.mat')
print(f['data'])
#scipy.io.savemat("matlab_matrix.mat", f['data'][0:14,0:128,0])
n1 = np.array(f['data'][0,0:128,0:14])
n1 = n1.transpose()
#nf = open('prueba.mat','w')
#nf.write(f['data'][0:14,0:128,0])
mdic = {"data": n1, "label": "data"}
scipy.io.savemat("matlab_matrix.mat", mdic)
print(n1.shape)


