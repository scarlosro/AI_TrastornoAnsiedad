from ctypes import sizeof
import numpy as np
import h5py
import scipy.io 
import glob
import tensorflow as tf

#obten todos los archivos matlab de la carpeta
matfiles = glob.glob('/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/Preprocesseddatamat/*.mat')


#diccionario de los matlabs ordenados para la lectura
data = {}
data = sorted(matfiles)
#print(data)


#Obtener las labels para la clasificación de los archivos.
labels_file = open("labels.txt","r")
labels=[]
for line in labels_file:
    labels.append(line.split('\n')[0])
#print(labels)

n_ansioso = 0
n_noansioso = 0
n_label=0

for mat_file in data:
    archivo = h5py.File(mat_file,'r')
    print(mat_file)
    for capa in range(12):
        #print("Empezamos con la capa ", capa )
        #print("La label es ", n_label , " " + labels[n_label] )
        posicion = 0
        for segundo in range (15):
            dato_segundo = np.array(archivo['data'][capa,posicion:posicion+128,0:14])
            dato_segundo = dato_segundo.transpose()
            #print("Vamos en el segundo ", segundo)
            #print("La etiqueta a evaluar es ", n_label)
            
            #print(labels[n_label])
            if   labels[n_label] == 'NORMAL':
                print('Vamos a guardar porque es ligera')
                dato_p_mat = {"data": dato_segundo, "label": 0}
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/noansiosoA/noansioso_' +  str(n_noansioso) + '.mat'
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_noansioso = n_noansioso + 1
            elif labels[n_label] == 'MODERADA' or labels[n_label] == 'SEVERA' or labels[n_label] == 'LIGERA':
                print('Vamos a guardar porque es moderada')
                dato_p_mat = {"data": dato_segundo, "label": 1}
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/ansiosoA/ansioso_' +  str(n_ansioso) + '.mat'
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_ansioso = n_ansioso + 1
            posicion = posicion + 128
        if capa%2 == 1:
            n_label = n_label + 1
            



#f = h5py.File('matlab_matrix.mat','r')
#mat = scipy.io.loadmat('matlab_matrix.mat')
#print(f['data'])
#scipy.io.savemat("matlab_matrix.mat", f['data'][0:14,0:128,0])
#n1 = np.array(f['data'][0,0:128,0:14])
#n1 = n1.transpose()
#nf = open('prueba.mat','w')
#nf.write(f['data'][0:14,0:128,0])
#mdic = {"data": n1, "label": "data"}
#scipy.io.savemat("matlab_matrix.mat", mdic)
#print(n1.shape)


