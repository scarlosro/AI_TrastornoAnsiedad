from ctypes import sizeof
import numpy as np
import h5py
import scipy.io 
import glob
import tensorflow as tf



####### PREPROCESAMIENTO, EL CUAL NOS PERMITE SEPARAR LOS EEGs en segundos para entrenar las redes neuronales ################


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

n_label = 0
n_ligera = 0
n_moderada = 0
n_normal = 0
n_severa = 0
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
            if labels[n_label] == 'LIGERA':
                #print('Vamos a guardar porque es ligera')
                dato_p_mat = {"data": dato_segundo, "label": 0}
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/ligera/ligera_' +  str(n_ligera) + '.mat'
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_ligera = n_ligera + 1
            elif labels[n_label] == 'MODERADA':
                #print('Vamos a guardar porque es moderada')
                dato_p_mat = {"data": dato_segundo, "label": 1}
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/moderada/moderada_' +  str(n_moderada) + '.mat'
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_moderada = n_moderada + 1
            elif labels[n_label] == 'NORMAL':
                dato_p_mat = {"data": dato_segundo, "label": 2}
                #print('Vamos a guardar porque es Normal')
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/normal/normal_' +  str(n_normal) + '.mat'
                ##print("el nombre de archivo es " + arc_nombre)
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_normal= n_normal + 1
            elif labels[n_label] == 'SEVERA':
                #print('Vamos a guardar porque es severa')
                dato_p_mat = {"data": dato_segundo, "label": 3}
                arc_nombre = '/Users/carlossanchez/Desktop/AI_TrastornoAnsiedad/severa/severa_' +  str(n_severa) + '.mat'
                scipy.io.savemat(arc_nombre, dato_p_mat)
                n_severa= n_severa + 1
            posicion = posicion + 128
        if capa%2 == 1:
            n_label = n_label + 1
            





