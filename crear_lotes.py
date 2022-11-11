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

from tensorflow.keras.utils import plot_model

#Creador de lotes para los batchs

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

