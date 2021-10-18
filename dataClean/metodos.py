import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

def eliminarColumna(_nombre, _data):
    _data = _data.drop(_nombre, axis=1)
    return _data

def codificarVariables(_data):
    print("variables codificadas: ")
    for c in _data.columns:
        if _data[c].dtype == 'object':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(_data[c].values)) 
            _data[c] = lbl.transform(list(_data[c].values))
    return _data

def escalarGaussianos(_data, _listaColumnas):
    escalador = StandardScaler()
    for i in _listaColumnas:
        print(i)
        print(_data.iloc[:,_listaColumnas[i]])
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data

def escalarNoGaussianos(_data, _listaColumnas):
    escalador = MinMaxScaler()
    for i in _listaColumnas:
        print(i)
        print(_data.iloc[:,_listaColumnas[i]])
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data

def overSampling(_data):
    x = _data.iloc[:, 0:21]
    y = _data.iloc[:, -1]
    over = RandomOverSampler()
    xSampling, ySampling = over.fit_resample(x,y)
    _data = pd.concat([xSampling, ySampling], axis=1)
    return _data

def reemplazarOutlersG(_data, _listaGaussianos):
    for i in _listaGaussianos:
        IQR = _data[i].quantile(0.75) - _data[i].quantile(0.25)
        media = _data[i].mean()
        extremoInferior = _data[i].quantile(0.25) - (IQR*1.5)
        extremoSuperior = _data[i].quantile(0.75) + (IQR*1.5)
        _data.loc[_data[i]<=extremoInferior, i] = extremoInferior
        _data.loc[_data[i]>=extremoSuperior, i] = extremoSuperior
    return _data

def reemplazarOutlersNG(_data, _listaNoGaussianos):
    for i in _listaNoGaussianos:
        IQR = _data[i].quantile(0.75) - _data[i].quantile(0.25)
        media = _data[i].mean()
        extremoInferior = _data[i].quantile(0.25) - (IQR*1.5)
        extremoSuperior = _data[i].quantile(0.75) + (IQR*1.5)
        _data.loc[_data[i]<=extremoInferior, i] = media
        _data.loc[_data[i]>=extremoSuperior, i] = media
    return _data


    

def graficarHistogramas(_data, _titulo = 'Histograma '):
    columnas = list(_data.columns)
    for i in columnas:
        plt.hist(_data[i])
        plt.title(_titulo+i)
        plt.show()

def graficarCajas(_data, _titulo = 'Cajas y Bigotes '):
    columnas = list(_data.columns)
    for i in columnas:
        plt.boxplot(_data[i])
        plt.title(_titulo + i)
        plt.show()
