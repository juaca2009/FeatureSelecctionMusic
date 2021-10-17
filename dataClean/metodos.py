import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data

def escalarNoGaussianos(_data, _listaColumnas):
    escalador = MinMaxScaler()
    for i in _listaColumnas:
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data


