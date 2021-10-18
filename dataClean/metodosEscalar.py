import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def escalarGaussianos(_data, _listaColumnas):
    escalador = StandardScaler()
    for i in _listaColumnas:
        #print(i)
        #print(_data.iloc[:,_listaColumnas[i]])
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data

def escalarNoGaussianos(_data, _listaColumnas):
    escalador = MinMaxScaler()
    for i in _listaColumnas:
        #print(i)
        #print(_data.iloc[:,_listaColumnas[i]])
        _data.iloc[:,[_listaColumnas[i]]] = escalador.fit(_data.iloc[:,[_listaColumnas[i]]]).transform(_data.iloc[:,[_listaColumnas[i]]])
    return _data
