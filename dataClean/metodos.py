import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler


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


def escalarAtributos(_data):
    escalador = RobustScaler()
    _data.iloc[:,2:3] = escalador.fit(_data.iloc[:,2:3]).transform(_data.iloc[:,2:3]) #lds
    _data.iloc[:,4:5] = escalador.fit(_data.iloc[:,4:5]).transform(_data.iloc[:,4:5]) #strpk
    _data.iloc[:,6:7] = escalador.fit(_data.iloc[:,6:7]).transform(_data.iloc[:,6:7]) #cntr
    _data.iloc[:,8:9] = escalador.fit(_data.iloc[:,8:9]).transform(_data.iloc[:,8:9]) #entr
    _data.iloc[:,9:10] = escalador.fit(_data.iloc[:,9:10]).transform(_data.iloc[:,9:10]) #danc
    _data.iloc[:,10:11] = escalador.fit(_data.iloc[:,10:11]).transform(_data.iloc[:,10:11]) #bpm
    _data.iloc[:,11:12] = escalador.fit(_data.iloc[:,11:12]).transform(_data.iloc[:,11:12]) #tufre
    _data.iloc[:,13:14] = escalador.fit(_data.iloc[:,13:14]).transform(_data.iloc[:,13:14]) #tmpo
    _data.iloc[:,16:21] = escalador.fit(_data.iloc[:,16:21]).transform(_data.iloc[:,16:21]) #mfccs
    return _data

