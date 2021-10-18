import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
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


def overSampling(_data):
    x = _data.iloc[:, 0:21]
    y = _data.iloc[:, -1]
    over = RandomOverSampler()
    xSampling, ySampling = over.fit_resample(x,y)
    _data = pd.concat([xSampling, ySampling], axis=1)
    return _data
