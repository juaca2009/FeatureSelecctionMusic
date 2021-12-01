import pandas as pd
import numpy as np

def reemplazarOutlersG(_data, _listaGaussianos):
    for i in _listaGaussianos:
        IQR = _data[i].quantile(0.75) - _data[i].quantile(0.25)
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


