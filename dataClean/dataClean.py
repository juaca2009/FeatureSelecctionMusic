import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from metodosGenerales import eliminarColumna, codificarVariables, overSampling 
from metodosOutlers import reemplazarOutlersG, reemplazarOutlersNG 
from metodosEscalar import escalarGaussianos, escalarNoGaussianos
from metodosGraficar import graficarCajas, graficarHistogramas


if __name__ == "__main__":
    Lgaussianos = {'flt':1,'lds':2, 'strpk':4, 'flu':7, 
                   'entr':8, 'danc':9, 'bpm':10, 'ptch':12, 
                   'tmpo':13, 'mfcc1':16, 'mfcc2':17, 'mfcc3':18, 
                   'mfcc4':19, 'mfcc5':20}
    LNgaussianos = {'zcr':0, 'alds':3, 'nrg':5, 'cntr':6, 'tufre':11}
    data = pd.read_csv("FeaturesObtain.csv", sep=',', header='infer')
    data = eliminarColumna(['Name', 'rof'], data)
    data = codificarVariables(data)
    data = overSampling(data)
    # graficarHistogramas(data, "Histograma con Over Sampling de ")
    data = reemplazarOutlersG(data, Lgaussianos)
    data = reemplazarOutlersNG(data, LNgaussianos)
    Lgaussianos = {
            'bpm':10, 'cntr':6, 'flu':7, 'lds':2, 
            'mfcc3':18, 'mfcc5':20, 'nrg':5, 'ptch':12, 
            'zcr':0}
    LNgaussianos = {
            'alds':3, 'danc':9, 'entr':8, 'flt':1, 
            'mfcc1':16, 'mfcc2':17, 'mfcc4':19, 'strpk':4,
            'tmpo':13, 'tufre':11}
    data = escalarGaussianos(data, Lgaussianos)
    data = escalarNoGaussianos(data, LNgaussianos)
    # data.to_csv('../dataClean.csv', sep=',', header=True, index = False)
