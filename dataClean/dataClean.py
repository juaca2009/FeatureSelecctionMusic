import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from metodos import escalarAtributos, eliminarColumna, codificarVariables


if __name__ == "__main__":
    data = pd.read_csv("FeaturesObtain.csv", 
                       sep=',', header='infer')
    data = eliminarColumna(['Name', 'rof'], data) 
    data = codificarVariables(data)
    data = escalarAtributos(data)    
    data.to_csv('./dataClean.csv', sep=',', header=True, index = False)
