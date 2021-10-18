import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

