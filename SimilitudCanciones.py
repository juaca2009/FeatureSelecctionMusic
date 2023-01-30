import pandas as pd
from numpy import dot
from numpy.linalg import norm


def similitud_entre_canciones(cancion_base, cacion_comparar):
    return dot(cancion_base, cacion_comparar) / (norm(cancion_base) * norm(cacion_comparar))


def similitud_una_muchas(cancion, canciones):
    similitud, fila = list(), list()
    similitud.append(canciones["Name"].tolist())
    fila.append(cancion["Name"].values[0])
    cancion_sin_nombre = cancion.drop(['Name'], axis=1)
    canciones_sin_nombre = canciones.drop(['Name'], axis=1)
    for i in range(len(canciones)):
        temp = canciones_sin_nombre.iloc[i]
        fila.append(similitud_entre_canciones(cancion_sin_nombre.values.tolist()[0], temp.values.tolist()))
    similitud.append(fila)
    return similitud


def similitud_muchas_muchas(canciones1, canciones2):
    matriz_similitud = list()
    matriz_similitud.append(canciones1["Name"].tolist())
    matriz_similitud.append(canciones2["Name"].tolist())
    canciones1_sin_nombre = canciones1.drop(['Name'], axis=1)
    canciones2_sin_nombre = canciones2.drop(['Name'], axis=1)
    for i in range(len(canciones1_sin_nombre)):
        fila = list()
        for j in range(len(canciones2_sin_nombre)):
            fila.append(similitud_entre_canciones(canciones1_sin_nombre.iloc[i], canciones2_sin_nombre.iloc[j]))
        matriz_similitud.append(fila)
    return matriz_similitud


# configuracion impresion pandas
def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None


if __name__ == '__main__':
    set_pandas_display_options()
    columnas = ['Name', 'flt', 'lds', 'alds', 'strpk', 'nrg', 'flu', 'entr', 'danc', 'bpm',
                'ptch', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4']
    data = pd.read_csv("dataCleanFinalConNombres.csv", sep=',', header='infer')

    # *** Prueba de una cancion contra una cancion ***
    # fila1 = pd.DataFrame(data.iloc[39]).transpose()
    # fila2 = pd.DataFrame(data.iloc[44]).transpose()
    # print(fila1)
    # print(fila2)
    # fila1 = fila1.drop(['Name'], axis=1)
    # fila2 = fila2.drop(['Name'], axis=1)
    # fl1 = fila1.values.tolist()
    # fl2 = fila2.values.tolist()
    # print(similitud_entre_canciones(fila1.values.tolist()[0], fila2.values.tolist()[0]))

    # *** prueba de una cancion contra muchas ***
    # fila1 = pd.DataFrame(data.iloc[39]).transpose()
    # filas = pd.DataFrame(data.iloc[29:81])
    # result = similitud_una_muchas(fila1, filas)
    # result_dataframe = pd.DataFrame([result[1][1:len(result[0]) + 1]], columns=result[0], index=result[1][:1])
    # print(result_dataframe)

    # *** prueba de muchas canciones contra muchas canciones ***
    filas1 = pd.DataFrame(data.iloc[29:81])
    filas2 = pd.DataFrame(data.iloc[29:81])
    result = similitud_muchas_muchas(filas1, filas2)
    columnas = result[0]
    index = result[1]
    result.pop(0)
    result.pop(0)
    data = result
    result_dataframe = pd.DataFrame(data, columns=columnas, index=index)
    print(result_dataframe)
