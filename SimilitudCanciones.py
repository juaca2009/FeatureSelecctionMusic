import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilitudCanciones:
    def similitud_total_canciones(self, matiz_canciones):
        """
        calcula la matriz por primera vez, todas las canciones contra todas las canciones
        """
        return cosine_similarity(matiz_canciones)

    def similitud_cancion_nueva(self, matriz_canciones, cancion_nueva):
        return cosine_similarity(cancion_nueva, matriz_canciones)


if __name__ == '__main__':
    columnas = ['flt', 'lds', 'alds', 'strpk', 'nrg', 'flu', 'entr', 'danc', 'bpm',
                'ptch', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'gen']
    datosPrueba = [[0.568667244689075,0.729437808132881,0.4992779143413797,0.5628362510189954,-0.438740706623046,
                   -0.4738622648584654,0.3431384485889257,0.2615132085497044,-0.0795412476704812,0.933254075247462,
                   0.3679945499230719,0.518245064260006,1.2129990303870597,0.5959841806816465,0], 
                   [0.5854031412574541,0.7291531697351371,0.870328246194318,0.4268285335477386,-0.2534865808724664,
                   0.0957840162673569,0.4781445807293902,0.5396885370107524,-0.1572761265025139,-0.7567099263608518,
                   0.428000806961295,0.3544408487443967,1.867719756027289,0.5806000011522088,0],
                   [0.3104848650957746,0.6639412999376224,0.8953853072593292,0.6864556388603458,-0.5303177472162646,
                   -0.1092975381436573,0.5856675370349906,0.4491890007059976,1.822327175954276,1.0980830548299971,
                   0.6850658526802786,0.4494339680818012,-0.6249855434587317,0.2723479821396519,6],
                   [0.5454100364051315,0.955126638887934,0.8102903344358685,0.6377047333777139,-0.6526030270044498,
                   -0.3830187985432187,0.4614929886238226,0.4612102445900171,-0.5419512532080161,-0.071247848848904,
                   0.448777551955223,0.5568364266238717,0.1776048394068163,0.7140161473050404,6]]
    #df = pd.DataFrame(datosPrueba, columns=columnas)
    df = pd.read_csv("dataCleanFinal.csv", sep=',', header='infer')
    print(df.shape)
    array = df.to_numpy()
    a = SimilitudCanciones()
    #print(array)
    print(a.similitud_total_canciones(array))