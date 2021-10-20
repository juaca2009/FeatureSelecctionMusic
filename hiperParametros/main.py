import pandas as pd
from decisionTrees import decisionTrees
from estocastica import estocastica
from randomForest import randomForest
from kVecinos import kVecinos 

def main(): 
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    dataArboles = data[['lds', 'nrg', 'flu', 'entr', 
                        'danc', 'bpm', 'ptch', 'mfcc1', 
                        'mfcc2', 'mfcc3', 'mfcc4', 'gen']]
    xArboles = dataArboles.iloc[:,0:11]
    yArboles = dataArboles.iloc[:,-1]

    dataRandom = data[['zcr', 'flt', 'alds', 
                       'strpk', 'nrg', 'cntr', 'flu', 
                       'entr', 'danc', 'bpm', 'ptch', 
                       'edm', 'mfcc1', 'mfcc2', 'mfcc3', 
                       'mfcc4', 'mfcc5', 'gen']]
    xRandom = dataRandom.iloc[:,0:17]
    yRandom = dataRandom.iloc[:,-1]

    dataKvecinos = data[['flt', 'lds', 'alds', 'strpk', 
                         'nrg', 'flu', 'entr', 'danc', 
                         'bpm', 'ptch', 'mfcc1', 'mfcc2', 
                         'mfcc3', 'mfcc4', 'gen']]
    xVecinos = dataKvecinos.iloc[:,0:14]
    yVecinos = dataKvecinos.iloc[:,-1]


    #arboles = decisionTrees(xArboles, yArboles)
    #mejorArboles = arboles.pruebas()
    #print(mejorArboles)

    #estocastico = estocastica(x, y)
    #mejorEstocastico = estocastico.pruebas()
    #print(mejorEstocastico)

    #random = randomForest(x, y)
    #mejorRandom = random.pruebas()
    #print(mejorRandom)

    vecinos = kVecinos(xVecinos, yVecinos)
    mejorVecinos = vecinos.pruebas()
    print(mejorVecinos)

if __name__ == '__main__':
    main()
