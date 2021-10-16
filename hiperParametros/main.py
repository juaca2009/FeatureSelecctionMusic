import pandas as pd
from decisionTrees import decisionTrees
from estocastica import estocastica
from randomForest import randomForest
from kVecinos import kVecinos 

def main(): 
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    arboles = decisionTrees(x, y)
    mejorArboles = arboles.pruebas()
    print(mejorArboles)

    #estocastico = estocastica(x, y)
    #mejorEstocastico = estocastico.pruebas()
    #print(mejorEstocastico)

    #random = randomForest(x, y)
    #mejorRandom = random.pruebas()
    #print(mejorRandom)

    #vecinos = kVecinos(x, y)
    #mejorVecinos = vecinos.pruebas()
    #print(mejorVecinos)

if __name__ == '__main__':
    main()
