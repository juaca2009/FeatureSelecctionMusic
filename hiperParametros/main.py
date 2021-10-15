import pandas as pd
from decisionTrees import decisionTrees
from estocastica import estocastica

def main():
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    #arboles = decisionTrees(x, y)
    #mejorArboles = arboles.pruebas()
    #print(mejorArboles)

    estocastico = estocastica(x, y)
    mejorEstocastico = estocastico.pruebas()
    print(mejorEstocastico)

if __name__ == '__main__':
    main()
