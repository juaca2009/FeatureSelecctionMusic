import pandas as pd
from decisionTrees import paramTrees

def main():
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    arboles = paramTrees(x, y)
    mejorArboles = arboles.pruebas()
    print(mejorArboles)



if __name__ == '__main__':
    main()
