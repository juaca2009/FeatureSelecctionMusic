import pandas as pd
from decisionTrees import decisionTrees

def main():
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    arboles = decisionTrees(x, y)
    mejorArboles = arboles.pruebas()
    print(mejorArboles)



if __name__ == '__main__':
    main()
