import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

class kVecinos():
    def __init__(self, _x, _y):
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(_x, _y, test_size=0.2, random_state=0)
        self.__space = {
                'n_neighbors': hp.choice('n_neighbors', [10, 50, 300, 750, 1200, 1300, 2500]),
                'weights': hp.choice('weights', ["uniform", "distance"]),
                'algorithm': hp.choice('algorithm', ["auto", "ball_tree", "kd_tree", "brute"]),
                'leaf_size': hp.quniform('leaf_size', 30, 500, 10),
                'p': hp.choice('p', [1, 2]), 
                'metric': hp.choice('metric', ["euclidean", "manhattan", "minkowski"])
                }
        self.nNeigborsConf = {0:10, 1:50, 2:300, 3:750, 4:1200, 5:1300, 6:2500}
        self.weightsConf = {0: "uniform", 1:"distance"}
        self.algorithmConf = {0: "auto", 1: "ball_tree", 2: "kd_tree", 3: "brute"}
        self.pConf = {0:1, 1:2}
        self.metricConf = {0: "euclidean",  1: "manhattan", 2: "minkowski"}

    def getSpace(self):
        return self.__space

    def getXtrain(self):
        return self.__xTrain

    def getXtest(self):
        return self.__xTest

    def getYtrain(self):
        return self.__yTrain

    def getYtest(self):
        return self.__yTest

    def setSpace(self, _space):
        self.__space = _space

    def objetive(self, _space):
        modelo = KNeighborsClassifier(n_neighbors = _space['n_neighbors'],
                                      weights = _space['weights'],
                                      algorithm = _space['algorithm'],
                                      leaf_size = _space['leaf_size'],
                                      p = _space['p'],
                                      metric = _space['metric'])
        precision = cross_val_score(modelo, self.__xTrain.values, self.__yTrain.values, scoring='f1_micro', cv = 5).mean()
        return {'loss': -precision, 'status': STATUS_OK}

    def pruebas(self):
        prueba = Trials()
        mejor = fmin(fn = self.objetive,
                     space = self.getSpace(),
                     algo = tpe.suggest,
                     max_evals = 80,
                     trials = prueba)
        salida = {
                'n_neighbors': self.nNeigborsConf[mejor['n_neighbors']],
                'weights': self.weightsConf[mejor['weights']],
                'algorithm': self.algorithmConf[mejor['algorithm']],
                'leaf_size': mejor['leaf_size'],
                'p': self.pConf[mejor['p']], 
                'metric': self.metricConf[mejor['metric']]
                }
        return salida
