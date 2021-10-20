import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

class decisionTrees():
    def __init__(self, _x, _y):
        #self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(_x, _y, test_size=0.2, random_state=0)
        self.__x = _x
        self.__y = _y
        self.__space = {
                'criterion': hp.choice('criterion', ["gini", "entropy"]),
                'splitter':hp.choice('splitter', ["best", "random"]),
                'max_depth':hp.uniform('max_depth', 7500, 10500),
                'min_samples_split': hp.uniform('min_samples_split', 0, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                'max_features': hp.choice('max_features', ["auto", "sqrt", "log2", None])
                }
        self.criterionConf = {0: 'gini', 1: 'entropy'}
        self.splitterConf = {0: 'best', 1: 'random'}
        self.maxFeaturesConf = {0: 'max_features', 1: 'sqrt', 2: 'log2', 3: 'None'}
        self.maxDepthConf = {0:3500, 1:4000, 2:4500, 3:5000, 4:5500, 5:6000, 6:6500, 7:7000}

    def getSpace(self):
        return self.__space

    def setSpace(self, _space):
        self.__space = _space

    def objetive(self, _space):
        modelo = DecisionTreeClassifier(criterion = _space['criterion'], 
                                        splitter = _space['splitter'],
                                        max_depth = _space['max_depth'], 
                                        min_samples_split = _space['min_samples_split'],
                                        min_samples_leaf = _space['min_samples_leaf'], 
                                        max_features = _space['max_features'])
        precision = cross_val_score(modelo, self.__x, self.__y, scoring="f1_micro", cv = 10).mean()
        return {'loss': -precision, 'status': STATUS_OK}

    def pruebas(self):
        prueba = Trials()
        mejor = fmin(fn = self.objetive, 
                space = self.getSpace(), 
                algo = tpe.suggest, 
                max_evals = 3500, 
                trials = prueba)
        salida = {
                'criterion': self.criterionConf[mejor['criterion']],
                'splitter': self.splitterConf[mejor['splitter']],
                'max_depth': mejor['max_depth'],
                'min_samples_split': mejor['min_samples_split'],
                'min_samples_leaf': mejor['min_samples_leaf'],
                'max_features': self.maxFeaturesConf[mejor['max_features']]
                }
        return salida


