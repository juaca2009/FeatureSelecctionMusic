import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

class paramTrees():
    def __init__(self, _x, _y):
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(_x, _y, test_size=0.2, random_state=0)
        self.__space = {
                'criterion': hp.choice('criterion', ["gini", "entropy"]),
                'splitter':hp.choice('splitter', ["best", "random"]),
                'max_depth':hp.quniform('max_depth', 10, 500, 10),
                'min_samples_split': hp.uniform('min_samples_split', 0, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                'max_features': hp.choice('max_features', ["auto", "sqrt", "log2", None])
                }
        self.criterionConf = {0: 'gini', 1: 'entropy'}
        self.splitterConf = {0: 'best', 1: 'random'}
        self.maxFeaturesConf = {0: 'max_features', 1: 'sqrt', 2: 'log2', 3: 'None'}

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
        modelo = DecisionTreeClassifier(criterion = _space['criterion'], splitter = _space['splitter'],
                                        max_depth = _space['max_depth'], min_samples_split = _space['min_samples_split'],
                                        min_samples_leaf = _space['min_samples_leaf'], max_features = _space['max_features'])
        precision = cross_val_score(modelo, self.__xTrain, self.__yTrain, cv = 5).mean()
        return {'loss': -precision, 'status': STATUS_OK}

    def pruebas(self):
        pruebas = Trials()
        mejor = fmin(fn = self.objetive, space = self.getSpace(), algo = tpe.suggest, max_evals = 80, trials = pruebas)
        salida = {
                'criterion': self.criterionConf[mejor['criterion']],
                'splitter': self.splitterConf[mejor['splitter']],
                'max_depth': mejor['max_depth'],
                'min_samples_split': mejor['min_samples_split'],
                'min_samples_leaf': mejor['min_samples_leaf'],
                'max_features': self.maxFeaturesConf[mejor['max_features']]
                }
        return salida


