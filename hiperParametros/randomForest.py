import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


class randomForest():
    def __init__(self, _x, _y):
        self.__x = _x
        self.__y = _y
        self.__space = {
                'n_estimators': hp.choice('n_estimators', [135, 136, 137, 138, 139, 140 ]),
                'criterion': hp.choice('criterion', ["gini", "entropy"]),
                'max_depth': hp.uniform('max_depth', 760, 780),
                'min_samples_split': hp.uniform('min_samples_split', 0, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                'max_features': hp.choice('max_features', ["auto", "sqrt", "log2"]),
                'warm_start': hp.choice('warm_start', [False, True])
                }
        self.criterionConf = {0: 'gini', 1: 'entropy'}
        self.stimatorsConf = {0:135, 1:136, 2:137, 3:138, 4:139, 5:140}
        self.maxFeaturesConf = {0: 'auto', 1: 'sqrt', 2: 'log2'}
        self.boolConf = {0: False, 1: True}

    def getSpace(self):
        return self.__space

    def setSpace(self, _space):
        self.__space = _space

    def objetive(self, _space):
        modelo = RandomForestClassifier(n_estimators = _space['n_estimators'],
                                        criterion = _space['criterion'],
                                        max_depth = _space['max_depth'],
                                        min_samples_split = _space['min_samples_split'],
                                        min_samples_leaf = _space['min_samples_leaf'],
                                        max_features = _space['max_features'],
                                        warm_start = _space['warm_start'])
        precision = cross_val_score(modelo, self.__x, self.__y, scoring='f1_micro', cv = 10).mean()
        return {'loss': -precision, 'status': STATUS_OK}

    def pruebas(self):
        prueba = Trials()
        mejor = fmin(fn = self.objetive,
                     space = self.getSpace(),
                     algo = tpe.suggest,
                     max_evals = 300,
                     trials = prueba)
        salida = {
                'n_estimators': self.stimatorsConf[mejor['n_estimators']],
                'criterion': self.criterionConf[mejor['criterion']],
                'max_depth': mejor['max_depth'],
                'min_samples_split': mejor['min_samples_split'],
                'min_samples_leaf': mejor['min_samples_leaf'],
                'max_features': self.maxFeaturesConf[mejor['max_features']],
                'warm_start': self.boolConf[mejor['warm_start']]
                }
        return salida
