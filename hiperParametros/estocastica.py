import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

class estocastica():
    def __init__(self, _x, _y):
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(_x, _y, test_size=0.2, random_state=0)
        self.__space = {
                'loss': hp.choice('loss', ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]),
                'penalty': hp.choice('penalty', ["l2", "l1", "elasticnet"]),
                'alpha': hp.uniform('alpha', 0, 1),
                'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                'fit_intercept': hp.choice('fit_intercept', [False, True]),
                'max_iter': hp.quniform('max_iter', 1000, 10000, 10),
                'shuffle': hp.choice('shuffle', [False, True]),
                'learning_rate': hp.choice('learning_rate', ["constant", "optimal", "invscaling", "adaptive"]),
                'eta0': hp.uniform('eta0', 0, 1),
                'power_t': hp.uniform('power_t', 0, 1),
                'early_stopping': hp.choice('early_stopping', [False, True]),
                'validation_fraction': hp.uniform('validation_fraction', 0, 1),
                'n_iter_no_change': hp.quniform('n_iter_no_change', 5, 1500, 10),
                'warm_start': hp.choice('warm_start', [False, True]),
                'average': hp.choice('average', [False, True])
                }
        self.lossConf = {0: "hinge", 1: "log", 2: "modified_huber", 
                         3: "squared_hinge", 4: "perceptron"}
        self.penaltyConfi = {0: 'l2', 1:'l1', 2:"elasticnet"}
        self.boolConf = {0: False, 1: True}
        self.learningRateConf = {0: "constant", 1: "optimal", 2: "invscaling", 3: "adaptive"}

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
        modelo = SGDClassifier(loss = _space['loss'],
                penalty = _space['penalty'],
                alpha = _space['alpha'],
                l1_ratio = _space['l1_ratio'],
                fit_intercept = _space['fit_intercept'],
                max_iter = _space['max_iter'],
                shuffle = _space['shuffle'],
                learning_rate = _space['learning_rate'],
                eta0 = _space['eta0'],
                power_t = _space['power_t'],
                early_stopping = _space['early_stopping'],
                validation_fraction = _space['validation_fraction'],
                n_iter_no_change = _space['n_iter_no_change'],
                warm_start = _space['warm_start'],
                average = _space['average'])
        presicion = cross_val_score(modelo, self.__xTrain, self.__yTrain, scoring="f1_micro", cv = 5).mean() 
        return {'loss': -presicion, 'status': STATUS_OK}

    def pruebas(self):
        prueba = Trials()
        mejor = fmin(fn = self.objetive,
                space = self.getSpace(),
                algo = tpe.suggest,
                max_evals = 80,
                trials = prueba)
        salida = {
                'loss': self.lossConf[mejor[ 'loss']],
                'penalty': self.penaltyConfi[mejor[ 'penalty']],
                'alpha': mejor['alpha'],
                'l1_ratio': mejor['l1_ratio'],
                'fit_intercept': self.boolConf[mejor['fit_intercept']],
                'max_iter': mejor['max_iter'],
                'shuffle': self.boolConf[mejor['shuffle']],
                'learning_rate': self.learningRateConf[mejor['learning_rate']],
                'eta0': mejor['eta0'],
                'power_t': mejor['power_t'],
                'early_stopping': self.boolConf[mejor['early_stopping']],
                'validation_fraction': mejor['validation_fraction'],
                'n_iter_no_change': mejor['n_iter_no_change'],
                'warm_start': self.boolConf[mejor[ 'warm_start']],
                'average': self.boolConf[mejor['average']]
                }
        return salida
