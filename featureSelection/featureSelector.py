import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class featuresSelector():
    def __init__(self, _x, _y, _arboles, _estocastico, _randomForest, _kVecinos):
        self.__xTrain, self.__xTest, self.__yTrain, self.__yTest = train_test_split(_x, _y, test_size=0.05, random_state=0)
        self.__arbolesDecision = _arboles
        self.__estocastico = _estocastico
        self.__randomForest = _randomForest
        self.__kVecinos = _kVecinos

    def FeatureSelectionArboles(self, _nFeatures):
        sfs = SFS(self.getArbolesDecision(), k_features = _nFeatures, forward=True, floating=False, 
                  verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(self.__xTrain, self.__yTrain)
        print(sfs.k_feature_names_)
        print(sfs.k_feature_idx_)

    def FeatureSelectionEstocastico(self, _nFeatures):
        sfs = SFS(self.getEstocastico(), k_features = _nFeatures, forward=True, floating=False, 
                  verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(self.__xTrain, self.__yTrain)
        print(sfs.k_feature_names_)
        print(sfs.k_feature_idx_)

    def FeatureSelectionRandomForest(self, _nFeatures):
        sfs = SFS(self.getRandomForest(), k_features = _nFeatures, forward=True, floating=False, 
                  verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(self.__xTrain, self.__yTrain)
        print(sfs.k_feature_names_)
        print(sfs.k_feature_idx_)

    def FeatureSelectionKVecinos(self, _nFeatures):
        sfs = SFS(self.getKVecinos(), k_features = _nFeatures, forward=True, floating=False, 
                  verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(self.__xTrain.values, self.__yTrain.values)
        print(sfs.k_feature_names_)
        print(sfs.k_feature_idx_)

    def getXtrain(self):
        return self.__xTrain

    def getXtest(self):
        return self.__xTest

    def getYtrain(self):
        return self.__yTrain

    def getYtest(self):
        return self.__yTest

    def getArbolesDecision(self):
        return self.__arbolesDecision

    def getEstocastico(self):
        return self.__estocastico

    def getRandomForest(self):
        return self.__randomForest

    def getKVecinos(self):
        return self.__kVecinos

    def setArbolesDecision(self, _arboles):
        self.__arbolesDecision = _arboles

    def setEstocastico(self, _estocastico):
        self.__estocastico = _estocastico

    def setRandomForest(self, _randomForest):
        self.__randomForest = _randomForest

    def setKVecinos(self, _kVecinos):
        self.__kVecinos = _kVecinos
