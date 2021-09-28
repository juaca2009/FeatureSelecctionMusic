import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

#definicion metodos
def codificarVariables(_data):
    print("variables codificadas: ")
    for c in _data.columns:
        if _data[c].dtype == 'object':
            print(c)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(_data[c].values)) 
            _data[c] = lbl.transform(list(_data[c].values))
    return _data


def escalarAtributos(_data):
    escalador = RobustScaler()
    _data.iloc[:,2:3] = escalador.fit(_data.iloc[:,2:3]).transform(_data.iloc[:,2:3]) #lds
    _data.iloc[:,4:5] = escalador.fit(_data.iloc[:,4:5]).transform(_data.iloc[:,4:5]) #strpk
    _data.iloc[:,6:7] = escalador.fit(_data.iloc[:,6:7]).transform(_data.iloc[:,6:7]) #cntr
    _data.iloc[:,8:9] = escalador.fit(_data.iloc[:,8:9]).transform(_data.iloc[:,8:9]) #entr
    _data.iloc[:,9:10] = escalador.fit(_data.iloc[:,9:10]).transform(_data.iloc[:,9:10]) #danc
    _data.iloc[:,10:11] = escalador.fit(_data.iloc[:,10:11]).transform(_data.iloc[:,10:11]) #bpm
    _data.iloc[:,11:12] = escalador.fit(_data.iloc[:,11:12]).transform(_data.iloc[:,11:12]) #tufre
    _data.iloc[:,13:14] = escalador.fit(_data.iloc[:,13:14]).transform(_data.iloc[:,13:14]) #tmpo
    _data.iloc[:,16:21] = escalador.fit(_data.iloc[:,16:21]).transform(_data.iloc[:,16:21]) #mfccs
    return _data


def imprimirCorrelacion(_data):
    correlacion = _data.corr()
    sns.heatmap(correlacion, annot = True)
    plt.show()

def featureSelectionNeighbors(_nFeatures, _xtrain, _ytrain):
    clf = NearestCentroid()
    sfs = SFS(clf, k_features = _nFeatures, forward=True, floating=False, 
             verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(_xtrain, _ytrain)
    print(sfs.k_feature_names_)
    print(sfs.k_feature_idx_)

def featureSelectionDecision(_nFeatures, _xtrain, _ytrain):
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=10)
    sfs = SFS(decision_tree, k_features = _nFeatures, forward=True, floating=False, 
              verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(_xtrain, _ytrain)
    print(sfs.k_feature_names_)
    print(sfs.k_feature_idx_)

def featureSelectionStochastic(_nFeatures, _xtrain, _ytrain):
    stochastic = SGDClassifier(loss="log", penalty="l1", max_iter=150)
    sfs = SFS(stochastic, k_features = _nFeatures, forward=True, floating=False, 
              verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(_xtrain, _ytrain)
    print(sfs.k_feature_names_)
    print(sfs.k_feature_idx_)
    
def featureSelectionRandomForest(_nFeatures, _xtrain, _ytrain):
    random = RandomForestClassifier(n_estimators=100, random_state=2)
    sfs = SFS(random, k_features = _nFeatures, forward=True, floating=False, 
              verbose=2, scoring='f1_micro', n_jobs=-1, cv=10).fit(_xtrain, _ytrain)
    print(sfs.k_feature_names_)
    print(sfs.k_feature_idx_)

def limpiarDatos():
    #carga del data set 
    _data = pd.read_csv("FeaturesObtain.csv", sep=',', header='infer')
    #eliminacion columna nombre
    _data = _data.drop(['Name','rof'], axis=1)
    #codificacion variables categoricas
    _data = codificarVariables(_data)
    return data

if __name__ == "__main__":
    data = pd.read_csv("dataClean.csv", sep=',', header='infer')

    #escalamiento de atributos
    data = escalarAtributos(data)

    #separacion de datos
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 

    #implementacion forward selection Nearest Neighbors
    #featureSelectionNeighbors(int(sys.argv[1]), x_train, y_train)
    
    #implementacion forward selection Decision Trees
    #featureSelectionDecision(int(sys.argv[1]), x_train, y_train)
    
    #implementacion forward selection stochastic
    #featureSelectionStochastic(int(sys.argv[1]), x_train, y_train)
    
    #implementacion forward selection ramdom featureSelectionRandomForest
    #featureSelectionRandomForest(int(sys.argv[1]), x_train, y_train)
    
    #data.to_csv('dataClean.csv', sep=',', header=True, index = False)
