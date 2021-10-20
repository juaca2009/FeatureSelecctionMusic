import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from featureSelector import featuresSelector

def main():
    data = pd.read_csv("./dataClean.csv", sep=',', header='infer')
    #menosGeneros = data.loc[:, 'gen'] <= 4
    #data = data.loc[menosGeneros]
    x = data.iloc[:, 0:21]
    y = data.iloc[:, -1]

    arboles = DecisionTreeClassifier(random_state=1, 
                                     max_depth=100)

    estocastico = SGDClassifier(loss = 'squared_hinge', 
                                penalty = 'l1', 
                                alpha = 0.661928220292505, 
                                l1_ratio = 0.831933837820263, 
                                fit_intercept = True, 
                                max_iter = 10010, 
                                shuffle = True, 
                                learning_rate = 'adaptive', 
                                eta0 = 0.6432646081468951, 
                                power_t = 0.2703289373438328, 
                                early_stopping = True, 
                                validation_fraction = 0.02516293604153648, 
                                n_iter_no_change = 910.0, 
                                warm_start = True, 
                                average = False)

    radom = RandomForestClassifier(n_estimators = 100, 
                                  criterion = 'entropy', 
                                  max_features = 'sqrt') 
    
    kVecinos = KNeighborsClassifier(n_neighbors = 10, 
                                   weights = 'distance', 
                                   algorithm = 'ball_tree', 
                                   leaf_size = 230.0, 
                                   p = 1, 
                                   metric = 'minkowski') 

    selector = featuresSelector(x, y, arboles, estocastico, radom, kVecinos, 0.1)
    selector.FeatureSelectionRandomForest(int(sys.argv[1]))


if __name__ == '__main__':
    main()
