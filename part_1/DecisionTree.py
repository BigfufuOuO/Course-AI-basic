from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 
import pdb
import json


# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

# model
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.tree = None
        
    def _isXsplit(self, X):
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > 1:
                return True
        return False
    
    def _caculateEntropy(self, D):
        '''
            D: a list of labels
        '''
        if len(np.unique(D)) == 1:
            return 0
        _, counts = np.unique(D, return_counts=True)
        p = counts / len(D)
        entropy = -np.sum(p * np.log2(p))
        return entropy
    
    def _getBestFeature(self, X, y):
        '''
            X: [n_samples_train, n_features]
            y: [n_samples_train, ]
            return: a dictionary:
                [feature coloum, 
                 entropy value,
                 float bool indicator,
                 if bool indicator is True, return the split point of the feature]
        '''
        entropy_list = []
        ## caculate Ent(D)
        entropy = self._caculateEntropy(y)
        for feature in X.columns:
            entropy_feature = 0
            if X[feature].dtype == 'float64':
                T = []
                sublist = []
                unique_set = np.unique(X[feature])
                for i in range(len(unique_set) - 1):
                    T.append((unique_set[i] + unique_set[i+1]) / 2)
                    
                for t in T:
                    entropy_feature = 0
                    less_index = np.where(X[feature] < t)
                    greater_index = np.where(X[feature] >= t)
                    entropy_feature += len(less_index[0]) / len(y) * self._caculateEntropy(y[less_index])
                    entropy_feature += len(greater_index[0]) / len(y) * self._caculateEntropy(y[greater_index])
                    sublist.append(entropy - entropy_feature)
                if len(sublist) > 0:
                    entropy_list.append([feature, max(sublist), 1, T[sublist.index(max(sublist))]])
                else:
                    entropy_list.append([feature, 0, 1, 0])
            else:
                for value in np.unique(X[feature]):
                    index = np.where(X[feature] == value)
                    entropy_feature += len(index) / len(y) * self._caculateEntropy(y[index])
                entropy_list.append([feature, entropy - entropy_feature, 0, 0])
        
        max_entropy = np.argmax(entropy_list, axis=0)[1]
        return entropy_list[max_entropy]

    
    def _debugTree(self, tree:dict):
        with open ('tree.csv', 'w') as f:
            df = pd.DataFrame(tree)
            df.to_csv(f)

    def fit(self, X, y):
        # X: [n_samples_train, n_features], 
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        if len(np.unique(y)) == 1:
            # print('y cannot be split')
            return y[0]
        
        # if X cannot be split
        if X.shape[1] == 1 or X.duplicated(keep=False).all() or X.shape[0] < 2:
            # print('X cannot be split')
            return np.argmax(np.bincount(y))
        
        bestFeature, _, float_indicator, float_split = self._getBestFeature(X, y)
        tree = {bestFeature: {}}
        
        # Generate child
        if float_indicator:
            less_index = np.where(X[bestFeature] < float_split)
            greater_index = np.where(X[bestFeature] >= float_split)
            tree[bestFeature]['<{}'.format(float_split)] = self.fit(X[X[bestFeature] < float_split].drop(columns=[bestFeature]), y[less_index])
            tree[bestFeature]['>={}'.format(float_split)] = self.fit(X[X[bestFeature] >= float_split].drop(columns=[bestFeature]), y[greater_index])
        else:
            for value in np.unique(X[bestFeature]):
                index = np.where(X[bestFeature] == value)
                if len(index) == 0:
                    tree[bestFeature][value] = np.argmax(np.bincount(y))
                    continue
                else:
                    split_X = X[X[bestFeature] == value].drop(columns=[bestFeature])
                    tree[bestFeature][value] = self.fit(split_X, y[index])
        return tree
        pass

    def predict(self, X):
        # X: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        y = np.zeros(X.shape[0])
        # TODO:
        return y

def load_data(datapath:str='./data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', ]
    discrete_features = ['Gender', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS']
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data('./part_1/data/ObesityDataSet_raw_and_data_sinthetic.csv')
    clf = DecisionTreeClassifier()
    tree = clf.fit(X_train, y_train)
    clf._debugTree(tree)
    
    y_pred = clf.predict(X_test)
    print(accuracy(y_test, y_pred))