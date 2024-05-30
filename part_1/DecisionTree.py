from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Debug import NumpyJSONEncoder
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
        self.continous_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        self.discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
        self.discrete_features_size = {'Gender':2, 'CALC':4, 'FAVC':2, 'SCC':2, 'SMOKE':2, 'family_history_with_overweight':2, 'CAEC':4, 'MTRANS':5}
        self.misscount = 0
        
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
            iv = 0
            if feature in self.continous_features:
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
                    max_t = T[sublist.index(max(sublist))]
                    Dv = len(np.where(X[feature] < max_t)[0]) / len(y)
                    iv = -Dv * np.log2(Dv) - (1 - Dv) * np.log2(1 - Dv)
                    entropy_list.append([feature, max(sublist) / iv, 1, T[sublist.index(max(sublist))]])
                else:
                    entropy_list.append([feature, 0, 1, 0])
            else:
                for value in np.unique(X[feature]):
                    index = np.where(X[feature] == value)
                    entropy_feature += len(index) / len(y) * self._caculateEntropy(y[index])
                    iv += -len(index) / len(y) * np.log2(len(index) / len(y))
                entropy_list.append([feature, (entropy - entropy_feature)/iv, 0, 0])
        
        max_entropy = np.argmax(entropy_list, axis=0)[1]
        return entropy_list[max_entropy]

    
    def _debugTree(self, tree:dict):
        encoder = NumpyJSONEncoder(tree)
        tree_json = encoder.to_json()
        with open ('tree.json', 'w') as f:
            json.dump(tree_json, f, indent=4)

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
            for value in range(self.discrete_features_size[bestFeature]):
                index = np.where(X[bestFeature] == value)
                if len(index[0]) == 0:
                    tree[bestFeature][value] = np.argmax(np.bincount(y))
                    continue
                else:
                    split_X = X[X[bestFeature] == value].drop(columns=[bestFeature])
                    tree[bestFeature][value] = self.fit(split_X, y[index])
        return tree
        pass

    def predict(self, X, tree):
        # X: [n_samples_test, n_features],
        # tree: a dictionary
        # return: y: [n_samples_test, ]
        # TODO:
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = self._singlePredict(X.iloc[i], tree)
        return y

    def _singlePredict(self, x, tree):
        feature = list(tree.keys())[0]
        value = x[feature]
        if feature in self.continous_features:
            compare = list(tree[feature].keys())[0].split('<')[1] # '<0.5' -> '0.5'
            if value < float(compare):
                if type(tree[feature]['<{}'.format(compare)]) != dict:
                    return tree[feature]['<{}'.format(compare)]
                else:
                    return self._singlePredict(x, tree[feature]['<{}'.format(compare)])
            else:
                if type(tree[feature]['>={}'.format(compare)]) != dict:
                    return tree[feature]['>={}'.format(compare)]
                else:
                    return self._singlePredict(x, tree[feature]['>={}'.format(compare)])
        else:
            query = tree[feature].get(value, None)
            if not query:
                self.misscount += 1
                # search all the keys in the tree, find the biggest probability
                return self._getBiggestProb(tree[feature])
            if type(tree[feature][value]) != dict:
                return tree[feature][value]
            else:
                return self._singlePredict(x, tree[feature][value])
            
    def _getBiggestProb(self, tree):
        counts = dict()
        def dfs(tree):
            for key, value in tree.items():
                if type(value) == dict:
                    dfs(value)
                else:
                    counts[value] = counts.get(value, 0) + 1
        
        dfs(tree)
        return max(counts, key=counts.get) # return the key with the biggest value

def load_data(datapath:str='./data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    
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
    
    y_pred = clf.predict(X_test, tree)
    print(accuracy(y_test, y_pred), clf.misscount)