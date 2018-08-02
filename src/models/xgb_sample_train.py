import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from xgb_trainer import xgb_trainer

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # output[col] = preprocessing.LabelEncoder().fit_transform(output[col].fillna('0'))
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)





def feature_extract(data_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tableName = os.path.join(dir_path, data_path)
    try:
        df = pd.read_csv(tableName)
    except (IOError, pd.errors.EmptyDataError):
        return None, None


    # colNames = list(df)
    colNames = df.columns
    labels = df[colNames[1]] # Next_Premium
    
    nonencodeCols = list(colNames[[
        12,14,16,17,18,19,38,39,40,41,42,43,44,45,
        46,47,48,49,50,51,52,53,54,55,56,57,58,59,
        60,61,62,63,64,65,66,67,68,69,70
    ]])
    encodeCols = list(colNames[[
        2,3,4,5,6,7,8,9,10,11,13,15,20,21,22,23,
        24,25,26,27,28,29,30,31,32,33,34,35,36,37,
        71,72,73

    ]])
    
    le = MultiColumnLabelEncoder(columns=encodeCols)
    df = le.fit_transform(df)

    temp_list = encodeCols+nonencodeCols
    feature_list = []
    print('===== {:20s} ====='.format('Disgarded Features'))
    for name in temp_list:
        variance = df[name].unique()
        # ignore low distinct features
        # threshold could be changed
        if len(variance) > 1 and len(variance) < len(df):
            feature_list.append(name)
        else:
            print('[{:50s}]'.format(name),  len(variance))

    features = df[feature_list]

    return features, labels
    

if __name__ == '__main__':
    # random.seed(100)
    np.random.seed(103)
    # torch.manual_seed(101)
    # torch.cuda.manual_seed_all(102)

    data_path = '..\\..\\zhiyong\\data\\sample-set.csv'
    features, labels = feature_extract(data_path)
    if features is None:
        quit()
    X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values, test_size=0.25, random_state=0)

    params = {
        'n_estimators':100, 'learning_rate':0.1, 'objective':'reg:linear',
        'max_delta_step':5, 'max_depth':3, 'gamma':0.0, 'subsample':0.7, 
        'colsample_bytree':0.5, 'colsample_bylevel':1.0, 'reg_alpha':0.0, 'reg_lambda':2.0
    }

    train_pred, test_pred, feature_importances = xgb_trainer(X_train, y_train, X_test, y_test, params=params)

    feature_names = features.columns
    sorted_idx = np.argsort(-feature_importances) # descending order
    print('===== feature importances =====')
    for idx in sorted_idx:
        print('[{:50s}]'.format(feature_names[idx]), '{:7.4f}'.format(feature_importances[idx]))
