import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

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





def feature_extract():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tableName = os.path.join(dir_path, 'data\\sample-set.csv')
    try:
        df = pd.read_csv(tableName)
    except (IOError, pd.errors.EmptyDataError) as e:
        return


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
    for name in temp_list:
        variance = df[name].unique()
        # ignore low distinct features
        # threshold could be changed
        if len(variance) > 1 and len(variance) < len(df):
            feature_list.append(name)
        else:
            print('Non-distinguishable features: [{}]'.format(name),  len(variance))

    features = df[feature_list]

    return features, labels
    

if __name__ == '__main__':
    features, labels = feature_extract()
