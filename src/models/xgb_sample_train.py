import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgb_trainer import xgb_trainer

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
        self.le_list = {}

    def fit(self,X,y=None):
        if self.columns is None:
            for colname,col in X.items():
                self.le_list[colname] = preprocessing.LabelEncoder()
                self.le_list[colname].fit(col.astype(str))
        else:
            for colname in self.columns:
                self.le_list[colname] = preprocessing.LabelEncoder()
                self.le_list[colname].fit(X[colname].astype(str))
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for colname, le in self.le_list.items():
            try:
                output[colname] = le.transform(output[colname].astype(str))
            except ValueError:
                print('New Labels Found in column [{}]'.format(colname))
                if colname == "('Insurance_Coverage', 'Liability')":
                    print('')
                if colname == "Distribution_Channel":
                    print('')
                le.classes_ = np.append('0000000', le.classes_) # treat all unseen labels as on class
                classes = np.unique(output[colname].astype(str))
                diff = np.setdiff1d(classes, le.classes_)
                for value in diff:
                    # output[colname].iloc[output[colname][output[colname]==value]] = np.nan
                    indices = output[colname][output[colname].astype(str)==value].index.tolist()
                    for idx in indices:
                        output.at[idx, colname] = '0000000'
                output[colname] = le.transform(output[colname].astype(str))
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)





def feature_extract(train_path, test_path):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # trainName = os.path.join(dir_path, train_path)
    # testName = os.path.join(dir_path, test_path)
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except (IOError, pd.errors.EmptyDataError):
        return None


    # colNames = train_df.columns
    
    nonencode_feature = [
        'qpt', 'Replacement_cost_of_insured_vehicle', 'Multiple_Products_with_TmNewa_(Yes_or_No?)', 
        'lia_class', 'plia_acc', 'pdmg_acc', 'Manafactured_Year_and_Month', "('Insured_Amount1', 'Damage')", 
        "('Insured_Amount1', 'Liability')", "('Insured_Amount1', 'Theft')", 
        "('Insured_Amount2', 'Damage')", "('Insured_Amount2', 'Liability')", 
        "('Insured_Amount2', 'Theft')", "('Insured_Amount3', 'Damage')", 'Engine_Displacement_(Cubic_Centimeter)',
        "('Insured_Amount3', 'Liability')", "('Insured_Amount3', 'Theft')", 
        "('Coverage_Deductible_if_applied', 'Damage')", "('Coverage_Deductible_if_applied', 'Liability')", 
        "('Coverage_Deductible_if_applied', 'Theft')", "('Premium', 'Damage')", "('Premium', 'Liability')", 
        "('Premium', 'Theft')", "('Paid_Loss_Amount', 'Damage')", "('Paid_Loss_Amount', 'Liability')", 
        "('Paid_Loss_Amount', 'Theft')", "('paid_Expenses_Amount', 'Damage')", "('Insurance_Coverage', 'Damage')", 
        "('Insurance_Coverage', 'Liability')", "('Insurance_Coverage', 'Theft')", 
        "('paid_Expenses_Amount', 'Liability')", "('paid_Expenses_Amount', 'Theft')", 
        "('Salvage_or_Subrogation?', 'Damage')", "('Salvage_or_Subrogation?', 'Liability')", 
        "('Salvage_or_Subrogation?', 'Theft')", "('At_Fault?', 'Damage')", 
        "('At_Fault?', 'Liability')", "('At_Fault?', 'Theft')", "('Deductible', 'Damage')", 
        "('Deductible', 'Liability')", "('Deductible', 'Theft')", "('number_of_claimants', 'Damage')", 
        "('number_of_claimants', 'Liability')", "('number_of_claimants', 'Theft')"
    ]
    encodeCols = [
        "Insured's_ID", 'Prior_Policy_Number', 'Cancellation', 'Vehicle_identifier', 
        'Vehicle_Make_and_Model1', 'Vehicle_Make_and_Model2',  
        'Imported_or_Domestic_Car', 'Coding_of_Vehicle_Branding_&_Type', 'fpt', 'Distribution_Channel', 
        'fassured', 'ibirth', 'fsex', 'fmarriage', 'aassured_zip', 'iply_area', 'dbirth', 
        'fequipment1', 'fequipment2', 'fequipment3', 'fequipment4', 'fequipment5', 'fequipment6', 
        'fequipment9', 'nequipment9', 
        "('Claim_Number', 'Damage')", "('Claim_Number', 'Liability')", "('Claim_Number', 'Theft')"
    ]

    # temp_list = encodeCols + nonencodeCols
    # temp_list = list(colNames)
    encode_feature = []
    print('===== {:20s} ====='.format('Disgarded Features'))
    for name in encodeCols:
        variance = train_df[name].unique()
        # ignore low distinct features
        # threshold could be changed
        if len(variance) > 1 and len(variance) < 0.7*len(train_df):
            encode_feature.append(name)
        else:
            print('[{:50s}]'.format(name),  len(variance))

    feature_list = encode_feature + nonencode_feature
    train_df[nonencode_feature].fillna(0)
    train_df = train_df[feature_list]
    test_df = test_df[feature_list]

    le = MultiColumnLabelEncoder(columns=encode_feature)
    le.fit(pd.concat([train_df, test_df]))
    train_features = le.transform(train_df)
    test_features = le.transform(test_df)

    return train_features, test_features
    

if __name__ == '__main__':
    # random.seed(100)
    np.random.seed(103)
    # torch.manual_seed(101)
    # torch.cuda.manual_seed_all(102)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, os.path.pardir, os.path.pardir, 'data', 'raw')

    y_train_path = os.path.join(data_path,'y_train.csv')
    y_train = pd.read_csv(y_train_path)
    y_test_path = os.path.join(data_path,'testing-set.csv')
    y_test = pd.read_csv(y_test_path)

    # # X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values, test_size=0.25, random_state=0)
    X_train, X_test = feature_extract(os.path.join(data_path,'X_train.csv'), os.path.join(data_path,'X_test.csv'))

    params = {
        'n_estimators':1000, 'learning_rate':0.2, 'objective':'reg:linear',
        'max_delta_step':0, 'max_depth':10, 'gamma':0.0, 'subsample':0.5, 
        'colsample_bytree':0.7, 'colsample_bylevel':0.8, 'reg_alpha':0.0, 'reg_lambda':1.0
    }

    train_pred, test_pred, feature_importances = xgb_trainer(X_train.values, y_train['Next_Premium'].values, X_test.values, y_test['Next_Premium'].values, params=params)
    
    feature_names = X_train.columns
    sorted_idx = np.argsort(-feature_importances) # descending order
    print('===== feature importances =====')
    for idx in sorted_idx:
        print('[{:50s}]'.format(feature_names[idx]), '{:7.4f}'.format(feature_importances[idx]))

    predictions = pd.DataFrame({'Policy_Number':y_test['Policy_Number'],'Next_Premium':test_pred})
    write_path = os.path.join(dir_path, os.path.pardir, os.path.pardir, 'data', 'testing-set.csv')
    predictions.to_csv(write_path, index=False, columns=["Policy_Number", "Next_Premium"])



    # X_train = pd.read_csv(os.path.join(dir_path, '..\\..\\data\\raw\\X_train.csv'))
    # train_pred = np.nansum(X_train[["('Premium', 'Damage')", "('Premium', 'Liability')", "('Premium', 'Theft')"]].values, axis=1)
    
    # X_test = pd.read_csv(os.path.join(dir_path, '..\\..\\data\\raw\\X_test.csv'))
    # test_pred = np.nansum(X_test[["('Premium', 'Damage')", "('Premium', 'Liability')", "('Premium', 'Theft')"]].values, axis=1)
    
    # print(mean_absolute_error(y_train['Next_Premium'].values, train_pred))
    
    
    
