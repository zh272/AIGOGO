import numpy as np
import os
import pandas as pd

def contour_observe(df,n2sflag=False):
    '''
        In: data in dataframe
        Out: contour data in dataframe
             int(Totalcol)
        Description: see contour for each column and get total number of the columns
    '''
    def rev_numeric_to_cat(df):
        '''
        In: df
        Out:df
        Description: convert all nemrical value to string
        '''
        #print(df.dtypes)
        for col in df.columns:
            if str(df[col].dtypes) != 'object': #'int' in str(df[col].dtypes):
                df[col]=df[col].astype('str', copy=False)
                #print(df[col].dtypes)
    
        return df

    # convert all nemrical value to string, (which we regard the value as a category) 
    if n2sflag:
        df=rev_numeric_to_cat(df)
    #get number of columns
    col_name=df.columns
    Totalcol=len(col_name)
    #get contour for each column
    contour=df.describe(include='all')

    return contour,Totalcol


def relation_observe(df,grp_col,agg_col):
    '''
        In: data in dataframe,
            str(grp_col)
            str(agg_col)
        Out: 
        Description: read data from directory /data/interim
    '''

    # normal frequency of category
    #dfcol_aggfreq = df.groupby(by=grp_col).agg({agg_col: lambda x: len(x)})
    # unique frequency of category
    dfcol_aggfreq = df.groupby(by=grp_col).agg({agg_col: lambda x: len(np.unique(x))})
    # map agg_col's element value to grp_col column
    #result = df[grp_col].map(dfcol_aggfreq[agg_col])

    if agg_col!='Policy_Number':
        dfcol_aggfreq['Policy_Number_freq'] = df.groupby(by=grp_col).agg({'Policy_Number' : lambda x: len(np.unique(x))})

    return dfcol_aggfreq

def uq_value_observe(df,col):
    '''
    In:
        DataFrame(df_policy),
        str(col),
    Out:
        DataFrame(dfzero) which is 
    Description:
        get policy_number with its number index which has Nan value in correpsonding column
    '''
    df_col_uni=df[col].unique()

    return df_col_uni,len(df_col_uni)

def nan_observe(df, col):
    '''
    In:
        DataFrame(df_policy),
        str(col),
    Out:
        DataFrame(dfzero) which is 
    Description:
        get policy_number with its number index which has Nan value in correpsonding column
    '''
    #### agg df
    df = df.groupby(level=0).agg({col: lambda x: x.iloc[0]})
    df = df.reset_index()
    #df_nan = df.loc[df[col]!=df.loc[[i for i in df.index],col]]#way1: judge if the value is NaN (if nan, the value would not match itself.)
    df_nan=df.loc[df.loc[:,col].isnull()]#way2

    # #### df_policy
    # df =df_policy.reset_index()
    # df_nan=df.loc[df.loc[:,col].isnull()]

    return df_nan


    


####################   I/O handling     ####################
def read_raw_data(file_name):#, index_col='Policy_Number'):
    '''
    In: file_name
    Out: raw_data
    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    if os.getcwd()[-1]=='O':
        raw_data_path = os.path.join(os.path.dirname('__file__'), 'data', 'raw') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
        raw_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'raw')

    file_path = os.path.join(raw_data_path, file_name)
    raw_data = pd.read_csv(file_path)#, index_col=index_col)

    return(raw_data)

def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name
    Out: interim_data
    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    if os.getcwd()[-1]=='O':
        interim_data_path = os.path.join(os.getcwd(), 'data', 'interim') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
        interim_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'interim')

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    return(interim_data)


def write_obs_data(df, file_name):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None
    Description:
        Write sample data to directory /data/interim
    '''
    if os.getcwd()[-1]=='O':
        interim_data_path = os.path.join(os.getcwd(), 'data', 'proceed') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
        interim_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'interim')

    write_sample_path = os.path.join(interim_data_path, file_name)
    df.to_csv(write_sample_path)
 
    return(None)


if __name__ == '__main__':
    

    # df_train_raw = read_raw_data('training-set.csv')
    # df_test_raw = read_raw_data('testing-set.csv')
    # df_claim_raw = read_raw_data('claim_0702.csv')
    df_policy_raw = read_raw_data('policy_0702.csv')
    #df_train = read_interim_data('X_train_bs.csv')
    # df_test = read_interim_data('X_test_bs.csv')
    # df_valid = read_interim_data('X_valid_bs.csv')

    

    ### contour observe section ###
    # if not os.path.isdir('contour_obs'):
    #     os.mkdir('contour_obs')
    # contour,total_col=contour_observe(df_train)
    # #contour.to_csv('feature_obs_train.csv')
    # print(total_col)#32
    # contour,total_col=contour_observe(df_claim_raw)
    # contour.to_csv('feature_obs_rawclaim.csv')
    # print(total_col)#19
    # contour,total_col=contour_observe(df_policy_raw)
    # contour.to_csv('feature_obs_rawpolicy.csv')
    # print(total_col)#40 (if print 41, additional 1 is index which is 'Policy_Number')

    # contour,total_col=contour_observe(df_claim_raw,n2sflag=True)
    # contour.to_csv('feature_obs_rawclaim_numToCat.csv')
    # print(total_col)#20
    # contour,total_col=contour_observe(df_policy_raw,n2sflag=True)
    # contour.to_csv('feature_obs_rawpolicy_numToCat.csv')
    # print(total_col)#41 (if print 41, additional 1 is index which is 'Policy_Number')

    
    # ### column relationship observation ###
    # if not os.path.isdir('column_obs'):
    #     os.mkdir('column_obs')
    # relation= relation_observe(df_policy_raw,'Vehicle_identifier','Policy_Number')
    # relation.to_csv(r'.\column_obs\feature_obs_VIToPN.csv')
    # relation= relation_observe(df_policy_raw,'Policy_Number','Vehicle_identifier')
    # relation.to_csv(r'.\column_obs\feature_obs_PNToVI.csv')
    # relation= relation_observe(df_policy_raw,'Manafactured_Year_and_Month','Policy_Number')
    # relation.to_csv(r'.\column_obs\feature_obs_MYnMToPN.csv')
    # relation= relation_observe(df_policy_raw,'Coding_of_Vehicle_Branding_&_Type','Vehicle_Make_and_Model2')
    # relation.to_csv(r'.\column_obs\feature_obs_CoVBnTToVMnM2.csv')
    # relation= relation_observe(df_policy_raw,'Vehicle_Make_and_Model2','Coding_of_Vehicle_Branding_&_Type')
    # relation.to_csv(r'.\column_obs\feature_obs_VMnM2ToCoVBnT.csv')
    relation=relation_observe(df_policy_raw,'Insurance_Coverage','Policy_Number')
    relation.to_csv(r'.\column_obs\feature_obs_ICToPN.csv')

    ### column value observation ###
    # type of Insurance_Coverage
    df_col_uni,total_uni_value=uq_value_observe(df_policy_raw,'Insurance_Coverage')
    print(total_uni_value)
    print(np.sort(df_col_uni))
    

    ### speical record section ###
    #type1=['00I','01A','01J','02K','03L','04M','05E','06F','07P','08H','14N','20B','20K','32N','33F','33O','34P','35H','36I','45@','51O','55J','56B','56K','57C','66C','66L','67D']
    #type2=['05N','09@','09I','10A','68E','68N']
    #type3=['12','14E','15F','15O','16G','16P','18@','18I','25G','26H','27I','29B','29K','37J','40M','41E','41N','42F','46A','47B','57L','65K','70G','70P','71H','72@']