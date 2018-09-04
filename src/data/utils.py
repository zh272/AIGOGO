import os
import pandas as pd

def read_data(file_name, path='interim', index_col='Policy_Number'):
    '''
    In: file_name
    Out: interim_data
    Description: read data from directory /data/path
    '''
    # set the path of raw data
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', path
    )

    file_path = os.path.join(data_path, file_name)
    data = pd.read_csv(file_path, index_col=index_col)

    return data

def write_data(df, file_name, path='interim'):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None
    Description:
        Write sample data to directory /data/path
    '''
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', path
    )
    write_path = os.path.join(data_path, file_name)
    df.to_csv(write_path)