import time
import requests
import numpy as np
import os
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

def get_external_vehicleid():
    '''
    Input:
        None
    Output:
        df_vehicleid
    Description:
        get the map of vehicles to replacement cost, damage cost id, and theft cost id
    '''
    # Get vehicle model info
    num_page = 761
    url_raw = r'https://www.tii.org.tw/opencms/insurance/insurance5/queryResult.html?pageCount=761&dataclass=A&carsclass=A&cars=%u5C0F%u5BA2%u8ECA&dyear=1900&dmonth=08&ddate=06&pageIndex='
    vehicleid_list = []
    for i in range(1, num_page + 1):
        print('Getting page {}'.format(i))
        url = url_raw + str(i)
        time.sleep(0.25)
        html = requests.get(url)
        page = BeautifulSoup(html.content, 'html.parser')
        card = page.find('table')
        vehicleid = pd.read_html(str(card))[0]
        vehicleid.columns = ['Code', 'Model', 'Rep_Cost', 'ID_dmg', 'ID_thf', 'Date1', 'Note', 'Date2']
        vehicleid_list.append(vehicleid)

    df_vehicleid = pd.concat(vehicleid_list)

    return(df_vehicleid)


def write_vehicle_data(df, file_name):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None

    Description:
        Write sample data to directory /data/interim
    '''
    interim_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'interim')
    write_sample_path = os.path.join(interim_data_path, file_name)
    df.to_csv(write_sample_path, encoding='utf_8_sig')

    return(None)

if __name__ == '__main__':
    #df_vehicleid = get_external_vehicleid()
    write_vehicle_data(df_vehicleid, 'vehicleid.csv')