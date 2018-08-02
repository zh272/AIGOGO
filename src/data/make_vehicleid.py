import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def get_niche_school_list():
    '''
    Input:
        ()
    Output:
        df_vehicleid
    Description:
        get the map of vehicles to replacement cost, damage cost id, and theft cost id
    '''
    url_raw = 'https://www.tii.org.tw/opencms/insurance/insurance5/queryResult.html'
    html = requests.get(url_raw)
    page = BeautifulSoup(html.content, 'html.parser')
    page.select('table')

    card = page.find('table')
    card = pd.DataFrame()
    print(html.content)

    df_school = pd.DataFrame(columns=['_id_name', 'url_niche'])
    pages = 340
    for i in range(1, pages + 1):
        time.sleep(0.25)
        if i == 1:
            suffix = ''
        else:
            suffix = '?page={}'.format(i)
        urli = ''.join([url_raw, suffix])
        html = requests.get(urli)
        page = BeautifulSoup(html.content, 'html.parser')
        cards = page.findAll('a', class_='card')
        for card in cards:
            grade = card.find('figure', class_='search-result-grade')
            grade = list(grade.children)[0].get_text()
            if grade != 'NG':
                link = card.find('a', class_='search-result__link')
                id_name = link.find('h2', 'search-result__title').get_text()
                id_name = id_name.replace('&apos;', "'")
                url_niche = link['href']
                row = { '_id_name' : id_name,
                        'url_niche' : url_niche }
                df_school = df_school.append(row, ignore_index=True)
        print('Page {} contains {} schools'.format(i, len(cards)))
    df_school = df_school.drop_duplicates()
    df_school.to_csv('school_list.csv', index = False)
    return(df_school)