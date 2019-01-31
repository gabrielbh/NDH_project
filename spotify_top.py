import requests
import bs4 as BeautifulSoup
import urllib.request
import re
import time
import json


def get_from_billboard(url, songs, artists):
    page = requests.get(url)
    obj = BeautifulSoup.BeautifulSoup(page.text, 'html.parser')
    a = obj.find_all(class_="chart-list-item__title-text")
    b = obj.find_all(class_="chart-list-item__artist")

    for i in range(len(a)):
        name = a[i].get_text().replace('\n', "")
        name = name.lower()
        art = b[i].get_text().replace('\n', "")
        art = art.lower()
        while name[0] == " ":
            name = name[1:]
        while art[0] == " ":
            art = art[1:]
        index = art.find('featuring')
        if index != -1:
            art = art[:index]
        if name not in songs and art not in artists:
            songs.append(name)
            artists.append(art)
    return songs, artists


# month_list = ['01', '04', '07', '10']
month_list = ['01']
songs = []
artists = []
# for year in range(2000, 2019):
for year in range(1958, 2012):
    print(year)
    for month in month_list:
        print(month)
        songs, artists = get_from_billboard('https://www.billboard.com/charts/hot-100/' + str(year) + '-' + month + '-01', songs, artists)
file = open('bilboard_1958_2012.txt', 'w')
for i in range(len(artists)):
    file.write(songs[i] +", " + artists[i] + '\n')
file.close()