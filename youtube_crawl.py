import requests
import bs4 as BeautifulSoup
import json

def item_not_exist(item):
    if item == None:
        return True


def get_data(url):
    page = requests.get(url)
    text = page.text
    # print(text)
    obj = BeautifulSoup.BeautifulSoup(text, 'html.parser')

    title = obj.find('title').get_text()
    artist = obj.find_all('body')
    a = type(artist)
    print(a)
    # p = artist.f

    for link in artist:
        if '<a class' in link:
            print(link)
    # a = 'a' in artist
    #     print(link.get('a'))
    # print(artist)
    # creator = obj.find_all('span', class_ = 'submitter__name')[0].get_text()
    # rating = obj.find('div', class_ = 'rating-stars')
    # if item_not_exist(rating):
    #     rating = 'None'
    # else:
    #     rating = rating['data-ratingstars']
    #
    #
    # path_to_num_made_it = obj.find('a' ,class_="read--reviews")
    # if item_not_exist(path_to_num_made_it):
    #     num_made_it = 'None'
    # else:
    #     num_made_it = path_to_num_made_it.find_all('span')
    #     if item_not_exist(num_made_it):
    #         num_made_it = 'None'
    #     else:
    #         num_made_it = num_made_it[1].get_text().split('\xa0')[0]
    #
    #
    # num_reviews = obj.find_all('span', class_ = 'recipe-reviews__header--count')[0].get_text()
    #
    # num_photos = obj.find_all('span', class_='picture-count-link')
    # if item_not_exist(num_photos):
    #     num_photos = 'None'
    # else:
    #     num_photos = num_photos[0].get_text().split(' ')[0]
    #
    #
    # ingredients = []
    # ingredients_path = obj.find_all('span', class_ = 'recipe-ingred_txt added')
    # for item in ingredients_path:
    #     ingredients.append(item.get_text())
    #
    # directions = []
    # directions_path = obj.find_all('span', class_ = 'recipe-directions__list--item')
    # for item in directions_path:
    #     directions.append(item.get_text())
    #
    #
    # time = obj.find_all('time')
    # if len(time) == 0:
    #     prep_time = 'None'
    #     cook_time = 'None'
    #     ready_in = 'None'
    # elif len(time) == 1:
    #     prep_time = time[0].get_text()
    #     cook_time = 'None'
    #     ready_in = 'None'
    # elif len(time) == 2:
    #     prep_time = time[0].get_text()
    #     cook_time = time[1].get_text()
    #     ready_in = 'None'
    # else:
    #     prep_time = time[0].get_text()
    #     cook_time = time[1].get_text()
    #     ready_in = time[2].get_text()
    #
    # return {'Title: ': title, 'Creator: ': creator, 'Rating: ': rating, 'NumMadeIt: ':num_made_it,
    #         'NumReviews: ': num_reviews, 'NumPhotos: ': num_photos, 'Ingredients: ': ingredients,
    #         'Directions: ': directions, 'PrepTime: ': prep_time, 'CookTime: ': cook_time, 'ReadyIn: ': ready_in}


# data = []
# page_num = 1
# while True:
#     #url of the page num:
#     url = 'https://www.allrecipes.com/recipes/156/bread/?internalSource=top%20hubs&referringContentType=Homepage&page=' + str(page_num)
#     page = requests.get(url)
#     text = page.text
#
#     soup_obj = BeautifulSoup.BeautifulSoup(text, 'html.parser')
#
#     recipes = soup_obj.find_all('div', class_ = 'fixed-recipe-card__info')
#     # all recepies in specific page:
#     for recipe in recipes:
#         data.append(get_data(recipe.find('a')['href']))
#     if page_num == 150:
#         break
#     page_num += 1
#
#
# file  = open('json_file.txt', 'w')
# json.dump(data, file, indent=4)

get_data('https://www.youtube.com/watch?v=d2smz_1L2_0')