
import requests
from bs4 import BeautifulSoup
import re
from replenish.get_data import bbc_scrape
import time
import pandas as pd

cuisines = ['afghan', 'african', 'american','australian', 'asian','austrian',
            'azerbaijan', 'balinese', 'belgian','brazilian','british','cajun-creole',
            'caribbean', 'chinese','cuba','danish', 'dinner', 'eastern-european', 'egyptian', 'english',
            'finland','french', 'german', 'greek', 'hungarian', 'indian', 'indonesian', 'irish', 'italian', 'jamaican',
            'japanese', 'jewish', 'korean', 'latin-american', 'lithuanian', 'mediterranean', 'mexican',
            'middle-eastern', 'moroccan', 'nepalese', 'nigerian', 'north-african', 'persian', 'peruvian', 'polish', 'portuguese',\
            'scandinavian', 'senegalese', 'scottish', 'southern-soul', 'spanish', 'swedish', 'swiss',
            'taiwanese', 'thai', 'tunisian', 'turkish', 'ukrainian', 'vietnamese', 'welsh', 'balkan', 'czech', 'czech-cuisine']

def nutrition_data(preference):
    '''Return a dataframe of nutritional information for each recipe
    given a specific category'''

    individual_recipes = bbc_scrape.category_specific_links(preference)

    #list for each item retrieved
    recipe_title=[]
    kcal=[]
    fats:[]
    saturates=[]
    carbs=[]
    sugars=[]
    fibre=[]
    protein=[]
    salt=[]

    #here the individual recipe links are pulled and iterated
    for recipe in individual_recipes:
        time.sleep(0.25)
        response = requests.get(recipe, allow_redirects=False)
        soup = BeautifulSoup(response.content, "html.parser")

        #scraping recipe title
        try:
            title = soup.find("div", class_= "headline post-header__title post-header__title--masthead-layout").find(
            "h1",class_='heading-1').string
            recipe_title.append(title)
        except:
            recipe_title.append('n')

        #scraping nutrition values and putting into a dictionary
        try:

            nutrition_per_serving = soup.find(
                "table", class_= "key-value-blocks hidden-print mt-xxs")

            nutrition_per_serving_all = nutrition_per_serving.find_all(
                "tbody", class_= "key-value-blocks__batch body-copy-extra-small")

            nutrition_per_serving_category = []
            for table in nutrition_per_serving_all:
                nutrition_per_serving_category.append(table.find_all("tr", class_= "key-value-blocks__item"))

            nutrition_dict = {}
            for i in range(2):
                for j in range(4):
                    nutrition_type = nutrition_per_serving_category[i][j].find("td", class_= "key-value-blocks__key").string
                    nutrition_value = nutrition_per_serving_category[i][j].find("td", class_= "key-value-blocks__value").text.strip('g')
                    nutrition_dict[nutrition_type] = float(nutrition_value)

            kcal.append(nutrition_dict['kcal'])
            fats.append(nutrition_dict['kcal'])
            saturates.append(nutrition_dict['saturates'])
            carbs.append(nutrition_dict['carbs'])
            sugars.append(nutrition_dict['sugars'])
            fibre.append(nutrition_dict['fibre'])
            protein.append(nutrition_dict['protein'])
            salt.append(nutrition_dict['salt'])
        except:
            kcal.append('n')
            fats.append('n')
            saturates.append('n')
            carbs.append('n')
            sugars.append('n')
            fibre.append('n')
            protein.append('n')
            salt.append('n')

    #Dictionary of nutritional values and associated recipe
    nutrition_dictionary = {'recipe_title':recipe_title,
                            'kcal':kcal,
                            'fat':fats,
                            'saturates':saturates,
                            'carbs':carbs,'sugars':sugars,
                            'fibre':fibre, 'protein':protein,
                            'salt':salt}
    return pd.DataFrame.from_dict(nutrition_dictionary)

def get_nutrition():
    '''scrape and return a data frame with recipe title
    and nutrition values for each recipe'''

    print("Scraping for nutrition from BBC Good Foods")
    nutrition_all_df_lst = []

    for cuisine in cuisines:
        df = nutrition_data(cuisine)
        nutrition_all_df_lst.append(df)

    print('Scraping done')

    final_df = pd.concat(nutrition_all_df_lst)
    final_df.reset_index(inplace=True)

    print('Finished scraping all')

    return final_df
