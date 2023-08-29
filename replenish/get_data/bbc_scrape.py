import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from time import sleep

cuisines = ['italian','indian','asian','british','american','chinese']
dietary= ['vegetarian','vegan', 'gluten-free','nut-free','healthy']

base_url= 'https://www.bbcgoodfood.com/search?tab=recipe'

def preference_based_search(preference=None):
    '''Return a url for a BBC GoodFood search that
    is specific to the input category/preference.'''
    if preference in cuisines:
        url= f'{base_url}&cuisines={preference}'
    elif preference in dietary:
        url= f'{base_url}&diet={preference}'
    else:
        url=base_url
    return url

def category_specific_links(preference):
    '''Given a URL for a category search in BBC GoodFood,
    return a list of specific links for each recipe from
    the categorical search'''
    search_url = preference_based_search(preference)
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")
    recipe_links = []
    for link in soup.find_all("a", class_="link d-block"):
        recipe_links.append(f'https://www.bbcgoodfood.com/recipes{link["href"]}')
    return recipe_links

def category_bbc_data(preference):
    '''Return a dataframe of recipe information for each recipe
    given a specific category'''

    individual_recipes = category_specific_links(preference)

    #list for each item retrieved
    recipe_title=[]
    prep_times =[]
    cooking_times=[]
    stars=[]
    review_count=[]
    difficulty_level=[]
    servings=[]
    description=[]
    recipe_ingredients=[]

    #here the individual recipe links are pulled and iterated
    for recipe in individual_recipes:
        #time.sleep(1)
        response = requests.get(recipe)
        soup = BeautifulSoup(response.content, "html.parser")

        #recipe title
        try:
            title = soup.find("div", class_= "headline post-header__title post-header__title--masthead-layout").find(
            "h1",class_='heading-1').string
            recipe_title.append(title)
        except:
            recipe_title.append('n')

        #prep and cooking times
        try:
            times = soup.find("div", class_= "icon-with-text__children").find_all('time')
            prep_times.append(times[0].text.strip(' mins'))
            cooking_times.append(times[1].text.strip(' mins'))
        except:
            prep_times.append('n')
            cooking_times.append('n')

        try:
        #star ratings
            star_rating = soup.find("div", class_= "rating__values").find("span",class_='sr-only').string
            star_rating_float = float(star_rating.strip('A star rating of ').strip(' out of 5.'))
            stars.append(star_rating_float)
        except:
            stars.append('n')

        try:
        #review count
            num_reviews = soup.find("div", class_= "rating__values").find("span",class_='rating__count-text body-copy-small').string
            num_reviews_float = float(num_reviews.strip(' ratings'))
            review_count.append(num_reviews_float)
        except:
            review_count.append('n')

        #servings
        #if soup.find("div", class_= "icon-with-text post-header__servings body-copy-small body-copy-bold icon-with-text--aligned") not None:
        try:
            serving = soup.find("div", class_= "icon-with-text post-header__servings body-copy-small body-copy-bold icon-with-text--aligned").find("div", class_= "icon-with-text__children").string
        #servings_int = int(serving.strip('Serves '))
            servings.append(serving)
        except:
            servings.append("None")

        try:
        #description
            tagline = soup.find("div", class_= "editor-content mt-sm pr-xxs hidden-print").find("p").string
            description.append(tagline)
        except:
            description.append('n')

        try:

            #difficulty level
            difficulty = soup.find("div", class_= "icon-with-text post-header__skill-level body-copy-small body-copy-bold icon-with-text--aligned").find("div", class_= "icon-with-text__children").string
            difficulty_level.append(difficulty)
        except:
            difficulty_level.append('n')


        try:

            #ingredients
            for ingredient_group in soup.find_all('section', class_='recipe__ingredients col-12 mt-md col-lg-6'):
                ingredients = ingredient_group.find_all('li')
                ingredient_text = ''
            for ingredient in ingredients:
                ingredient_text += ingredient.get_text() + ', '
            recipe_ingredients.append(ingredient_text)
        except:
            recipe_ingredients.append('n')

        #Creating a Dictionary
    category_dictionary= {
        'recipe_title':recipe_title,
        'stars':stars,
        'prep_times':prep_times,
        'cooking_times': cooking_times,
        'review_count':review_count,
        'difficulty_level':difficulty_level,
        'servings':servings,
        'description':description,
        'ingredients':recipe_ingredients
    }

    #Dictionary to dataframe
    df = pd.DataFrame.from_dict(category_dictionary)

    return df

def get_image(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    image_url = soup.find('img')
    pass


def load_data():
    main_df = category_bbc_data(cuisines[0])

    all_df = []
    for cuisine in cuisines[1:]:
        df = category_bbc_data(cuisine)
        big_df = pd.merge(main_df,df, how='outer')
        all_df.append(big_df)
    final_df = pd.concat(all_df)

    return final_df
