import requests
from bs4 import BeautifulSoup
import re

cuisines = ['italian','indian','asian']
dietary= ['vegetarian','vegan', 'gluten-free','nut-free']

base_url= 'https://www.bbcgoodfood.com/search?tab=recipe'

def preference_based_search(preference=None):
    '''Return a url for a BBC GoodFood search that
    is specific to the input category/preference.'''
    if preference in cuisines:
        url= f'{baseurl}&cuisines={preference}'
    elif preference in dietary:
        url= f'{baseurl}&diet={preference}'
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
    individual_recipes = category_specific_links(preference_based_search(preference))

    #here the individual recipe links are pulled and iterated
    for recipe in individual_recipes:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, "html.parser")

        #recipe title
        recipe_title=[]

        #prep and cooking times
        prep_time = soup.find("div", class_= "icon-with-text__children").find_all('time')
        times=[prep_time[0].text.strip(' mins'), prep_time[1].text.strip(' mins')]


        #ingredients
        ingredients=[]
