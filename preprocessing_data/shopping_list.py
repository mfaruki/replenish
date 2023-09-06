import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import string
import requests

all_cuisine_df = pd.read_csv('../raw_data/bbc_final_df.csv', index_col=0)

df = all_cuisine_df.reset_index()

def cleaner(text):
    strip_text = []
    text = text.replace(' ,', ' ') #.replace(',,', ',')

    new_lst = []
    for word in text.split():
        if word.isdigit():
            new_string = ', ' + word
            new_lst.append(new_string)
        else:
            new_lst.append(word)
            # for word in text.split():
    #     strip_text.append(word.strip())
    return ' '.join(new_lst)

def ing_list(sample_txt):
    elements = []
    # for punc in string.punctuation:
    #     sample_txt = sample_txt.replace(punc,'')
    for word in sample_txt.split(','):
        elements.append(str(word))
    new_lst = []

    for w in elements:
        if w=='' or w==' ':
            pass
        else:
            new_lst.append(w.strip())

    return new_lst

def remove_sw_sep(lst):
    sw_for_september = ['chopped','roughly','roughly chopped','crushed', 'and',
                    'dried','serve','split','zest and juice','deseeded and',
                    'low-salt','ready-to-eat dried','reduced-fat','serve','zest and juice','clear']
    no_stopwords = []
    for text in lst:
        if text not in sw_for_september:
            no_stopwords.append(text)
    return no_stopwords

def testing_api(ingredient_col):
    url = "https://zestful.p.rapidapi.com/parseIngredients"

    payload = { "ingredients": ingredient_col}
    headers = {
    	"content-type": "application/json",
    	"X-RapidAPI-Key": "4acb036e43mshcc9cdc7d8cd744dp1219d4jsn80eb52df264f",
    	"X-RapidAPI-Host": "zestful.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    test_dict = dict(response.json())


    product_lst = []
    quantity_lst = []
    unit_lst = []

    for ing in test_dict['results']:
        product=ing['ingredientParsed']['product']
        quantity=ing['ingredientParsed']['quantity']
        unit=ing['ingredientParsed']['unit']

        product_lst.append(product)
        quantity_lst.append(quantity)
        unit_lst.append(unit)

    final_dict = {'product': product_lst, 'quantity': quantity_lst, 'unit': unit_lst}


    return final_dict

def final_dataframe(recipe_index:list):
    df1 = pd.DataFrame(testing_api(df['modified'][recipe_index[0]]))
    df2 = pd.DataFrame(testing_api(df['modified'][recipe_index[1]]))
    df3 = pd.DataFrame(testing_api(df['modified'][recipe_index[2]]))
    final_df = pd.concat([df1, df2, df3])
    final_df2 = final_df.groupby('product').sum()[['quantity']].reset_index().sort_values('quantity', ascending=False)
    final_df3 = final_df2.merge(final_df, on='product', how='inner')
    final_df3 = final_df3.drop('quantity_y', axis=1)
    final_df3.drop_duplicates(inplace=True)
    final_df3['unit'].fillna(' ', inplace=True)recip   e_index

    return final_df3

def get_shopping_list(recipe_index:list) -> pd.DataFrame:
    df['modified'] = df['specific_ingredients'].apply(cleaner)
    df['modified'] = df['modified'].apply(ing_list)
    df['modified'] = df['modified'].apply(remove_sw_sep)
    final_df = final_dataframe(recipe_index)

    return final_df
