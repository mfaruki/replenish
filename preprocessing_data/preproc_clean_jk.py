from replenish.get_data import bbc_scrape
from preprocessing_data.preproc_jedijoni import time_cleaner, serving_cleaner

cuisines = ['italian', 'indian', 'asian', 'british', 'american', 'chinese']
dietary = ['vegetarian', 'vegan', 'gluten-free', 'nut-free', 'healthy']

preference_list = []
preference_list.extend(cuisines)
preference_list.extend(dietary)

def recipe_preproc(preference_list):
    for preference in preference_list:
        data = bbc_scrape.category_bbc_data(preference)
        df_clean = data[data['servings'] != 'None']
        df_clean['prep_times'] = df_clean['prep_times'].apply(time_cleaner)
        df_clean['cooking_times'] = df_clean['cooking_times'].apply(time_cleaner)
        df_clean['servings'] = df_clean['servings'].apply(serving_cleaner)
    return df_clean
