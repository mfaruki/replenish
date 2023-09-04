from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt

bbc_final_df = pd.read_csv('/home/jay_mistry/code/mfaruki/replenish/raw_data/bbc_final_df.csv')

### DIETARY COLUMN TRANSFORMER

def dietary_functon(bbc_final_df):
    """Takes the raw data from bbc_final_df as an input and cleans the data
    1. Drops null rows for recipe names
    2. Removes duplicate recipes due to them having different dietary values (converts to one row per recipe)
    """

    #Dropping null rows - ie. n for recipe name
    filtered_df = bbc_final_df[bbc_final_df['recipe_title']!='n']

    # Removing duplicates by combining the dietary column
    grouped_df = filtered_df.groupby('recipe_title').sum()[['dietary']].reset_index()
    grouped_df.rename(columns = {'dietary':'combined'}, inplace=True)

    # Joining combined dietary column to original dataframe and dropping duplicates
    merged_df = grouped_df.merge(filtered_df,how='left',on='recipe_title')
    dropped_df = merged_df.drop(columns = 'dietary')
    df = dropped_df.drop_duplicates('recipe_title')
    return df

df = dietary_functon(bbc_final_df)

### FINAL INGREDIENTS COLUMN TRANSFORMER

def ingredients_list(sample_txt):
    """Converts the string of ingredients into a list of unique ingredients"""
    elements = []
    for punc in string.punctuation:
        sample_txt = sample_txt.replace(punc,'')
    for word in sample_txt.split():
        elements.append(str(word))
    return list(set(elements))

def model_specific_preprocessing(tokenized_sentence):
    """Stopwords and lemmatizing the ingredients list - outputs a 'sentence' of ingredients"""
    stop_words = set(stopwords.words('english'))
    no_stopwords = [w for w in tokenized_sentence if not w in stop_words]
    verb_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "v") for tok in no_stopwords]
    noun_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "n")for tok in verb_lemmatized]
    ingredient_string = ' '.join(noun_lemmatized)
    return ingredient_string

def final_ingredients_function(df):
    """Applies relevant preprocessing functions to the ingredients column to prepare for vectorising"""
    df['ingredients_list'] = df['final_ingredients'].apply(ingredients_list)
    df['clean_text']=df['ingredients_list'].map(lambda x: model_specific_preprocessing(x))
    return df

df = final_ingredients_function(df)

### VECTORISER AND MODEL WITH OPTIMIZED PARAMETERS

############### Optimized params #################
min_df = 0.00001
max_df = 0.3
clusters = 75

def final_model(df, min_df, max_df, clusters):
    """Applies K-means clustering to the cleaned list of ingredients from df (on 'clean_text' column)
    Returns the df with clusters applied in a 'cluster' column
    """
    vectorizer = CountVectorizer(min_df=min_df, max_df = max_df)
    counted_words = pd.DataFrame(vectorizer.fit_transform(df['clean_text']).toarray(), columns = vectorizer.get_feature_names_out())
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(counted_words)
    return df

model_df = final_model(df, min_df, max_df, clusters)

### CUSTOM SCORING METRICS FOR MODEL FINETUNING

# 1. Cluster size metric - want to minimise
def cluster_size_metric(model_df):
    cluster_size = model_df.groupby('cluster').count()[['recipe_title']] / len(model_df)
    equal_cluster_size = 1 / model_df['cluster'].nunique()
    cluster_size_score = abs(cluster_size - equal_cluster_size).mean()
    return cluster_size_score[0]

# 2. Cluster cuisine distribution metric - want to minimise
def cluster_cuisine_metric(model_df):
    all_cuisine = model_df.groupby('preference').count()[['recipe_title']] / len(model_df)
    cuisine_errors = []
    for i in range(model_df['cluster'].nunique()):
        cluster_cuisine = model_df[model_df['cluster']==i].groupby(['preference']).count()[['recipe_title']] / len(model_df[model_df['cluster']==i])
        cuisine_diff = all_cuisine - cluster_cuisine
        cuisine_errors.append(abs(cuisine_diff.fillna(all_cuisine)).sum()[0])
    cluster_cuisine_score = np.mean(cuisine_errors)
    return cluster_cuisine_score

# 3. Clusters with least variation of ingredients per recipe - want to maximise
def cluster_ingredient_similarity_metric(model_df):
    similarity_scores = []
    for i in range(model_df['cluster'].nunique()):
        ingredients_cluster = model_df[model_df['cluster']==i]['clean_text']
        ingredients_cluster_list = ingredients_cluster.tolist()
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(ingredients_cluster_list)
        similarity = cosine_similarity(vectors)
        similarity_scores.append(similarity.mean())
    ingredient_similarity_score = np.mean(similarity_scores)
    return ingredient_similarity_score

### MANUAL GRIDSEARCH AND GRAPHS

#min_df = [0.00001, 0.0001, 0.001]
#max_df = [0.25, 0.3, 0.35]
#n_clusters = [50, 75, 100]

def manual_gridsearch(df,min_df,max_df,n_clusters):
    """Applies K-means clustering to the cleaned list of ingredients from df (on 'clean_text' column)
    Loops through list of vlaues for min_df, max_df and n_clusters
    Returns a DataFrame of each combination of parameters and their respective custom metric scores
    """
    params = {}
    for cluster in n_clusters:
        for max in max_df:
            for min in min_df:
                vectorizer = CountVectorizer(min_df=min, max_df = max)
                counted_words = pd.DataFrame(vectorizer.fit_transform(df['clean_text']).toarray(), columns = vectorizer.get_feature_names_out())
                num_features = counted_words.shape[1]
                kmeans = KMeans(n_clusters=cluster, random_state=0)
                df['cluster'] = kmeans.fit_predict(counted_words)
                params[f'min={str(min)},max={str(max)},clusters={str(cluster)}'] = {'cluster':cluster,
                                                                                    'min':min,
                                                                                    'max':max,
                                                                                    'num_features':num_features,
                                                                                    'cluster_size_metric':cluster_size_metric(df),
                                                                                    'cluster_cuisine_metric':cluster_cuisine_metric(df),
                                                                                    'cluster_ingredient_similarity_metric':cluster_ingredient_similarity_metric(df),
                                                                                    }
    params_df = pd.DataFrame(params).transpose()
    return params_df

def gridsearch_graphs(params_df):
    params_df_cluster = params_df.sort_values('cluster')
    params_df_min = params_df.sort_values('min')
    params_df_max = params_df.sort_values('max')

    fig, ax = plt.subplots(3, 3, figsize=(20,20))

    ax[0,0].plot(params_df_cluster['cluster_size_metric'])
    ax[0,0].set_title('By cluster: Size Dispersion (to minimise)')
    ax[0,0].set_xticklabels(params_df_cluster['cluster'],rotation=90)
    ax[0,1].plot(params_df_cluster['cluster_cuisine_metric'])
    ax[0,1].set_title('By cluster: Cuisine Dispersion (to minimise)')
    ax[0,1].set_xticklabels(params_df_cluster['cluster'],rotation=90)
    ax[0,2].plot(params_df_cluster['cluster_ingredient_similarity_metric'])
    ax[0,2].set_title('By cluster: Ingredient Similarity (to maximise)')
    ax[0,2].set_xticklabels(params_df_cluster['cluster'],rotation=90)

    ax[1,0].plot(params_df_min['cluster_size_metric'])
    ax[1,0].set_title('By min_df: Size Dispersion (to minimise)')
    ax[1,0].set_xticklabels(params_df_min['min'],rotation=90)
    ax[1,1].plot(params_df_min['cluster_cuisine_metric'])
    ax[1,1].set_title('By min_df: Cuisine Dispersion (to minimise)')
    ax[1,1].set_xticklabels(params_df_min['min'],rotation=90)
    ax[1,2].plot(params_df_min['cluster_ingredient_similarity_metric'])
    ax[1,2].set_title('By min_df: Ingredient Similarity (to maximise)')
    ax[1,2].set_xticklabels(params_df_min['min'],rotation=90)

    ax[2,0].plot(params_df_max['cluster_size_metric'])
    ax[2,0].set_title('By max_df: Size Dispersion (to minimise)')
    ax[2,0].set_xticklabels(params_df_max['max'],rotation=90)
    ax[2,1].plot(params_df_max['cluster_cuisine_metric'])
    ax[2,1].set_title('By max_df: Cuisine Dispersion (to minimise)')
    ax[2,1].set_xticklabels(params_df_max['max'],rotation=90)
    ax[2,2].plot(params_df_max['cluster_ingredient_similarity_metric'])
    ax[2,2].set_title('By max_df: Ingredient Similarity (to maximise)')
    ax[2,2].set_xticklabels(params_df_max['max'],rotation=90)

    plt.show()
