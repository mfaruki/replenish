from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import string
import numpy as np

### PREPROCESSING FOR KMEANS
def model_specific_preprocessing(tokenized_sentence):
    stop_words = set(stopwords.words('english'))
    no_stopwords = [w for w in tokenized_sentence if not w in stop_words]
    verb_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "v") for tok in no_stopwords]
    noun_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "n")for tok in verb_lemmatized]
    ingredient_string = ' '.join(noun_lemmatized)
    return ingredient_string

### PREPROCESSING FOR INGREDIENTS
def ingredients_list(sample_txt):
    elements = []
    for punc in string.punctuation:
        sample_txt = sample_txt.replace(punc,'')
    for word in sample_txt.split():
        elements.append(str(word))
    return list(set(elements))

### KMEANS MODEL
def k_means(bbc_final_df, clusters=10, min_df=0.005, max_df=0.99):

    """Returns the df inputted with an additional column with their respective cluster.
    Ensure that the ingredients columns is a string of a list called 'final_ingredients'.
    I.E row 1, col ingredient is '[bread,cheese,pasta,onions]'"""

    # Step 1: Filter out recipes with no names & remove duplicates, keeping dietary info
    filtered_df = bbc_final_df[bbc_final_df['recipe_title']!='n']
    grouped_df = filtered_df.groupby('recipe_title').sum()[['dietary']].reset_index()
    grouped_df.rename(columns = {'dietary':'combined'}, inplace=True)
    merged_df = grouped_df.merge(filtered_df,how='left',on='recipe_title')
    dropped_df = merged_df.drop(columns = 'dietary')
    df = dropped_df.drop_duplicates('recipe_title')

    # Step 2: Convert the final ingredients into a "sentence" for vectorising
    df['ingredients_list'] = df['final_ingredients'].apply(ingredients_list)
    df['clean_text']=df['ingredients_list'].map(lambda x: model_specific_preprocessing(x))

    # Step 3: Convert ingredients sentence into a bag-of-words representation
    vectorizer = CountVectorizer(min_df=min_df, max_df = max_df)
    counted_words = pd.DataFrame(vectorizer.fit_transform(df['clean_text']).toarray(), columns = vectorizer.get_feature_names_out())

    # Step 4: Apply KMeans clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(counted_words)

    return df

### CUSTOM MODEL SCORING METRICS

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
