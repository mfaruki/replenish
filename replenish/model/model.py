from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import string



def model_specific_preprocessing(tokenized_sentence):
    stop_words = set(stopwords.words('english'))
    no_stopwords = [w for w in tokenized_sentence if not w in stop_words]
    #new = ' '.join(no_stopwords)

    #token_sentence= word_tokenize(new)
    # Lemmatizing the verbs
    verb_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "v") for tok in no_stopwords]
    # 2 - Lemmatizing the nouns
    noun_lemmatized = [WordNetLemmatizer().lemmatize(tok, pos = "n")for tok in verb_lemmatized]

    ingredients = []
    for word in noun_lemmatized:
        ingredients.append(word.replace(" ","_"))
    final_ingredients = ' '.join(ingredients)
    return final_ingredients

#KMEANS CLUSTERING

def k_means(df, name_of_ingredient_column='ingredients', clusters=10):

    """Returns a dataset of recipes, each with their respective cluster.
    Ensure that the ingredients columns regards lists of ingredients.
    I.E row 1, col ingredient is [bread,cheese,pasta,onions]"""

    # Step 1: Preprocess the ingredients - to be changed

    ingredients_list = df['ingredients'].tolist()

    # Step 2: Convert ingredients into a bag-of-words representation
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=0.005)
    X = vectorizer.fit_transform(ingredients_list)

    # Step 4: Apply KMeans clustering
    num_clusters = clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans_res = kmeans.fit_predict(X)

    print(f'This is the set of clusters')
    set(kmeans_res)

    df['cluster'] = kmeans.fit_predict(X)

    return df

