from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import string



def model_specific_preprocessing(tokenized_sentence):
    stop_words = set(stopwords.words('english'))
    no_stopwords = [w for w in tokenized_sentence if not w in stop_words]

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

# LDA MODEL

def LDA_model(df, ingredient_column = 'ingredients', preference_column = 'preference', max_df = 0.99, min_df = 0.0001, n_components = 10, max_iter = 100):

    """Returns modelled_df of recipes, each with their respective 'Topic' cluster and weights per topic.
    Also returns a dictionary of sorted_ingredients and sorted_cuisines per cluster
    Ensure that the ingredients columns regards lists of ingredients and df has 'preference' column.
    eg. df[ingredients][0] = [black beans, russet potatoes, green onions, r...]"""

    # Step 1 = Preprocess ingredients list into an 'ingredient sentence' for vectorising
    df['clean_text']=df[ingredient_column].map(lambda x: model_specific_preprocessing(x))

    # Step 2 = TfidfVectorizer which convert each recipe's 'ingredient sentence' into a matrix of weights
    count_vectorizer = CountVectorizer(max_df = max_df, min_df=min_df)
    counted_words = pd.DataFrame(count_vectorizer.fit_transform(df['clean_text']).toarray(), columns = count_vectorizer.get_feature_names_out())

    # Step 3 - LDA model which fits the model and assigns each recipe to its 'best topic'
    lda_model = LatentDirichletAllocation(n_components=n_components, max_iter = max_iter, random_state=0)
    lda_model.fit(counted_words)
    document_topic_mixture = lda_model.transform(counted_words)
    topics = pd.DataFrame(document_topic_mixture)
    most_freq = topics.idxmax(axis=1)
    most_freq.name = "Topic"
    modelled_df = topics.join(most_freq).join(df[ingredient_column]).join(df[preference_column])

    # Step 4 - Dictionary of sorted_ingredients and sorted_cuisines
    topic_mixture = pd.DataFrame(lda_model.components_,columns = count_vectorizer.get_feature_names_out())
    n_components = topic_mixture.shape[0]
    sorted_ingredients, sorted_cuisines = {}, {}
    for topic in range(n_components):
        topic_df = topic_mixture.iloc[topic].sort_values(ascending = False)
        sorted_ingredients[f'Topic {topic}'] = topic_df
        cuisine_df = modelled_df[modelled_df['Topic'] == topic].groupby(preference_column).count()['Topic'].sort_values(ascending = False)
        sorted_cuisines[f'Topic {topic}'] = cuisine_df

    return modelled_df, sorted_ingredients, sorted_cuisines

# Convert recipe string to list

def ing_list(sample_txt):
    elements = []
    for punc in string.punctuation:
        sample_txt = sample_txt.replace(punc,'')
    for word in sample_txt.split():
        elements.append(str(word))
    return elements
#df['fi2'] = df['final_ingredients'].apply(ing_list)
