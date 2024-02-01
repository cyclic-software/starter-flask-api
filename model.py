# Importing Libraries
from nltk.corpus.reader import reviews
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
import pickle as pk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# load the pickle files 
count_vector = pk.load(open('pickle_file/count_vector.pkl','rb'))            # Count Vectorizer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl','rb')) # TFIDF Transformer
model = pk.load(open('pickle_file/model.pkl','rb'))                          # Classification Model
recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))   # User-User Recommendation System 

nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])

product_df = pd.read_csv('sample30.csv',sep=",")


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

stopword_list= stopwords.words('english')

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

#predicting the sentiment of the product review comments
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

#Recommend the products based on the sentiment from model
def recommend_products(user_name):
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl','rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name','reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: normalize_and_lemmaize(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df
    
def top5_products(df):
    total_product=df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name','predicted_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['reviews_text'],on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] ==  1][:5])
    return output_products