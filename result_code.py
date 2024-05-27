from datetime import datetime, timedelta, timezone, date
import pandas as pd
from tqdm import tqdm
import re
import regex
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import word_tokenize 
from sudachipy import tokenizer 
from sudachipy import dictionary 
import jieba 
import nltk
from nltk.corpus import stopwords
import string
import emoji
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier


mode_sudachipy = tokenizer.Tokenizer.SplitMode.C
nltk.download('stopwords')
stop_words = set(stopwords.words())

df = pd.read_csv('data_to_proc.csv')
df.dttm = pd.to_datetime(df.dttm)

def create_tech_column(df):
#create special column for cleaning
    df['text_cleanned'] = df['text']

def do_dommy_typeOfSending(df):
#conduct dommy control typeOfSending
    df['message_source_a'] = (df['typeOfSending'] == 'source_a').map(int)
    df['message_source_b'] = (df['typeOfSending'] == 'source_b').map(int)
    df['message_source_c'] = (df['typeOfSending'] == 'source_c').map(int)
    df.drop(columns = ['typeOfSending'], axis = 1, inplace = True)

def detect_and_clean_links(df, save_context = False):
#detect and clean links to the network
    dummy = ''
    if save_context:
        dummy = ' detectedHref '
        
    df['text_count_ref'] = df.text_cleanned.str.count(r'https?://\S+|www\.\S+')
    df['text_cleanned'] = df.text_cleanned.map(lambda x: re.sub(r'https?://\S+|www\.\S+', dummy, x))

def detect_and_clean_accounts(df, save_context = False):
#detect and clear links to accounts
    dummy = ''
    if save_context:
        dummy = ' detectedAccount '
        
    df['text_count_account_refs'] = df['text_cleanned'].str.count(r'@\S+')
    df['text_cleanned'] = df.text_cleanned.map(lambda x: re.sub(r'@\S+', dummy, x))

def take_numbers_features(df):
#calculates numerical indicators in messages and names
    #messages
    df['matches_text'] = df.text_cleanned.map(lambda x: re.findall('\d+', x))
    df['text_max_number_seq'] = df.text_cleanned.map(lambda matches: max(len(match) for match in matches) if matches else 0)
    df['text_len_number_seq'] = df.text_cleanned.map(lambda matches:len(''.join(matches)))
    df['text_len_number_seq_uniq'] = df.text_cleanned.map(lambda matches:len(set(''.join(matches))))
    df.drop(columns = ['matches_text'], inplace = True)
    #names
    df['matches_username'] = df.author_name.map(lambda x: re.findall('\d+', x))
    df['username_max_number_seq'] = df.matches_username.map(lambda matches: max(len(match) for match in matches) if matches else 0)
    df['username_len_number_seq'] = df.matches_username.map(lambda matches:len(''.join(matches)))
    df['username_len_number_seq_uniq'] = df.matches_username.map(lambda matches:len(set(''.join(matches))))
    df.drop(columns = ['matches_username'], inplace = True)

def detect_numbers(df, save_context = False):
#rewrite numbers to words
    dummy = ''
    if save_context:
        dummy = ' detectedNumber '
      
    df['text_cleanned'] = df.text_cleanned.map(lambda x: re.sub(r'\d+', dummy, x))

def take_len_features(df):
    #look at the length of the lines
    df['text_simbols_all'] = df['text'].map(len)
    df['name_simbols_all'] = df['author_name'].map(len)
    #look at how many unique characters there are
    df['text_simbols_unique'] = df['text'].map(set).map(len)
    df['name_simbols_unique'] = df['author_name'].map(set).map(len)
    #look at the percentage of unique characters
    df['text_simbols_perc_unique'] = (df['text_simbols_unique'] / df['text_simbols_all']).fillna(0)
    df['name_simbols_perc_unique'] = (df['name_simbols_unique'] / df['name_simbols_all']).fillna(0)

def take_capital_features(df):
    #number of capital letters
    df['text_count_capitals'] = df['text_cleanned'].map(lambda x: sum(1 for char in x if char.isupper()))
    df['name_count_capitals'] = df['author_name'].map(lambda x: sum(1 for char in x if char.isupper()))
    #volume of capitals
    df['text_percent_capitals'] = (df['text_count_capitals'] / df['text_simbols_all']).fillna(0)
    df['name_percent_capitals'] = (df['name_count_capitals'] / df['name_simbols_all']).fillna(0)
    #convert everything to lowercase
    df['text_cleanned'] = df['text_cleanned'].str.lower()

def clean_text(text):
#leaves only letters in the text, removes extra spaces
    cleaned_text = regex.sub(r'[^\p{L}\p{Z}]', ' ', text)
    return re.sub(r'\s{2,}', ' ', cleaned_text).strip()

def proceed_cleaning_to_letters(df):
#leaving only text in text fields
    df['text_cleanned'] = df['text_cleanned'].map(clean_text)
    df.drop(columns = ['author_name'], inplace = True)

def replace_repeated_chars(text):
#Reducing the number of unique words obtained by simply repeating symbols
    pattern = r'(.)\1{2,}'
    replaced_text = re.sub(pattern, r'\1\1', text)
    return replaced_text

def reduction_of_massive_character_duplication(df):
#replaces repeating a character more than once with one duplication
    df['text_cleanned'] = df['text_cleanned'].map(replace_repeated_chars)

def test_min_max_text_distance (text):
#Returns the maximum distance between letters of words among words in the text
    res_arr = []
    res = 0
    if len(text) > 1:
        for w in text.split():
            tmp_arr = []
            if len(w) > 1:
                for l in w:
                    tmp_arr.append(ord(l))
                res_arr.append(max(tmp_arr) - min (tmp_arr))
        if len(res_arr) > 0:
            res = max(res_arr)
    return res

def proceed_min_max_text_distance (df):
    df['min_max_distance'] = df.text_cleanned.map(test_min_max_text_distance)

def predict_lng (df):
#determine the language of the message
    detector = LanguageDetectorBuilder.from_all_languages().build()
    ln = df.shape[0]
    chank_size = 100000
    chanks = [[i,min(i + chank_size, ln)] for i in range(0, ln, chank_size)]
    for rng in tqdm(chanks):
        lng_predict = detector.detect_languages_in_parallel_of(df.loc[rng[0]:rng[1], 'text_cleanned'].values)
        df.loc[rng[0]:rng[1], 'lng_predict'] = list(map(lambda x: 'NoData' if x == None else x.name,lng_predict))

def create_lang_bias (df):
#Marks those languages in which the spread between characters is too large without spam
    df['JAPANESE_bias'] = (df.lng_predict == 'JAPANESE').map(int)
    df['CHINESE_bias'] = (df.lng_predict == 'CHINESE').map(int)
    df['VIETNAMESE_bias'] = (df.lng_predict == 'VIETNAMESE').map(int)
    df['KOREAN_bias'] = (df.lng_predict == 'KOREAN').map(int)
    df['TURKISH_bias'] = (df.lng_predict == 'TURKISH').map(int)

def words_tokenizer (dat, mode = mode_sudachipy):
#Selects special tokenizers for texts with and without spaces
    message = dat['text_cleanned']
    lng_predict = dat['lng_predict']
    if lng_predict == 'CHINESE':
        return list(jieba.cut(message, cut_all=False))
    elif lng_predict == 'JAPANESE':
        tokenizer_obj = dictionary.Dictionary().create()
        return tokenizer_obj.tokenize(message, mode).__str__().split()
    else:
        return word_tokenize(message)

def words_tokenezation_process(df):
#tokenizes messages by words
    ln = df.shape[0]
    chank_size = 100000
    chanks = [[i,min(i + chank_size, ln)] for i in range(0, ln, chank_size)]
    for rng in tqdm(chanks):
        df.loc[rng[0]:rng[1], 'words_tokenizer'] = df.loc[rng[0]:rng[1],['text_cleanned','lng_predict']].apply(
            lambda x: words_tokenizer(x), axis = 1)

def remove_empty_strings(array):
#leaves only non-empty elements
    return [element for element in array if element]

def remove_empty_tokenizer_elements(df):
#Leaves only tokens with information
    df['words_tokenizer'] = df['words_tokenizer'].map(remove_empty_strings)

def create_words_count_features(df):
#Creates features that reflect word count
    df['text_len_words'] = df.words_tokenizer.map(len)
    df['text_len_words_unique'] = df.words_tokenizer.map(set).map(len)
    df['text_perc_words_unique'] = (df['text_len_words_unique'] / df['text_len_words']).fillna(0)

def do_stemming (row):
#Performs stemming depending on the message language
    lng_predict = row['lng_predict'].lower()
    words_tokenizer_cleaned = row['words_tokenizer']
    res = []
    if len(words_tokenizer_cleaned) > 0:
        if lng_predict == 'arabic':
            stemmer = SnowballStemmer("arabic")
        elif lng_predict == 'dutch':
            stemmer = SnowballStemmer("dutch")
        elif lng_predict == 'french':
            stemmer = SnowballStemmer("french") 
        elif lng_predict == 'german':
            stemmer = SnowballStemmer("german") 
        elif lng_predict == 'hungarian':
            stemmer = SnowballStemmer("hungarian") 
        elif lng_predict == 'italian':
            stemmer = SnowballStemmer("italian") 
        elif lng_predict == 'norwegian':
            stemmer = SnowballStemmer("norwegian") 
        elif lng_predict == 'porter':
            stemmer = SnowballStemmer("porter") 
        elif lng_predict == 'portuguese':
            stemmer = SnowballStemmer("portuguese") 
        elif lng_predict == 'romanian':
            stemmer = SnowballStemmer("romanian") 
        elif lng_predict == 'russian':
            stemmer = SnowballStemmer("russian") 
        elif lng_predict == 'spanish':
            stemmer = SnowballStemmer("spanish") 
        elif lng_predict == 'swedish':
            stemmer = SnowballStemmer("swedish") 
        else:
            stemmer = SnowballStemmer("english")
        res = list(map(lambda x: stemmer.stem(x), words_tokenizer_cleaned))
    return res

def stemming_row (df):
#Performs array stemming
    df['words_tokenizer'] = df[['lng_predict','words_tokenizer']].apply(do_stemming, axis = 1)

def clear_stop_words (df):
#correction stopwords
    df['words_tokenizer'] = df['words_tokenizer'].map(lambda text: [x for x in text if x not in stop_words])

def create_joined_result (df):
#creates a field for IFIDF embedding
    df['words_tokenizer_joined'] = df['words_tokenizer'].map(' '.join).fillna('')

def delet_redundant_cells (df):
#removes lines that are not needed during training
    df.drop(
        columns = ['lng_predict', 'evaluation', 'text', 'dttm', 'author', 'direction', 'text_cleanned']
        , inplace = True
    )


def count_mess_by_author (id, date, h ,df):
    #counts the author's previous moderations over a period of time
    dt_lim = date - timedelta(hours=h)
    return df[(df.author  == id) & (df.dttm < date) & (df.dttm >= dt_lim)].shape[0]

def count_uniqie_mess_by_author (id, date, h ,df):
    #counts the author's previous unique moderations over a period of time
    dt_lim = date - timedelta(hours=h)
    return df[(df.author  == id) & (df.dttm < date) & (df.dttm >= dt_lim)]['text'].unique().shape[0]

def count_mess_by_author_chanel (id_a, id_c, date, h ,df):
    #counts the author's previous moderations over a period of time to this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.author  == id_a) & (df.direction  == id_c) &  (df.dttm < date) & (df.dttm >= dt_lim)].shape[0]

def count_uniqie_mess_by_author_chanel (id_a, id_c, date, h ,df):
    #counts the author's previous unique moderations over a period of time to this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.author  == id_a) & (df.direction  == id_c) &  (df.dttm < date) & (df.dttm >= dt_lim)]['text'].unique().shape[0]

def count_bad_mess_by_direction (id_c, date, h ,df):
    #сounts the number of spam messages sent for moderation from this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.direction  == id_c) & (df.dttm < date) & (df.dttm >= dt_lim) & (df.isSpam == 1)].shape[0]

def count_bad_uniqie_mess_by_direction (id_c, date, h ,df):
    #сounts the number of unique spam messages sent for moderation from this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.direction  == id_c) & (df.dttm < date) & (df.dttm >= dt_lim) & (df.isSpam == 1)]['text'].unique().shape[0]

def count_authors_by_direction (id_c, date, h ,df):
    #сounts the number of authors whose messages sent for moderation from this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.direction  == id_c) & (df.dttm < date) & (df.dttm >= dt_lim)]['author'].unique().shape[0]

def count_authors_bad_by_direction (id_c, date, h ,df):
    #сounts the number of authors whose bad messages sent for moderation from this channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.direction  == id_c) & (df.dttm < date) & (df.dttm >= dt_lim)& (df.isSpam == 1)]['author'].unique().shape[0]

def count_total_mess_copies (text, date, h ,df):
    #number of occurrences of text in the global list of moderations
    dt_lim = date - timedelta(hours=h)
    return df[(df.text  == text) & (df.dttm < date) & (df.dttm >= dt_lim)].shape[0]

def count_local_mess_copies (id_c, text, date, h ,df):
    #number of occurrences of text on a channel
    dt_lim = date - timedelta(hours=h)
    return df[(df.direction  == id_c) &(df.text  == text) & (df.dttm < date) & (df.dttm >= dt_lim)].shape[0]


def prepare_df_hist_data (df):
#combines all processes for extracting historical features
    #counts the author's previous moderations - hour
    df['prev_mess_count_h'] = df.apply(lambda x: count_mess_by_author(x.author, x.dttm, 1, df), axis = 1)
    #counts the author's previous moderations - day
    df['prev_mess_count_d'] = df.apply(lambda x: count_mess_by_author(x.author, x.dttm, 24, df), axis = 1)
    
    #counts the author's previous unique moderations - hour
    df['prev_uniq_mess_count_h'] = df.apply(lambda x: count_uniqie_mess_by_author(x.author, x.dttm, 1, df), axis = 1)
    #counts the author's previous unique moderations - day
    df['prev_uniq_mess_count_d'] = df.apply(lambda x: count_uniqie_mess_by_author(x.author, x.dttm, 24, df), axis = 1)
    
    #counts the author's previous moderations in this chanel - hour
    df['prev_mess_in_chanel_count_h'] = df.apply(
        lambda x: count_mess_by_author_chanel(x.author, x.direction, x.dttm, 1, df), axis = 1
    )
    #counts the author's previous moderations in this chanel - day
    df['prev_mess_in_chanel_count_d'] = df.apply(
        lambda x: count_mess_by_author_chanel(x.author, x.direction, x.dttm, 24, df), axis = 1
    )
    
    #counts the author's previous moderations in this chanel - hour
    df['prev_uniq_mess_in_chanel_count_h'] = df.apply(
        lambda x: count_uniqie_mess_by_author_chanel(x.author, x.direction, x.dttm, 1, df), axis = 1
    )
    #counts the author's previous moderations in this chanel - day
    df['prev_uniq_mess_in_chanel_count_d'] = df.apply(
        lambda x: count_uniqie_mess_by_author_chanel(x.author, x.direction, x.dttm, 24, df), axis = 1
    )
    
    #counts the author's previous bad moderations - hour
    df['prev_bad_mess_count_h'] = df.apply(lambda x: count_bad_mess_by_direction(x.direction, x.dttm, 1, df), axis = 1)
    #counts the author's previous bad moderations - day
    df['prev_bad_mess_count_d'] = df.apply(lambda x: count_bad_mess_by_direction(x.direction, x.dttm, 24, df), axis = 1)
    
    #counts the author's previous bad unique moderations - hour
    df['prev_bad_mess_count_unique_h'] = df.apply(lambda x: count_bad_uniqie_mess_by_direction(x.direction, x.dttm, 1, df), axis = 1)
    #counts the author's previous bad unique moderations - day
    df['prev_bad_mess_count_unique_d'] = df.apply(lambda x: count_bad_uniqie_mess_by_direction(x.direction, x.dttm, 24, df), axis = 1)
    
    #сounts the number of authors whose messages sent for moderation from this channel - hour
    df['prev_authors_count_h'] = df.apply(lambda x: count_authors_by_direction(x.direction, x.dttm, 1, df), axis = 1)
    #сounts the number of authors whose messages sent for moderation from this channel - day
    df['prev_authors_count_d'] = df.apply(lambda x: count_authors_by_direction(x.direction, x.dttm, 24, df), axis = 1)
    
    #сounts the number of authors whose bad messages sent for moderation from this channel - hour
    df['prev_bad_authors_count_h'] = df.apply(lambda x: count_authors_bad_by_direction(x.direction, x.dttm, 1, df), axis = 1)
    #сounts the number of authors whose bad messages sent for moderation from this channel - day
    df['prev_bad_authors_count_d'] = df.apply(lambda x: count_authors_bad_by_direction(x.direction, x.dttm, 24, df), axis = 1)
    
    #сounts number of occurrences of text in the global list of moderations - hour
    df['prev_text_global_count_h'] = df.apply(lambda x: count_total_mess_copies(x.text, x.dttm, 1, df), axis = 1)
    #сounts number of occurrences of text in the global list of moderations - day
    df['prev_text_global_count_d'] = df.apply(lambda x: count_total_mess_copies(x.text, x.dttm, 24, df), axis = 1)
    
    #сounts number of occurrences of text on a channel - hour
    df['prev_text_local_count_h'] = df.apply(lambda x: count_local_mess_copies(x.direction, x.text, x.dttm, 1, df), axis = 1)
    #сounts number of occurrences of text on a channel - day
    df['prev_text_local_count_d'] = df.apply(lambda x: count_local_mess_copies(x.direction, x.text, x.dttm, 24, df), axis = 1)


def metrics_insert (res_obj, model_name, answer, prediction):
#calculates all metrics based on predictions

    #precision_pos
    precision_pos = precision_score(answer, prediction, pos_label=1)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('precision_pos')
    res_obj['value'].append(precision_pos)       

    #precision_neg
    precision_neg = precision_score(answer, prediction, pos_label=0)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('precision_neg')
    res_obj['value'].append(precision_neg)  

    #recall_pos
    recall_pos = recall_score(answer, prediction, pos_label=1)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('recall_pos')
    res_obj['value'].append(recall_pos) 

    #recall_neg
    recall_neg = recall_score(answer, prediction, pos_label=0)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('recall_neg')
    res_obj['value'].append(recall_neg) 

    #balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(answer, prediction)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('balanced_accuracy')
    res_obj['value'].append(balanced_accuracy) 

    #f1_pos
    f1_pos = f1_score(answer, prediction, pos_label=1)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('f1_pos')
    res_obj['value'].append(f1_pos)     

    #f1_neg
    f1_neg = f1_score(answer, prediction, pos_label=0)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('f1_neg')
    res_obj['value'].append(f1_neg)     

    #matthews_corrcoef
    matthews = matthews_corrcoef(answer, prediction)
    res_obj['model'].append(model_name)
    res_obj['metric'].append('matthews')
    res_obj['value'].append(matthews)     


def preproc (df, save_context = False):
#calls functions responsible for data preprocessing in turn
    create_tech_column(df)
    do_dommy_typeOfSending(df)
    detect_and_clean_links(df, save_context)
    detect_and_clean_accounts(df, save_context)
    take_numbers_features(df)
    proceed_min_max_text_distance(df)
    detect_numbers(df, save_context)
    take_len_features(df)
    take_capital_features(df)
    proceed_cleaning_to_letters(df)
    reduction_of_massive_character_duplication(df)
    predict_lng(df)
    create_lang_bias (df)
    words_tokenezation_process(df)
    remove_empty_tokenizer_elements(df)
    create_words_count_features(df)
    clear_stop_words(df)
    stemming_row (df)
    delet_redundant_cells(df)


#add history_data
prepare_df_hist_data (df)
#remove the very first day from the sample, since there will be no historical data for it
df = df[df.dttm >= df.dttm.min() + timedelta(days=1)]


def get_sentence_vector(arr, model):
    word_vectors = [model.wv[word] for word in arr if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def res_w3v_features (df, model):
    df['vector'] = df['words_tokenizer'].apply(lambda x: get_sentence_vector(x, model))
    y = df.isSpam
    X = df.drop(columns = [ 'words_tokenizer', 'isSpam'])
    coll_to_join = [x for x in X.columns if x != 'vector']
    X['joined'] = X[coll_to_join].apply(lambda x: x.to_list(), axis = 1)
    X.drop(columns = coll_to_join, inplace = True)
    X = X.apply(lambda x: np.hstack(x.to_list()), axis = 1)
    return list(X), y


results = {'model':[], 'metric':[], 'value':[]}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in tqdm(skf.split(df, df.isSpam)):
    #defining sample boundaries
    tmp_train = df.iloc[train_index].copy()
    tmp_test = df.iloc[test_index].copy()

    w2v_model = Word2Vec(sentences=tmp_train.words_tokenizer, vector_size=300, window=5, workers=-1, seed = 42)

    X_train, y_train = res_w3v_features(tmp_train, w2v_model)
    X_test, y_test = res_w3v_features(tmp_test, w2v_model)

    counts = np.bincount(y_train)
    scale_pos_weight = counts[0] / counts[1]

    #LogisticRegression
    for s in ['lbfgs', 'liblinear']:
        print('LR-{}'.format(s))
        model = LogisticRegression(class_weight = 'balanced', random_state = 42, n_jobs = -1, max_iter = 10000,  solver = s)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'LR-{}'.format(s), y_test, prediction)

    #GaussianNB
    for a in [0.001, 0.01, 0.1, 1.0]:
        print('NB-{}'.format(a))
        model = GaussianNB(priors = (y_train.value_counts()/ len(y_train)).to_list(), var_smoothing = a)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'NB-{}'.format(a), y_test, prediction)

    #RT 
    for depth in [10, 50, 100, 200]:
        print('RT-{}'.format(depth))
        model = DecisionTreeClassifier(random_state = 42, class_weight = 'balanced', max_depth = depth)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'RT-{}'.format(depth), y_test, prediction)

    #RandomForestClassifier
    for ne in range (100, 501, 100):
        print('RFC-{}'.format(ne))
        model = RandomForestClassifier(n_jobs = -1, random_state = 42, class_weight = 'balanced', n_estimators = ne)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'RFC-{}'.format(ne), y_test, prediction)

    #LDA
    for solv in ['svd', 'lsqr']:
        print('LDA-{}'.format(solv))
        model = LinearDiscriminantAnalysis(priors = (y_train.value_counts()/ len(y_train)).to_list(), solver = solv)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'LDA-{}'.format(solv), y_test, prediction)

    #QDA
    for rp in [0.01, 0.1, 1]:
        print('QDA-{}'.format(rp))
        model = QuadraticDiscriminantAnalysis(priors = (y_train.value_counts()/ len(y_train)).to_list(), reg_param = rp)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'QDA-{}'.format(rp), y_test, prediction)

    #XGB
    for bust in ['gbtree','gblinear','dart']:
        print('XGB-{}'.format(bust))
        model = xgb.XGBClassifier(booster= bust, scale_pos_weight= scale_pos_weight)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'XGB-{}'.format(bust), y_test, prediction)

    #ExtraTC
    for ne in range (100, 501, 100):
        print('ExtraTC-{}'.format(ne))
        model = ExtraTreesClassifier(n_jobs = -1, random_state = 42, n_estimators = ne, class_weight='balanced')
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'ExtraTC-{}'.format(ne), y_test, prediction)

    #CatB
    for depth in [1,3,6,9]:
        print('CatB-{}'.format(depth))
        model = CatBoostClassifier(iterations=100, loss_function='Logloss'
            , scale_pos_weight= scale_pos_weight, depth = depth, verbose = False)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'CatB-{}'.format(depth), y_test, prediction)
        
    #SVM 
    for kern in ['linear', 'poly', 'rbf', 'sigmoid']:
        print('SVM-{}'.format(kern))
        model = SVC(kernel= kern, random_state=42, class_weight='balanced', cache_size = 16000)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'SVM-{}'.format(kern), y_test, prediction)

    #BaggingC
    for ne in range (100, 501, 100):
        print('BaggingC-{}'.format(ne))
        model = BaggingClassifier(n_jobs = -1, random_state = 42, n_estimators = ne)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'BaggingC-{}'.format(ne), y_test, prediction)

    #GradientBoostingClassifier
    for ne in range (100, 501, 100):
        print('GBC-{}'.format(ne))
        model = GradientBoostingClassifier(random_state = 42, n_estimators = ne)
        model = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics_insert(results, 'GBC-{}'.format(ne), y_test, prediction)


pd.DataFrame(results).pivot_table(index = 'model', columns = 'metric', values = 'value', aggfunc = 'mean')
