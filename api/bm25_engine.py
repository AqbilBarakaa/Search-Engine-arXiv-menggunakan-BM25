import os
import json
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from rank_bm25 import BM25Okapi

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

DATA_FILE = os.path.join(MODELS_DIR, 'full_data_processed.p')
INDEX_FILE = os.path.join(MODELS_DIR, 'inverted_indices.p')
RAW_DATA_FILE = os.path.join(DATA_DIR, 'arxiv-metadata-oai-snapshot.json')


def preprocess(text):
    if not text or pd.isna(text):
        return ""

    stop_words = set(stopwords.words("english"))
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = re.sub("(\\d|\\W)+", " ", text)
    words = text.split()
    
    important_stopwords = {'no', 'not', 'nor', 'only', 'against', 'above', 'below'}
    stop_words = stop_words - important_stopwords

    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    processed_words = []

    for word in words:
        if word not in stop_words:
            stemmed = ps.stem(word) 
            lemmatized = lem.lemmatize(stemmed)  
            if len(lemmatized) > 1:  
                processed_words.append(lemmatized)

    return " ".join(processed_words) if processed_words else ""


def load_raw_data(limit=10000):
    data = []
    with open(RAW_DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            if item.get("abstract") and item.get("title"):
                data.append({
                    "paper_id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", "")
                })
            if i >= limit:
                break

    df = pd.DataFrame(data)
    df["abstract"] = df["abstract"].apply(preprocess)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    pickle.dump(df, open(DATA_FILE, "wb"))
    return df


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topN):
    sorted_items = sorted_items[:topN]
    score_vals, feature_vals = [], []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    return dict(zip(feature_vals, score_vals))


def get_abstract_keywords(entry, cv, X, tfidf_transformer, feature_names, topN):
    abstract = entry['abstract']
    if type(abstract) == float:
        return []
    tf_idf_vector = tfidf_transformer.transform(cv.transform([abstract]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords_dict = extract_topn_from_vector(feature_names, sorted_items, topN)
    return list(keywords_dict.keys())


def get_title_keywords(entry):
    title = preprocess(entry['title'])
    if type(title) == float:
        return []
    return title.split(' ')


def get_final_keywords(entry, cv, X, tfidf_trans, feature_names, topN):
    from_abstract = get_abstract_keywords(entry, cv, X, tfidf_trans, feature_names, topN)
    from_title = get_title_keywords(entry)
    return list(set(from_abstract + from_title))


def get_corpus(df):
    return df['abstract'].tolist()


def add_keywords(df, topN=10):
    stop_words = stopwords.words("english")
    corpus = get_corpus(df)
    
    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=1000)
    X = cv.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)

    feature_names = cv.get_feature_names_out()

    df = df.reindex(columns=['paper_id', 'title', 'abstract', 'keywords'])
    df['keywords'] = df.apply(lambda row: get_final_keywords(row, cv, X, tfidf_transformer, feature_names, topN), axis=1)
    return df


def create_inverted_indices(df):
    inverted_ind = {}
    for i in range(df.shape[0]):
        entry = df.iloc[i]
        paper_id = entry['paper_id']
        keywords = entry['keywords']
        for k in keywords:
            inverted_ind.setdefault(k, []).append(paper_id)
    return inverted_ind


def build_index():
    df = pickle.load(open(DATA_FILE, "rb"))
    df_with_keywords = add_keywords(df, 10)
    inverted_indices = create_inverted_indices(df_with_keywords)
    pickle.dump(inverted_indices, open(INDEX_FILE, "wb"))
    return inverted_indices


def get_potential_articles(query):
    inverted_indices = pickle.load(open(INDEX_FILE, "rb"))
    query = preprocess(query)
    query_terms = query.split(' ')
    potential_articles = []
    for word in query_terms:
        if word in inverted_indices:
            potential_articles += inverted_indices[word]
    return list(set(potential_articles))


def bm25_search(articles, df_dic, query, title_weight=1.0, abstract_weight=2.0):
    if not articles:
        return []

    corpus_title, corpus_abstract = [], []
    for article in articles:
        arr = df_dic.get(article, ["", ""])
        corpus_title.append(preprocess(arr[0])) if arr[0] else corpus_title.append("")
        corpus_abstract.append(preprocess(arr[1])) if arr[1] else corpus_abstract.append("")

    query = preprocess(query)
    tokenized_query = query.split(" ")
    tokenized_corpus_title = [doc.split(" ") for doc in corpus_title if doc]
    tokenized_corpus_abstract = [doc.split(" ") for doc in corpus_abstract if doc]

    if not tokenized_corpus_title and not tokenized_corpus_abstract:
        return []

    bm25_title = BM25Okapi(tokenized_corpus_title, k1=1.2, b=0.6)
    bm25_abstract = BM25Okapi(tokenized_corpus_abstract) if tokenized_corpus_abstract else None

    title_scores = bm25_title.get_scores(tokenized_query) if bm25_title else np.zeros(len(articles))
    abstract_scores = bm25_abstract.get_scores(tokenized_query) if bm25_abstract else np.zeros(len(articles))

    doc_scores = (title_weight * title_scores) + (abstract_weight * abstract_scores)
    doc_scores = np.maximum(0, doc_scores)

    results = []
    for i, pid in enumerate(articles):
        paper_data = df_dic.get(pid, ["", ""])
        results.append({
            'paper_id': pid,
            'title': paper_data[0],
            'abstract': paper_data[1] if len(paper_data) > 1 else "",
            'score': float(doc_scores[i]),
            'title_score': float(title_scores[i]),
            'abstract_score': float(abstract_scores[i])
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results


def search(query):
    try:
        df = pickle.load(open(DATA_FILE, "rb"))
        if df.empty:
            return []
            
        df_dic = df.set_index('paper_id').T.to_dict('list')
        articles = get_potential_articles(query)

        if not articles:
            return []

        return bm25_search(articles, df_dic, query)
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []


def init_data():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("Loading raw data...")
        load_raw_data()
    if not os.path.exists(INDEX_FILE):
        print("Building index...")
        build_index()
