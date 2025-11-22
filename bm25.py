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
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from rank_bm25 import BM25Okapi

# Unduh resource NLTK yang dibutuhkan
nltk.download('stopwords')
nltk.download('wordnet')

# --- PREPROCESSING ---
def preprocess(text):
    if not text or pd.isna(text):
        return ""

    stop_words = set(stopwords.words("english"))
    
    # Hapus karakter non-alfabet
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    
    # Bersihkan angka dan simbol
    text = re.sub("(\\d|\\W)+", " ", text)
    words = text.split()
    
    # Simpan beberapa stopword penting untuk konteks ilmiah
    important_stopwords = {'no', 'not', 'nor', 'only', 'against', 'above', 'below'}
    stop_words = stop_words - important_stopwords

    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    processed_words = []

    for word in words:
        if word not in stop_words:
            stemmed = ps.stem(word)  # Stemming
            lemmatized = lem.lemmatize(stemmed)  # Lemmatization
            if len(lemmatized) > 1:  # Lewati karakter tunggal
                processed_words.append(lemmatized)

    return " ".join(processed_words) if processed_words else ""

# --- EKSTRAK DATA DARI JSON ---
def getDataFromJson():
    data = []
    with open('./data/arxiv-metadata-oai-snapshot.json', 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            # Ambil hanya data yang memiliki abstrak dan judul
            if item.get("abstract") and item.get("title"):
                data.append({
                    "paper_id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", "")
                })
            if i >= 5000:  # Batasi data hingga 5000 dokumen
                break

    df = pd.DataFrame(data)
    df["abstract"] = df["abstract"].apply(preprocess)
    pickle.dump(df, open("full_data_processed_FINAL.p", "wb"))  # Simpan data yang sudah diproses
    return df

# --- TF-IDF KEYWORDS ---
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

def getAbstractKeywords(entry, cv, X, tfidf_transformer, feature_names, topN):
    abstract = entry['abstract']
    if type(abstract) == float:
        return []
    tf_idf_vector = tfidf_transformer.transform(cv.transform([abstract]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords_dict = extract_topn_from_vector(feature_names, sorted_items, topN)
    return list(keywords_dict.keys())

def getTitleKeywords(entry):
    title = preprocess(entry['title'])
    if type(title) == float:
        return []
    return title.split(' ')

def getFinalKeywords(entry, cv, X, tfidf_trans, feature_names, topN):
    # Gabungkan keyword dari abstrak dan judul
    fromAbstract = getAbstractKeywords(entry, cv, X, tfidf_trans, feature_names, topN)
    fromTitle = getTitleKeywords(entry)
    return list(set(fromAbstract + fromTitle))

def getCorpus(df):
    return df['abstract'].tolist()

def addKeywords(df, topN, makeFile, fileName):
    stop_words = stopwords.words("english")
    corpus = getCorpus(df)
    
    # Gunakan CountVectorizer dengan filter fitur
    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=1000)
    X = cv.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)

    feature_names = cv.get_feature_names_out()

    # Tambahkan kolom keywords
    df = df.reindex(columns=['paper_id', 'title', 'abstract', 'keywords'])
    df['keywords'] = df.apply(lambda row: getFinalKeywords(row, cv, X, tfidf_transformer, feature_names, topN), axis=1)

    if makeFile:
        pickle.dump(df, open(fileName, "wb"))
    return df

def createInvertedIndices(df):
    invertInd = {}
    for i in range(df.shape[0]):
        entry = df.iloc[i]
        paper_id = entry['paper_id']
        keywords = entry['keywords']
        for k in keywords:
            invertInd.setdefault(k, []).append(paper_id)
    return invertInd

def organize():
    # Fungsi utama untuk menyusun pipeline: load, ekstrak keyword, buat indeks
    df = pickle.load(open("full_data_processed_FINAL.p", "rb"))
    df_with_keywords = addKeywords(df, 10, False, "full_data_withKeywords_FINAL.p")
    invertedIndices = createInvertedIndices(df_with_keywords)
    pickle.dump(invertedIndices, open("invertedIndices_FINAL.p", "wb"))

def getPotentialArticleSubset(query):
    invertedIndices = pickle.load(open("invertedIndices_FINAL.p", "rb"))
    query = preprocess(query)
    queryTerms = query.split(' ')
    potentialArticles = []
    for word in queryTerms:
        if word in invertedIndices:
            potentialArticles += invertedIndices[word]
    return list(set(potentialArticles))

# --- BM25 RETRIEVAL ---
def bm25(articles, df_dic, title_w, abstract_w, query):
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

    # Kombinasikan skor dari judul dan abstrak menggunakan bobot
    doc_scores = (title_w * title_scores) + (abstract_w * abstract_scores)
    doc_scores = np.maximum(0, doc_scores)

    results = []
    for i, pid in enumerate(articles):
        paper_data = df_dic.get(pid, ["", ""])
        results.append({
            'paper_id': pid,
            'title': paper_data[0],
            'abstract': paper_data[1] if len(paper_data) > 1 else "",
            'score': doc_scores[i],
            'title_score': title_scores[i],
            'abstract_score': abstract_scores[i]
        })

    # Urutkan berdasarkan skor akhir
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    return results

# --- INTERFACE UNTUK RETRIEVAL ---
def retrieve(query):
    try:
        df = pickle.load(open("full_data_processed_FINAL.p", "rb"))
        if df.empty:
            raise ValueError("Loaded dataframe is empty!")
            
        df_dic = df.set_index('paper_id').T.to_dict('list')
        articles = getPotentialArticleSubset(query)

        if not articles:
            print(f"No articles found for query: {query}")
            return []

        results = bm25(articles, df_dic, 1, 2, query)  # title weight = 1, abstract weight = 2
        return results
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        return []
