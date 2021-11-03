import nltk
import math
import numpy as np
import difflib
from scipy.sparse import coo_matrix
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stops = set(stopwords.words('english'))


def tokenize(raw):
    tokenized = []
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(raw)):
        tmp = ' '.join(map(str, filter(lambda x: x not in stops, raw[i].split(" "))))
        tokenized.append(list(filter(lambda x: len(x) > 1 and not x.isnumeric(), tokenizer.tokenize(tmp))))
        # print('\rdone: {}/{}'.format(i + 1, len(raw)), end='', flush=True)
    return tokenized


def custom_lemmatize(word):
    tmp = WordNetLemmatizer().lemmatize(word)
    if tmp[0] == "_":
        tmp = tmp[1:]
    if tmp[-1] == "_":
        tmp = tmp[:-1]
    return tmp


def lemmatize(tokenized):
    lemmatized = []
    for i in range(len(tokenized)):
        lemmatized.append(list(map(custom_lemmatize, tokenized[i])))
        # print('\rdone: {}/{}'.format(i + 1, len(tokenized)), end='', flush=True)
    return lemmatized


def calculate_tf(processed):
    TF = []
    for i in range(len(processed)):
        tmp = {}
        for word in processed[i]:
            try:
                tmp[word] = tmp[word] + 1
            except KeyError:
                tmp[word] = 1
        for word, freq in tmp.items():
            tmp[word] = 1.0 + math.log10(freq)
        TF.append(tmp)
        # print('\rdone: {}/{}'.format(i + 1, len(processed)), end='', flush=True)
    return TF


class TFIDFModel:

    def __init__(self, vectors_norm, words_dict, IDF):
        self.vectors_norm = vectors_norm
        self.words_dict = words_dict
        self.IDF = IDF

    def calculate_vector_norm(self, TF):
        row = []
        col = []
        score = []
        for i in range(len(TF)):
            tmp_score = []
            for word in TF[i]:
                row.append(i)
                col.append(self.words_dict[word])
                tmp_score.append(TF[i][word] + self.IDF[word])
            score = score + list(np.array(tmp_score) / np.linalg.norm(np.array(tmp_score)))
            # print('\rdone: {}/{}'.format(i + 1, len(TF)), end='', flush=True)
        return coo_matrix((score, (row, col)), shape=(len(TF), len(self.words_dict)))

    def word_correction(self, query):
        all_words = list(self.words_dict.keys())
        for i, word in enumerate(query[0]):
            if word not in self.words_dict:
                query[0][i] = difflib.get_close_matches(word, all_words)[0]
        return query

    def query_top_n(self, query, top_n=10):
        query[0] = query[0].lower().replace("\n", " ")
        query_tokenized = tokenize(query)
        query_lemmatized = lemmatize(query_tokenized)
        query_tf = calculate_tf(self.word_correction(query_lemmatized))
        query_vector_norm = self.calculate_vector_norm(query_tf)
        result = query_vector_norm.dot(self.vectors_norm.transpose()).toarray()[0]
        top_result = result.argsort()[-top_n:][::-1]
        # print(f'{result[top_result]}')
        # print(f'top match(n={top_n}): {top_result}')
        return top_result
