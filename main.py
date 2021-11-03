import pickle
import time
from fastapi import FastAPI
from Model.TFIDF import TFIDFModel

app = FastAPI()

# Getting data
with open('./tfidf-bbc-50000.pkl', 'rb') as f:
    data, vectors_norm, words_dict, IDF = pickle.load(f)
    print("[loading_data]\tLoading data success.")

# Create object
tfidf = TFIDFModel(vectors_norm, words_dict, IDF)


@app.get("/")
def read_root():
    return {"status": 1}


@app.get("/query/{query_string}")
def get_query(query_string: str):
    start = time.time()

    # query = ["Global warming animal"]
    query = [query_string]
    result = tfidf.query_top_n(query)
    doc_ids = [list(data.keys())[i] for i in result]
    doc_names = [data[i]['title'] for i in doc_ids]

    print(f'[time_used]\t{time.time()- start} seconds')
    return doc_names
