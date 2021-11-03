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


# TO MAKE SYSTEM RUN FASTER
#                                                                   DBBD
#                                                                 .BBBBBBM:
#                                                               iBBBBBEDX5:
#                                                               BBBBBBBBMQK2r
#                                                             iBBBBBDdqv2IMU.
#                                                             jBBBBBZqi:: 7B.
#                                                             iBBBBEBgLB7: Q:
#                                                             .BBBBQBM Qr :d.
#                                                               BBBBBBB r: :I
#                                                               BBBBBBM:Ui::
#                                                               BBBBBBq.iigB.:
#                                                               :BBBBBqi. 7:
#                                                           .iYbBBBBBB5:   Bs.::.
#                                                       iMRbDQBBRP2uuU7:.iBi.7.:::.
#                                                       rBBj7Igb5uviirL7i.MB:.i.. :7
#                                                       BBBMBBBEuLv2L:ri:dSB ::. .i:.
#                                                       BBBBBBBBBBBb. i7Bi Q LL. i7 .
#                                                       BBQBBBBBBBB.  LBM YM 77. Y7.:
#                                                       BBBBBBBBBBBQ  BB:7BU.ju .grJ:
#                                                     iBBBBBBBBBBBB7 U7iBB7b iQMBjP .
#                                                     SBBDBBBBBBBBBM   buQZB  BBBB. .
#                                                     gBB5BB.BBBBBBB:.sq7BLE. QBBi  .
#                                                     BBMRBBdBBXYLjgMdPJrB. : BBMrLri.
#                                                     BBBBBBBEJrIZBQDPLi:.... BBQ17i:.
#                                                     BBBBBBBQBBBBQq2Lr::.:v7BBqi.  .:
#                                                       BBBB7rBBBBBMdUs7LKBBBBg:   ..:.
#                                                           PBBBBBEQKjUPDBBBX .7...:.
#                                                           vBBBBBbMBBBMv.  iBBr  .
#                                                   Yr71K77u7jbQDiKBBBBBBBBBLq7...r7.
#                                                 7LMBEbqESJv772qi:LQMqQSEYPZJssrIBBBBPMBi
#                                               BBBBQjvv7LjuuUu7riL:L.:7::::J5BBBBBBBBQK2Y:.
#                                             LBB5QMXU5X55uL7:ir7:K7.j7i5BBBBBMZMMDSXS7i:...
#                                             7BRUIUqbq5j7rr777rrQM rXuRBBBBBBBBgKDMMMMUi:
#                                               BBbP17uSjL7iir7sEBM .jjBBBBgBBBBBQRbbMBEirr...
#                                               BBBBbqbSUUqbgQdBj.rL7iBBBdbIBBBBBRDb2XIi:7i:
#                                                 7KXPgQQMBR7KL::rJ7r7rrrrb1Ji71Lvr:.i7i...
