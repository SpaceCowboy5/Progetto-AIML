# import delle librerie

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import time as time
import pickle
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import random
import networkx as nx
from sklearn.metrics import classification_report
from tqdm import tqdm

tqdm.pandas()

from IPython.display import HTML, display

pd.set_option('display.max_columns', None)  # viene usato per mostrare completamente le colonne nel dataset


# pd.set_option('display.max_rows', None) # viene usato per mostrare completamente le righe nel dataset

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


# CSV Loading
print("CARICAMENTO DATASET")
startreading = time.time()

song_data = pd.read_csv("CODICE PAPER+DATASET/songs_final.csv")
user_data = pd.read_csv("CODICE PAPER+DATASET/user_data_final.csv")

# tolgo il 90% del dataset
user_data = user_data.head(int(round(len(user_data) * 0.01)))

# da usare per cambiare il nome di user_id in user
# user_data.rename(columns={'user_id': 'user'}, inplace=True)
# user_data.to_csv('CODICE PAPER+DATASET/user_data_final.csv', index=False)

# inserisco triple_id in user_data
user_data.insert(0, 'triple_id', range(0, 0 + len(user_data)))
print(song_data.head())
print(user_data.head())
print('Numero utenti nel dataset: ' + str(user_data['user'].nunique()))

# tolgo dalle canzoni totali quelle senza dati di ascolto (0 listenings)
listened_songs = user_data["song_id"].drop_duplicates()
song_data = song_data[song_data["song_id"].isin(listened_songs)]
print('Numero utenti nel dataset: ' + str(user_data['user'].nunique()))
# User data splitting

#  write dict
# set_dict = {}


# def add_to_dict(key, value):
# print(key)
# set_dict[key] = value


# def train_test_set(x):
# if len(x) < 5:
#     x.apply(lambda y: add_to_dict(y["triple_id"], "train"), axis=1)
# else:
#     x = x.sample(frac=1)
#     train_set = x.head(int(round(len(x) * (0.8))))
#     train_set.apply(lambda y: add_to_dict(y["triple_id"], "train"), axis=1)
#     test_set = x.tail(int(round(len(x) * (0.2))))
#     test_set.apply(lambda y: add_to_dict(y["triple_id"], "test"), axis=1)


# user_data.groupby('user').progress_apply(train_test_set)
# set dovrebbe essere un attributo con valore test o
# training a seconda dell'appartenenza della riga al particolare set, per questo motivo vado a effettuare lo
# splitting del dataset user
print("SPLITTING DATASET USER")
user_data = user_data.sample(frac=1)
train_set = user_data.head(int(round(len(user_data) * 0.8)))
test_set = user_data.tail(int(round(len(user_data) * 0.2)))
train_set["set"] = "train"
test_set["set"] = "test"

print(train_set['user'].nunique())
print(test_set['user'].nunique())

print(train_set.columns)
print(test_set.columns)

user_data = pd.concat([train_set, test_set])

print(user_data.head())
print(user_data.tail())

# modifico artist_mbdtags
# song_data["artist_mbtags"] = song_data["artist_mbtags"].fillna("[]")
# song_data.to_csv('songs_final.csv', index=False)

print(song_data["tags"].head(20))

# user_data_total conterrà tutte le righe di training e test
user_data_total = user_data

print(user_data_total.columns)

# load training data labels(user_data invece conterrà tutte le righe di training)
user_data = user_data_total[user_data_total["set"] != "test"]

endreading = time.time()

print("Tempo necessario per la lettura dei datasets: " + str(endreading - startreading) + " s")

# Creazione ipergrafo

# dataframe with listened songs
print("CREAZIONE IPERGRAFO")
startreading = time.time()
user_hyperedges = user_data.groupby('user')['song_id'].apply(list).reset_index(name='songs')
user_hyperedges["user_id"] = "u_ " + (user_hyperedges.index.map(str))
user_hyperedges["user_id_matrix"] = user_hyperedges.index
print(user_hyperedges.columns)
print(user_hyperedges)

# add new user ids and data
user_data = user_data.merge(user_hyperedges, on="user", how="left").set_index(user_data.columns.values[0])
print(user_data)
# user_data = user_data[["user_id", "user_id_matrix", "song_id", "play_count", "triple_id", "set"]]
# user_data = user_data["user_id", "user_id_matrix", "song_id", "play_count", "triple_id", "set"]
# user_data = user_data[["user","user_id","user_id_matrix","song_id","play_count","triple_id","set"]]

# dataframe per releases

release_hyperedges = song_data.groupby(['artist_name', 'release_name'])['song_id'].apply(list).reset_index(name='songs')
release_hyperedges["release_id"] = "r_ " + (release_hyperedges.index.map(str))
print(release_hyperedges)

# dataframe per artists

artists_hyperedges = song_data.groupby('artist_name')['song_id'].apply(list).reset_index(name='songs')
artists_hyperedges["artist_id"] = "a_ " + (artists_hyperedges.index.map(str))
print(artists_hyperedges)

# dataframe per tag

tags_hyperedges = pd.DataFrame(columns=["tag", "tag_id", "songs"])

tag_dict = {}


# DA CAPIRE
def add_tag_edges(tag_song_row):
    # print(tag_song_row)
    j = 0
    tags = eval(tag_song_row[1])
    print(tags)
    #print(tags[0][0])
    while j < len(tags):
        print(tags[j][0])
        if tag_dict.get(tags[j][0]) == None:
            tag_dict[tags[j][0]] = {}
            tag_dict[tags[j][0]]["songs"] = []
            tag_dict[tags[j][0]]["tag_id"] = "t_" + str(len(tag_dict) - 1)
        tag_dict[tags[j][0]]["songs"].append(tag_song_row[0])
        j += 1


song_data[["song_id", "tags"]].apply(add_tag_edges, axis=1)

print(tag_dict["black metal"])

# dict with hyperedges

hyperedges = {}


# hyperedge users

def add_user_edges(user_row):
    hyperedges[user_row['user_id']] = {}
    hyperedges[user_row['user_id']]['members'] = user_row['songs']
    hyperedges[user_row['user_id']]['members'].append(user_row['user_id'])
    hyperedges[user_row['user_id']]['category'] = 'user'


user_hyperedges.apply(add_user_edges, axis=1)


# hyperedge releases
def add_release_edges(release_row):
    hyperedges[release_row['release_id']] = {}
    hyperedges[release_row['release_id']]['members'] = release_row['songs']
    hyperedges[release_row['release_id']]['members'].append(release_row['release_id'])
    hyperedges[release_row['release_id']]['category'] = 'release'


release_hyperedges.apply(add_release_edges, axis=1)


# hyperedge artists
def add_artist_edges(artist_row):
    hyperedges[artist_row['artist_id']] = {}
    hyperedges[artist_row['artist_id']]['members'] = artist_row['songs']
    hyperedges[artist_row['artist_id']]['members'].append(artist_row['artist_id'])
    hyperedges[artist_row['artist_id']]['category'] = 'artist'


artists_hyperedges.apply(add_artist_edges, axis=1)

# hyperedge tags
for tag in tag_dict:
    tag_id = tag_dict[tag]["tag_id"]
    hyperedges[tag_id] = {}
    hyperedges[tag_id]['members'] = tag_dict[tag]["songs"]
    hyperedges[tag_id]['members'].append(tag_id)
    hyperedges[tag_id]['category'] = 'tag'

# print(hyperedges['u_20'])
# print(hyperedges['r_30'])
# print(hyperedges['a_2'])
print(hyperedges['t_70'])

# compute hyperedges max_size e min_size

max_size = 0
min_size = 100000

for h_index in hyperedges:
    members = hyperedges[h_index]["members"]
    if len(members) < min_size:
        min_size = len(members)
    if len(members) > max_size:
        max_size = len(members)

print(min_size, max_size)

# count hyperedge number per category

cat_amounts = {}

for h in hyperedges:
    if hyperedges[h]["category"] not in cat_amounts:
        cat_amounts[hyperedges[h]["category"]] = 1
    else:
        cat_amounts[hyperedges[h]["category"]] = 1

print(cat_amounts)

# plot hyperedges distribution

# pd_df = pd.DataFrame(list(cat_amounts.items()))
# pd_df.columns = ["Dim", "Count"]
# sort df by Count column
# pd_df = pd_df.sort_values(['Count']).reset_index(drop=True)

# plt.figure(figsize=(12, 8))
# plot barh chart with index as x values
# ax = sns.barplot(pd_df.index, pd_df.Count)
# ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
# ax.set(xlabel="Hyperedge category", ylabel='Count')
# add proper Dim values as x labels
# ax.set_xticklabels(pd_df.Dim)
# for item in ax.get_xticklabels(): item.set_rotation(90)
# for i, v in enumerate(pd_df["Count"].iteritems()):
#  ax.text(i, v[1], "{:,}".format(v[1]), color='m', va='bottom', rotation=45)
# plt.tight_layout()
# plt.show()

# for each node, get hyperedges
vertexMemberships = {}
for h_index in hyperedges:
    hyperedge = hyperedges[h_index]
    nodes = hyperedge["members"]
    for node in nodes:
        if node in vertexMemberships:
            vertexMemberships[node].append(h_index)
        else:
            nodeMembershipList = []
            nodeMembershipList.append(h_index)
            vertexMemberships[node] = nodeMembershipList

print(vertexMemberships['SOZOBWN12A8C130999'])

# save hyperedges and nodes

print("Salvataggio dati hyperedge e nodi...\n")

print("Numero di hyperedge: " + str(len(hyperedges)))
pickle.dump(hyperedges, open('CODICE PAPER+DATASET/hyperedges.p', 'wb'))
print("hyperedges.p salvato.\n")

print("Numero di nodi: " + str(len(vertexMemberships)))
pickle.dump(vertexMemberships, open('CODICE PAPER+DATASET/vertexMemberships.p', 'wb'))
print("vertexMemberships.p salvato.\n")

print("Salvataggio dati hyperedge e nodi completato.")

hyperedges = pickle.load(open('CODICE PAPER+DATASET/hyperedges.p', 'rb'))
vertexMemberships = pickle.load(open('CODICE PAPER+DATASET/vertexMemberships.p', 'rb'))
endreading = time.time()
print("Tempo necessario per la creazione/salvataggio hyperedges: " + str(endreading - startreading) + " s")

# Compute random walks (SaT e TaS)
print("CALCOLA RANDOM WALKS")

startreading = time.time()


#  SubsampleAndTraverse: definisce random walk tra vertici
def SubsampleAndTraverse(length, num_walks, hyperedges, vertexMemberships, alpha=1., beta=0):
    walksSAT = []
    for vertex in vertexMemberships:
        hyperedge_index = random.choice(vertexMemberships[vertex])
        hyperedge = hyperedges[hyperedge_index]
        walk_vertex = []
        curr_vertex = vertex
        for _ in range(num_walks):
            initial = True
            hyperedge_num = hyperedge_index
            curr_hyperedge = hyperedge
            for i in range(length):
                proba = (float(alpha) / len(curr_hyperedge["members"])) + beta
                if random.random() < proba:
                    adjacent_hyperedges = vertexMemberships[curr_vertex]
                    hyperedge_num = random.choice(adjacent_hyperedges)
                    curr_hyperedge = hyperedges[hyperedge_num]
                walk_vertex.append(str(curr_vertex))
                next_nodes = curr_hyperedge["members"][:]
                if curr_vertex in next_nodes: next_nodes.remove(curr_vertex)
                curr_vertex = random.choice(next_nodes)
            walksSAT.append(walk_vertex)
        if (len(walksSAT) % 100000) == 0:
            print(str(len(walksSAT) * 100 / (num_walks * len(vertexMemberships))) + "%")
    return walksSAT


walksSAT = SubsampleAndTraverse(length=100, num_walks=10, hyperedges=hyperedges, vertexMemberships=vertexMemberships,
                                alpha=1, beta=0)
print(walksSAT[1])
endreading = time.time()
print("Tempo necessario per la creazione dei random walk': " + str(endreading - startreading) + " s")
# walksSAT = pickle.load(open('CODICE PAPER+DATASET/walksSAT.p', 'rb'))


# delta = int(10603110 / 50)
# for i in range(0, 10603110, delta):
#  print(i)
#  if i + delta < 10603110:
#      filename = "walksSAT-" + str(i) + "-" + str(i + delta)
#     pickle.dump(walksSAT[i: i + delta], open("CODICE PAPER+DATASET/walksSAT/" + filename, "wb"))
# else:
#    filename = "walksSAT-" + str(i) + "-" + str(10603110)
#     pickle.dump(walksSAT[i:], open("CODICE PAPER+DATASET/walksSAT/" + filename, "wb"))

# import os
# import json
# import pickle

# prova = os.listdir('CODICE PAPER+DATASET/walksSAT')[1]

# f = open('CODICE PAPER+DATASET/walksSAT/' + prova, 'rb')
# lista = pickle.load(f)
# f.close()

# f = open("CODICE PAPER+DATASET/walksSAT/prova.json", 'w')
# json.dump(lista, f)
# f.close()

# import os
# import gc

# walksSAT = []
# for f in os.listdir('CODICE PAPER+DATASET/walksSAT'):
#   pikd = open('CODICE PAPER+DATASET/walksSAT/' + f, 'rb')
#  for el in pickle.load(pikd):
#      walksSAT.append(el)
#  pikd.close()

# import os

# walksSAT = []
# for ws in os.listdir('CODICE PAPER+DATASET/walksSAT'):
#   walksSAT.append(pickle.load(open('CODICE PAPER+DATASET/walksSAT/' + ws, 'rb')))

# salvataggio dati SaT
# ("Salvataggio dati SaT...")
# pickle.dump(walksSAT, open('CODICE PAPER+DATASET/walksSAT.p', 'wb'))
# print("walksSAT.p salvato.\n")

# import os
# import pickle

# files = os.listdir('CODICE PAPER+DATASET/walksSAT')

# aggregated = pickle.load(open('CODICE PAPER+DATASET/walksSAT/' + files[0], 'rb'))
# pickle.dump(aggregated, open('CODICE PAPER+DATASET/walksSAT.p', 'wb'))
# os.remove('CODICE PAPER+DATASET/walksSAT/' + files[0])

# for f in files[1:]:
#  print(f)
# aggregated = pickle.load(open('drive/MyDrive/TESI MIRKO/walksSAT.p', 'rb'))
# for el in pickle.load(open('CODICE PAPER+DATASET/walksSAT/' + f, 'rb')):
#      aggregated.append(el)
# pickle.dump(aggregated, open('CODICE PAPER+DATASET/walksSAT.p', 'wb'))
# os.remove('CODICE PAPER+DATASET/walksSAT/' + f)

# Generate context embeddings

from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time as time


class EpochLogger(CallbackAny2Vec):
    # Callback to log information about training

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


epoch_logger = EpochLogger()


def EmbedWord2Vec(walks, dimension):
    time_start = time.time()
    print("Creating embeddings.")
    model = Word2Vec(sentences=walks, vector_size=dimension, window=5, min_count=0, sg=1, workers=16, epochs=1,
                     callbacks=[epoch_logger])
    node_ids = model.wv.index_to_key
    node_embeddings = model.wv.vectors
    print("Embedding generation runtime: ", time.time() - time_start)
    return node_ids, node_embeddings


vertex_embedding_dimension = 200

import gc

gc.collect()

# creazione embeddings dei nodi
print(datetime.datetime.now())
print("Creazione vertex embeddings...")
vertex_ids, vertex_embeddings = EmbedWord2Vec(walks=walksSAT, dimension=vertex_embedding_dimension)
print(datetime.datetime.now())
print("Vertex embeddings completati. ( " + str(len(vertex_embeddings)) + " embeddings)")

print("Context embedding per nodo  " + vertex_ids[0] + ":")
print(vertex_embeddings[0])

context_embeddings = dict(zip(vertex_ids, vertex_embeddings))
print(context_embeddings['SOAUWYT12A81C206F1'])

# salvataggio vertex embeddings
print("Salvataggio context embeddings...")
pickle.dump(context_embeddings, open('CODICE PAPER+DATASET/context_embeddings.p', 'wb'))
print("context_embeddings.p salvato.\n")

# Mood detection (compute arousal e valence per user)

# get arousal e valence foreach song listened by each user_id

user_data["arousal"] = user_data.merge(song_data, on="song_id", how="left").set_index("triple_id")["arousal"]
user_data["valence"] = user_data.merge(song_data, on="song_id", how="left").set_index("triple_id")["valence"]

print(user_data[user_data["user_id"] == "u_60"])

# compute arousal e valence for each user as average of each listend song (weighted on play_count)

user_data['weighted_arousal'] = user_data['arousal'] * user_data['play_count']
user_data['weighted_valence'] = user_data['valence'] * user_data['play_count']

user_mood = user_data.groupby('user_id').agg \
    ({'weighted_arousal': 'sum', 'weighted_valence': 'sum', 'play_count': 'sum'}).reset_index()
user_mood["arousal"] = user_mood["weighted_arousal"] / user_mood["play_count"]
user_mood["valence"] = user_mood["weighted_valence"] / user_mood["play_count"]

print(user_mood)

print(user_mood[user_mood['user_id'] == 'u_0'][['user_id', 'arousal', 'valence']])

# load max per feature of each song of each user for max pooling

user_max = user_data.groupby('user_id').agg({'arousal': 'max', 'valence': 'max'}).reset_index()
user_mood["arousal"] = user_mood["arousal"] + user_max["arousal"]
user_mood["valence"] = user_mood["valence"] + user_max["valence"]

user_mood.head()

# compute dict with valence and arousal of song and user

arousal_valence_dict = {}


def generate_arousal_valence_dict(row):
    if arousal_valence_dict.get(row[0]) is None:
        arousal_valence_dict[row[0]] = {}
    arousal_valence_dict[row[0]]["valence"] = row[1]
    arousal_valence_dict[row[0]]["arousal"] = row[2]


user_mood[["user_id", "valence", "arousal"]].progress_apply(generate_arousal_valence_dict, axis=1)

song_data[["song_id", "valence", "arousal"]].progress_apply(generate_arousal_valence_dict, axis=1)

# sove valence e arousal
print("Salvataggio valence e arousal...")
pickle.dump(arousal_valence_dict, open('CODICE PAPER+DATASET/arousal_valence_dict.p', 'wb'))
print("arousal_valence_dict.p salvato.\n")


# Concatenate features

def generate_embeddings(row, e_dict):
    embedding = np.concatenate((context_embeddings[row[0]], content_embeddings[row[0]]))
    embedding = np.append(embedding, arousal_valence_dict[row[0]]["arousal"])
    embedding = np.append(embedding, arousal_valence_dict[row[0]]["valence"])
    e_dict[row[0]] = embedding


# creo embedding canzoni
song_embeddings = {}
song_data[["song_id"]].apply(lambda x: generate_embeddings(x, song_embeddings), axis=1)

print("Creati  " + str(len(song_embeddings)) + " song embedding.")
print(song_embeddings["SOZOBWN12A8C130999"])

# %%

# creo embedding utenti
user_embeddings = {}
user_hyperedges[["user_id"]].progress_apply(lambda x: generate_embeddings(x, user_embeddings), axis=1)

print("Creati  " + str(len(user_embeddings)) + " user embedding.")

# %%

# salvataggio song embeddings
print("Salvataggio song embeddings...")
pickle.dump(song_embeddings, open('song_embeddings.p', 'wb'))
print("song_embeddings.p salvato.\n")

# salvataggio user embeddings
print("Salvataggio user embeddings...")
pickle.dump(user_embeddings, open('user_embeddings.p', 'wb'))
print("user_embeddings.p salvato.\n")

# %% md

# Normalizzazione embeddings (Z-Score)

# %%

import numpy as np
from scipy.stats import zscore

song_keys, song_vals = zip(*song_embeddings.items())
song_embeddings = dict(zip(song_keys, zscore(song_vals, ddof=1)))

user_keys, user_vals = zip(*user_embeddings.items())
user_embeddings = dict(zip(user_keys, zscore(user_vals, ddof=1)))

# %%

# salvataggio song embeddings con z-score
print("Salvataggio song embeddings...")
pickle.dump(song_embeddings, open('song_embeddings_zscore.p', 'wb'))
print("song_embeddings_zscore.p salvato.\n")

# salvataggio user embeddings con z-score
print("Salvataggio user embeddings...")
pickle.dump(user_embeddings, open('user_embeddings_zscore.p', 'wb'))
print("user_embeddings_zscore.p salvato.\n")

# Recommendation system (distanza coseno pesata)


from numpy import dot
from numpy.linalg import norm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

pd.options.display.max_rows = None


# cos sim

def cos_sim(a, b, w):
    return 1 - spatial.distance.cosine(a, b, w)
    # return dot(a, b)/(norm(a)*norm(b))


# weights vector
w_1 = np.full(16, 1 / (16 * 3))  # peso context-embedding
w_2 = np.full(25, 1 / (25 * 3))  # peso content-embedding
w_3 = np.full(2, 1 / (2 * 3))  # peso arousal e valence

# vettore peso w (stesso peso ad ogni sistema)
w_equi = np.concatenate((w_1, w_2, w_3), axis=None)

# vettore peso w_context (solo context-based)
w_context = np.concatenate((np.ones(200), np.zeros(25), np.zeros(2)), axis=None)

# vettore peso w_content (solo context-based)
w_content = np.concatenate((np.zeros(16), np.ones(50), np.zeros(2)), axis=None)

# vettore peso w_av (solo context-based)
w_av = np.concatenate((np.zeros(16), np.zeros(25), np.ones(2)), axis=None)

w_1_mix = np.full(16, 5 / (16 * 3))  # peso context-embedding
w_2_mix = np.full(100, 1 / (50 * 3))  # peso content-embedding
w_3_mix = np.full(2, 0 / (2 * 3))  # peso arousal e valence
w_mix = np.concatenate((w_1_mix, w_2_mix, w_3_mix), axis=None)

# remove 0 lisetnings

listened_songs = user_data_total["song_id"].drop_duplicates()
song_data_listened = song_data[song_data["song_id"].isin(listened_songs)].reset_index()

# use song unique id for matrix
song_data_listened["id"] = song_data_listened.index
user_data_total["song_matrix_id"] = \
    user_data_total.merge(song_data_listened, left_on="song_id", right_on="song_id", how="left")["id"]

# caricamento dati set (per utenti su cui fare la predizione)

#  test set
user_test = user_data_total[user_data_total["set"] == "test"]
user_test = user_test.merge(user_hyperedges, on="user", how="left").set_index(user_test.columns.values[0])[
    ["user", "user_id", "user_id_matrix", "song_id", "play_count", "triple_id", "set", 'song_matrix_id']]
users_test_songs = user_test.groupby('user_id_matrix')['song_matrix_id'].apply(list).reset_index(name="songs")

# training degli utenti nel test set
user_training = user_data_total[user_data_total["set"] == "train"]
user_training = user_training.merge(user_hyperedges, on="user", how="left").set_index(user_training.columns.values[0])[
    ["user", "user_id", "user_id_matrix", "song_id", "play_count", "triple_id", "set", 'song_matrix_id']]
user_training = user_training[user_training["user_id_matrix"].isin(users_test_songs["user_id_matrix"])]
users_training_songs = user_training.groupby('user_id_matrix')['song_matrix_id'].apply(list).reset_index(name="songs")

# test_user_list = user_test.user_id.unique()
# test_song_list = user_test.song_id.unique()
# print(user_training)
# print(user_test)
# print(users_test_songs)


# compute matrix user/song test set
users_test_matrix = users_test_songs['songs'].tolist()

import scipy.sparse as sp

matrix = sp.lil_matrix((len(users_test_matrix), len(song_data_listened)), dtype=int)
for row in range(len(users_test_matrix)):
    for column in users_test_matrix[row]:
        matrix[row, column] = 1

users_test_matrix = matrix

# %%

# compute matrix user/song (only test set users)
users_train_matrix = users_training_songs['songs'].tolist()

import scipy.sparse as sp

matrix = sp.lil_matrix((len(users_train_matrix), len(song_data_listened)), dtype=int)
for row in range(len(users_train_matrix)):
    for column in users_train_matrix[row]:
        matrix[row, column] = 1

users_train_matrix = matrix

# user_embeddings_to_array
user_hyperedges['embedding'] = user_hyperedges.user_id.map(user_embeddings)
users_test_songs['user_embedding'] = users_test_songs.merge(user_hyperedges, on="user_id_matrix", how="left")[
    "embedding"]
user_embeddings_array = users_test_songs['user_embedding'].tolist()
user_embeddings_array = np.array(user_embeddings_array)

# song_embeddings to array
song_data_listened['embedding'] = song_data_listened.song_id.map(song_embeddings)
song_embeddings_array = song_data_listened['embedding'].tolist()
song_embeddings_array = np.array(song_embeddings_array)


# compute scores and top-k prediction for each user

def compute_score_matrix(user_embeddings_array, song_embeddings_array, w):
    num = (user_embeddings_array * w) @ song_embeddings_array.T
    norms_user = ((user_embeddings_array * w) * user_embeddings_array).sum(1, keepdims=True) ** .5
    norms_song = ((song_embeddings_array * w) * song_embeddings_array).sum(1, keepdims=True) ** .5
    den = norms_user @ norms_song.T
    users_score_matrix = num / den
    return users_score_matrix


print(str(datetime.datetime.now()) + " - Start")

i = 0
k = 100
step = 10000
users_predictions = np.zeros((len(user_embeddings_array), k), dtype=int)

while (i < len(user_embeddings_array)):
    j = min(i + step, len(user_embeddings_array))

    # calcola score
    print(str(datetime.datetime.now()) + " - Calcolo scores " + str(i) + "-" + str(j))
    users_score_matrix = compute_score_matrix(user_embeddings_array[i:j], song_embeddings_array, w_context)

    # calcola top k suggestions
    print(str(datetime.datetime.now()) + " - Rimuovo canzoni di training " + str(i) + "-" + str(j))
    users_score_matrix = users_score_matrix - users_train_matrix[
                                              i:j] * 2  # peggioro lo score delle canzoni di train per non predirle
    print(str(datetime.datetime.now()) + " - Calcolo top k suggestions con argsort " + str(i) + "-" + str(j))
    users_predictions[i:j] = np.argsort(-users_score_matrix)[:, 0:k]
    # print(str(datetime.datetime.now())+" - Calcolo top k suggestions con argpartition "+str(i)+"-"+str(j))
    # users_predictions[i:j] = np.argpartition(-users_score_matrix, range(k))[:,0:k] # ordino in base allo score e restituisco le prime k

    i = j

print(str(datetime.datetime.now()) + " - Fine")

# top_k_songs = np.argpartition(users_score_matrix, -k)[:,-k:]
# for top in range(0,k):
#    print(top_k_songs[:,top].flatten())
#    users_pred_matrix[i:(i+u), top_k_songs[:,top].flatten()] = 1
# print(top_k_songs[0])
# for u in range(0, step):
# users_score_matrix[u,users_train_matrix[i+u]] = -1     # dai score minimo alle canzoni di test (per non predirle)
# top_k_songs = np.argsort(users_score_matrix[u])[0:k]   # ottieni le top k canzoni con score migliore
# users_pred_matrix[i+u,top_k_songs] = 1                 # segnale nella matrice di predizione
# if u%100==0: print(u)

# %%

# calcolo predictions per un utente
user_id = random.randint(0, users_predictions.shape[0] - 1)
print(user_id)

# canzoni ascoltate dall'utente
song_data_listened[song_data_listened.id.isin(users_train_matrix.rows[user_id])]

# top 100 consigli per l'utente
predictions_df = pd.DataFrame({'id': users_predictions[user_id]})
predictions_df["rank"] = predictions_df.index + 1
predictions_df.merge(song_data_listened, on='id', how='left').set_index('rank')

# canzoni del test set dell'utente
song_data_listened[song_data_listened['id'].isin(users_test_matrix.rows[user_id])]


# Evaluation con l'utilizzo di MAP e RECALL


def check_true_positives(songs_test, top_songs):
    #  controlla quante canzoni del test set compaiono
    found = 0
    for test_song in songs_test:
        if test_song in top_songs:
            found += 1
    return found


def AP(songs_test, top_songs):
    # average precision
    found = 0
    score = 0
    for i in range(len(songs_test)):
        if songs_test[i] in top_songs:
            found += 1
            score += found / (i + 1)
    return score / min(len(songs_test), len(top_songs))


users_test_matrix.shape

# compute recall and AP @100/50/20/10/5
print(str(datetime.datetime.now()) + " - Start evaluation")

for k in [100, 50, 20, 10, 5]:

    true_positives = np.zeros(users_test_matrix.shape[0])
    user_recalls = np.zeros(users_test_matrix.shape[0])
    user_APS = np.zeros(users_test_matrix.shape[0])
    test_songs_n = 0

    for i in range(users_test_matrix.shape[0]):
        test_songs = users_test_matrix.rows[i]
        test_songs_n += len(test_songs)
        predictions = users_predictions[i]

        true_positives[i] = check_true_positives(test_songs, predictions[0:k])

        user_recalls[i] = float(true_positives[i] / (len(test_songs)))

        user_APS[i] = AP(test_songs, predictions[0:k])

        # print("["+str(i)+"]: "+str(user_recalls[i]*100)+"%")

    macro_recall = np.mean(user_recalls)
    print("Macro-avg Recall@" + str(k) + " (per utente): " + str(macro_recall * 100) + "%")
    # micro_recall = np.sum(true_positives)/test_songs_n
    # print("Micro-avg Recall@"+str(k)+" (per canzone): "+str(micro_recall*100)+"%")
    avg_ap = np.mean(user_APS)
    print("AP@" + str(k) + ": " + str(avg_ap * 100) + "%")

print(str(datetime.datetime.now()) + " - End evaluation")
