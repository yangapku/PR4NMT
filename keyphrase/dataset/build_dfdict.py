from __future__ import print_function
import pickle
import os.path
from nltk.stem.porter import *

data_dir = "dataset/keyphrase/punctuation-20000validation-20000testing/"
df_dict_unstem_fn = "df_dict_unstem.pkl"
df_dict_stem_fn = "df_dict_stem.pkl"
df_dict_unkstem_fn = "df_dict_unkstem.pkl"

fopen = open(os.path.join(data_dir, "all_600k_dataset.pkl"), 'rb')
train_set, validation_set, test_sets, idx2word, word2idx = pickle.load(fopen)
print("Loaded trainset and vocabulary.")

stemmer = PorterStemmer()
def stem_with_try(stemmer, word):
    try: # stem some words will raise exception
        result = stemmer.stem(word)
    except:
        result = word
    finally:
        return result
voc_size = 50000

df_dict_unstem = dict()
for id in range(2, voc_size): # not include <eol> and <unk>
    df_dict_unstem[id] = 0
stemmed_wordset = set()
for id in range(2, voc_size):
    stemmed_wordset.add(stem_with_try(stemmer, idx2word[id]))
df_dict_stem, df_dict_unkstem = dict(), dict()
for stemmed_word in stemmed_wordset:
    df_dict_stem[stemmed_word] = 0
    df_dict_unkstem[stemmed_word] = 0
print("Initialized DF dictionaries.")

n_doc = len(train_set['source'])
for i in range(n_doc):
    if i % 10000 == 0 and i > 0:
        print("Summarizaing doc %d." % i)
    wordset_unstem = set(train_set['source'][i])
    for wordid in wordset_unstem:
        if wordid in df_dict_unstem:
            df_dict_unstem[wordid] += 1
    wordset_stem = set([stem_with_try(stemmer, idx2word[wordid]) for wordid in wordset_unstem])
    for word in wordset_stem:
        if word in df_dict_stem:
            df_dict_stem[word] += 1
    doc_withoutunk = filter(lambda x: x < voc_size, wordset_unstem)
    wordset_unkstem = set([stem_with_try(stemmer, idx2word[wordid]) for wordid in doc_withoutunk])
    for word in wordset_unkstem:
        if word in df_dict_unkstem:
            df_dict_unkstem[word] += 1
print("Calculated DF for words in vocabulary.")

# pickle into binary file
for df, fn in zip([df_dict_unstem, df_dict_stem, df_dict_unkstem], [df_dict_unstem_fn, df_dict_stem_fn, df_dict_unkstem_fn]):
    fout = open(os.path.join(data_dir, fn), 'wb')
    pickle.dump(df, fout)
    fout.close()
print("DF dict binary files were saved.")
