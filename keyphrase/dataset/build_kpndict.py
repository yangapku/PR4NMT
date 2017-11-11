from __future__ import print_function
import pickle
import os.path
from nltk.stem.porter import *

stemmer = PorterStemmer()
def stem_with_try(stemmer, word):
    try: # stem some words will raise exception
        result = stemmer.stem(word)
    except:
        result = word
    finally:
        return result

data_dir = "dataset/keyphrase/punctuation-20000validation-20000testing/"
keyphrase_dict_stem_fn = "kpn_dict_stem.pkl"
keyphrase_dict_unstem_fn = "kpn_dict_unstem.pkl"

fopen = open(os.path.join(data_dir, "all_600k_dataset.pkl"), 'rb')
train_set, validation_set, test_sets, idx2word, word2idx = pickle.load(fopen)
print("Loaded trainset and vocabulary.")

keyphrase_dict_stem, keyphrase_dict_unstem = dict(), dict()
n_doc = len(train_set['target'])

for docid in range(n_doc):
    if docid % 10000 == 0 and docid > 0:
        print("Summarizaing doc %d." % docid)
    for keyphrase in train_set['target'][docid]:
        words_stem = [stem_with_try(stemmer, idx2word[wordid]) for wordid in keyphrase]
        words_unstem = [idx2word[wordid] for wordid in keyphrase]
        for s, d in zip([" ".join(words_stem), " ".join(words_stem)], [keyphrase_dict_stem, keyphrase_dict_unstem]):
            if s in d:
                d[s] += 1
            else:
                d[s] = 1

for fn, d in zip([keyphrase_dict_stem_fn, keyphrase_dict_unstem_fn], [keyphrase_dict_stem, keyphrase_dict_unstem]):
    fout = open(os.path.join(data_dir, fn), "wb")
    pickle.dump(d, fout)
    fout.close()
print("Keyphraseness dict binary files were saved.")        