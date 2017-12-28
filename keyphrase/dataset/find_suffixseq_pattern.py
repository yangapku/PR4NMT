from __future__ import print_function
import pickle
from nltk.stem.wordnet import WordNetLemmatizer

lemmatize = True
if lemmatize:
    l = WordNetLemmatizer()

suffix_file = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/suffix/suffix.txt", "r")
suffix = [line.rstrip() for line in suffix_file.readlines()]

fp = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/all_600k_dataset.pkl", "rb")
a = pickle.load(fp)

goldens = a[0]['target']
idx2word = a[-2]
print("Loaded in pickle format training dataset and suffix list!")

suffix_seq_cnter = dict()

def get_suffix_seq(phrase):
    suffix_seq = []
    for word in phrase:
        flag = False
        for s in suffix:
            if word.endswith(s):
                suffix_seq.append(s)
                flag = True
                break
        if not flag:
            suffix_seq.append('N')
    return suffix_seq

for i in range(len(goldens)):
    doc = goldens[i]
    if i % 100 == 0:
        print("Analysing document %d ..." % i)
    for j in range(len(doc)):
        phrase = doc[j]
        if len(phrase) > 0 and len(phrase) <= 6:
            words = [idx2word[wordid] for wordid in phrase]
            if lemmatize:
                words = [l.lemmatize(word) for word in words]
            suffix_seq = get_suffix_seq(words)
            suffix_str = "_".join(suffix_seq)
            if suffix_str not in suffix_seq_cnter:
                suffix_seq_cnter[suffix_str] = 1
            else:
                suffix_seq_cnter[suffix_str] += 1
print("Stated suffix sequence pattern!")

threshold = 10
suffix_seq_list = list()
for item in suffix_seq_cnter:
    if suffix_seq_cnter[item] > threshold:
        suffix_seq_list.append(item)

fout = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/suffix_seq_dict_lem10.pkl", "wb")
pickle.dump(suffix_seq_list, fout)
print("Finished storing frequent suffix sequence pattern.")
