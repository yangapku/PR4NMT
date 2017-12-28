from __future__ import print_function
import pickle
from nltk import pos_tag

fp = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/all_600k_dataset.pkl", "rb")
a = pickle.load(fp)

goldens = a[0]['target']
idx2word = a[-2]
print("Loaded in pickle format training dataset!")

pos_tag_seq = dict()

for i in range(len(goldens)):
    doc = goldens[i]
    if i % 100 == 0:
        print("Analysing document %d ..." % i)
    for j in range(len(doc)):
        phrase = doc[j]
        if len(phrase) > 0 and len(phrase) <= 6:
            words = [idx2word[wordid] for wordid in phrase]
            tags = pos_tag(words)
            tags_str = "_".join([word_tag_tuple[1] for word_tag_tuple in tags])
            if tags_str not in pos_tag_seq:
                pos_tag_seq[tags_str] = 1
            else:
                pos_tag_seq[tags_str] += 1
print("Stated pos_tag sequence pattern!")

threshold = 5
pos_tag_seq_list = list()
for item in pos_tag_seq:
    if pos_tag_seq[item] > threshold:
        pos_tag_seq_list.append(item)

fout = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/postag_seq_dict.pkl", "wb")
pickle.dump(pos_tag_seq_list, fout)
print("Finished storing frequent pos_tag sequence pattern.")