from __future__ import print_function, division
import pickle
import numpy as np
from nltk.stem.porter import *
from collections import Counter

class Feature(object):
    def __init__(self):
        self.n_doc = 527830 # size of trainset documents
        pass
    
    def get_feature(self, source, cands, goldens=None):
        raise NotImplementedError
    
class TfidfFeature(Feature):
    def __init__(self, idx2word, stem=True, filter_unk=False): 
        # filter_unk incicates whether consider unk word when summarizing tf dict, only work when stem=True
        super(TfidfFeature, self).__init__()
        self.stem = stem # indicate whether to do stemming
        # load pre-calculated df dictionary
        if self.stem:
            if filter_unk:
                fin = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/df_dict_unkstem.pkl", 'rb')
            else:
                fin = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/df_dict_stem.pkl", 'rb')
        else:
            fin = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/df_dict_unstem.pkl", 'rb')
        self.df = pickle.load(fin)
        self.idx2word = idx2word
        if self.stem:
            self.stemmer = PorterStemmer()

    def stem_with_try(self, word):
        try: # stem some words will raise exception
            result = self.stemmer.stem(word)
        except:
            result = word
        finally:
            return result 

    def calc_df(self, wordid):
        if self.stem:
            raw_word = self.idx2word[wordid]
            try:
                word = self.stem_with_try(raw_word)
            except:
                word = raw_word
            finally:
                return self.df[word]
        return self.df[wordid]

    def count_source(self, source):
        source_text = [self.stem_with_try(self.idx2word[wordid]) if self.stem 
                        else self.idx2word[wordid] 
                        for wordid in source]
        source_counter = Counter(source_text)
        return source_counter

    def calc_tf(self, wordid, source_counter):
        raw_word = self.idx2word[wordid]
        word = self.stem_with_try(raw_word) if self.stem else raw_word
        return source_counter[word] if word in source_counter else 0

    def get_feature(self, source, cands, goldens=None):
        n_cand = len(cands)
        n_golden = len(goldens) if goldens is not None else 0
        feature = np.zeros((n_cand + n_golden, 1), dtype="float32")
        source_counter = self.count_source(source)
        for idx in range(n_cand):
            tfidf = 0.
            for word in cands[idx]:
                if word > 1: # ignore <eol> and <unk>
                    tfidf += -1. * self.calc_tf(word, source_counter) * np.log2((1.0 + self.calc_df(word)) / (1. * self.n_doc))
            feature[idx, 0] = tfidf
        if goldens is not None:
            for idx in range(n_golden):
                tfidf = 0.
                for word in goldens[idx]:
                    if word > 1: # ignore <eol> and <unk>
                        tfidf += -1. * self.calc_tf(word, source_counter) * np.log2((1.0 + self.calc_df(word)) / (1. * self.n_doc))
                feature[idx + n_cand, 0] = tfidf
        return feature

class LengthFeature(Feature):
    def __init__(self, max_length=6):
        # (1 + max_length) 0-1 features indicating phrase with correspoding length (0~max_len), phrases longer than max_len are regarded as max_len
        # there is a feature for empty phrase, which may be the case sometimes
        super(LengthFeature, self).__init__()
        self.max_length = max_length
    
    def get_feature(self, source, cands, goldens=None):
        n_cand = len(cands)
        n_golden = len(goldens) if goldens is not None else 0
        feature = np.zeros((n_cand + n_golden, self.max_length + 1), dtype="float32")
        l_cands = map(lambda x:len(x) if len(x) <= self.max_length else self.max_length, cands)
        if goldens is not None:
            l_goldens = map(lambda x:len(x)  if len(x) <= self.max_length else self.max_length, goldens)
            feature[range(n_cand + n_golden), l_cands + l_goldens] = 1.
        else:
            feature[range(n_cand), l_cands] = 1.
        return feature

class KeyphrasenessFeature(Feature):
    def __init__(self, idx2word, stem=True):
        super(KeyphrasenessFeature, self).__init__()
        self.stem = stem
        if self.stem:
            fin = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/kpn_dict_stem.pkl", 'rb')
        else:
            fin = open("/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/keyphrase/punctuation-20000validation-20000testing/kpn_dict_unstem.pkl", 'rb')
        self.phrase_counter = pickle.load(fin)
        self.idx2word = idx2word
        if self.stem:
            self.stemmer = PorterStemmer()
    
    def stem_with_try(self, word):
        try: # stem some words will raise exception
            result = self.stemmer.stem(word)
        except:
            result = word
        finally:
            return result    

    def cal_kpn(self, phrase): # input is list of wordid
        if self.stem:
            phrase_str = " ".join([self.stem_with_try(self.idx2word[wordid]) for wordid in phrase])
        else:
            phrase_str = " ".join([self.idx2word[wordid] for wordid in phrase])
        return 1. * self.phrase_counter[phrase_str] if phrase_str in self.phrase_counter else 0.

    def get_feature(self, source, cands, goldens=None):
        n_cand = len(cands)
        n_golden = len(goldens) if goldens is not None else 0
        feature = np.zeros((n_cand + n_golden, 1), dtype="float32")
        feature[:n_cand] = np.asarray(map(lambda x:self.cal_kpn(x), cands)).reshape((n_cand, 1))
        if goldens is not None:
            feature[n_cand:] = np.asarray(map(lambda x:self.cal_kpn(x), goldens)).reshape((n_golden, 1))
        return feature

class StopwordFeature(Feature):
    def __init__(self, word2idx):
        super(StopwordFeature, self).__init__()
        stopword_file = open('/home/yangan/projects/keywords/seq2seq-keyphrase/dataset/stopword/stopword_en.txt', "r")
        stopword_set = set([w.strip() for w in stopword_file])
        self.stopword_idxset = set()
        for word in stopword_set:
            if word in word2idx:
                self.stopword_idxset.add(word)
    
    def has_stopword(self, phrase):
        return np.any(map(lambda x: x in self.stopword_idxset, phrase))

    def get_feature(self, source, cands, goldens=None):
        # feature indicating whether phrase contains stopwords
        n_cand = len(cands)
        n_golden = len(goldens) if goldens is not None else 0
        feature = np.zeros((n_cand + n_golden, 1), dtype="float32")
        feature[:n_cand] = np.asarray(map(lambda x:self.has_stopword(x), cands), dtype="float32").reshape((n_cand, 1))
        if goldens is not None:
            feature[n_cand:] = np.asarray(map(lambda x:self.has_stopword(x), goldens), dtype="float32").reshape((n_golden, 1))
        return feature

class PositionFeature(Feature):
    def __init__(self, n_level = 5):
        self.n_level = n_level

    def cal_pos(self, phrase, source):
        feat = np.zeros(self.n_level + 1)
        # transform into string and find occurence, a little ugly...
        phrase_id_str = " ".join([str(id) for id in phrase])
        source_id_str = " ".join([str(id) for id in source])
        pos = source_id_str.find(phrase_id_str)
        if pos == -1: # phrase does not appear in source
            feat[-1] = 1.
        else:
            offset = source_id_str[:pos].count(" ")
            offset_level = int(offset / len(source) * self.n_level)
            feat[offset_level] = 1.
        return feat

    def get_feature(self, source, cands, goldens=None):
        # (n_level + 1) features indicating whether and where the phrase appears in source text
        n_cand = len(cands)
        n_golden = len(goldens) if goldens is not None else 0
        feature = np.zeros((n_cand + n_golden, self.n_level + 1), dtype="float32")
        feature[:n_cand] = np.asarray(map(lambda x:self.cal_pos(x, source), cands), dtype="float32")
        if goldens is not None:
            feature[n_cand:] = np.asarray(map(lambda x:self.cal_pos(x, source), goldens), dtype="float32")
        return feature