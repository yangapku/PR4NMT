import argparse, pickle
import os.path

parser = argparse.ArgumentParser()

parser.add_argument("-f", help="Path of input pkl file.", type=str)
parser.add_argument("-o", help="Output directory.", type=str)
parser.add_argument("-n", help="Number of keeping candidates.", type=int, default=200)

args = parser.parse_args()

fp = open(args.f, "rb")
data = pickle.load(fp)
idx2word = data[-1]
n_doc = len(data[-4])

for i in range(n_doc):
    fout = open(os.path.join(args.o, "%d.txt.phrases" % i), "w")
    n_out = min(args.n, len(data[-4][i]))
    for j in range(n_out):
            ph_l = [idx2word[wordid] for wordid in data[-4][i][j]] 
            if len(ph_l) > 0 and data[-4][i][j][-1] == 0:
                ph_l.pop()
            ph = " ".join(ph_l)
            fout.write(ph+"\n")
    fout.close()
