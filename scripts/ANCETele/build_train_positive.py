# A quick script to strip out the first occurance line of each unique qid(first item in line)
# This is to build the list of <query_id>-<train_positive_sample> to be used for generating "train-positive" neighbor negatives
#(a.k.a Tele-negatives)
#It would be terribly slow to use awk for this, so you have this python script instead.
from argparse import ArgumentParser
import os
import tqdm
parser = ArgumentParser()
parser.add_argument('--qrel_file', required=True)
parser.add_argument("--save_to",required=True)
parser.add_argument("--corpus_file",required=True)
parser.add_argument("--save_name",default="train.positive.txt")

args = parser.parse_args()
os.makedirs(args.save_to, exist_ok=True)
encountered_qid = set()
corpus_dict ={}
print(f"Reading corpus texts at {args.corpus_file}...")
with open(args.corpus_file,"r") as corpus_fin:
    corpus_lines = corpus_fin.readlines()
    for line in tqdm.tqdm(corpus_lines):
        pid,data = line.strip().split("\t",maxsplit=1)
        corpus_dict[int(pid)]=data
print(f"Reading qrels at {args.qrel_file}...")
with open(args.qrel_file,"r") as fin:
    lines = fin.readlines()
print("Processing and saving to {}...".format(os.path.join(args.save_to,args.save_name)))   
with open(os.path.join(args.save_to,args.save_name),"w") as fout:
    for line in tqdm.tqdm(lines):
        #print(line)
        qid,_,doc,_ = line.split("\t")
        if qid not in encountered_qid:
            encountered_qid.add(qid)
            fout.write(qid+"\t"+corpus_dict[int(doc)]+"\n")

        