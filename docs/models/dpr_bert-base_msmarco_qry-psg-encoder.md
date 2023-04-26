# DPR (BERT-base, MS MARCO)


This model is DPR trained on MS MARCO [[HF Download Link]](https://huggingface.co/OpenMatch/dpr_bert-base_msmarco_qry-psg-encoder). The training details and evaluation results are as follows:

|Model|Pretrain Model|Train w/ Marco Title|Marco Dev MRR@10|BEIR Avg NDCG@10|
|:----|:----|:----|:----|:----|
|DPR|bert-base-uncased|w/|32.4|35.5|

|BERI Dataset|NDCG@10|
|:----|:----|
|TREC-COVID|58.8|
|NFCorpus|23.4| 
|FiQA|20.6| 
|ArguAna|39.4| 
|Touché-2020|22.3| 
|Quora|78.0| 
|SCIDOCS|11.9| 
|SciFact|49.4| 
|NQ|43.9| 
|HotpotQA|45.3| 
|Signal-1M|20.2| 
|TREC-NEWS|31.8| 
|DBPedia-entity|28.7|
|Fever|65.0| 
|Climate-Fever|14.9|
|BioASQ|24.1|
|Robust04|32.3|
|CQADupStack|28.3|
