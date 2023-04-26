# ANCE-Tele (cocodr-base, MS MARCO)


This model is ANCE-Tele trained on MS MARCO [[HF Download Link]](https://huggingface.co/OpenMatch/ance-tele_coco-base_msmarco_qry-psg-encoder). The training details and evaluation results are as follows:

|Model|Pretrain Model|Train w/ Marco Title|Marco Dev MRR@10|BEIR Avg NDCG@10|
|:----|:----|:----|:----|:----|
|ANCE-Tele|[cocodr-base](https://huggingface.co/OpenMatch/cocodr-base)|w/o|37.3|44.2|

|BERI Dataset|NDCG@10|
|:----|:----|
|TREC-COVID|77.4|
|NFCorpus|34.4 | 
|FiQA|29.0 | 
|ArguAna|45.6 | 
|Touché-2020|22.3 | 
|Quora|85.8 | 
|SCIDOCS|14.6 | 
|SciFact|71.0 | 
|NQ|50.5 | 
|HotpotQA|58.8 | 
|Signal-1M|27.2 | 
|TREC-NEWS|34.7 | 
|DBPedia-entity|36.2 |
|Fever|71.4 | 
|Climate-Fever|17.9 |
|BioASQ|42.1 |
|Robust04|41.4 |
|CQADupStack|34.9 |
