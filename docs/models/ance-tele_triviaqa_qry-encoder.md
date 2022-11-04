# ANCE-Tele Query Encoder (TriviaQA)

This model is the **query** encoder of ANCE-Tele trained on TriviaQA, described in the EMNLP 2022 paper ["Reduce Catastrophic Forgetting of Dense Retrieval Training with Teleportation Negatives"](https://arxiv.org/pdf/2210.17167.pdf). The associated GitHub repository is available at https://github.com/OpenMatch/ANCE-Tele.

ANCE-Tele only trains with self-mined negatives (teleportation negatives) without using additional negatives (e.g., BM25, other DR systems) and eliminates the dependency on filtering strategies and distillation modules.


|NQ (Test)|R@5|R@20|R@20|
|:---|:---|:---|:---|
|ANCE-Tele|76.9|83.4|87.3|


```
@inproceedings{sun2022ancetele,
  title={Reduce Catastrophic Forgetting of Dense Retrieval Training with Teleportation Negatives},
  author={Si Sun, Chenyan Xiong, Yue Yu, Arnold Overwijk, Zhiyuan Liu and Jie Bao},
  booktitle={Proceedings of EMNLP 2022},
  year={2022}
}
```
