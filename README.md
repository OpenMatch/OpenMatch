# OpenMatch v2

OpenMatch v2 is an all-in-one toolkit for information retrieval (IR) currently under active development. It supports training and evaluation of various dense retrievers and re-rankers with deep integration of Huggingface Transformers and Datasets.

## Features

- Human-friendly interface for dense retriever and re-ranker training and testing
- Various PLMs supported (BERT, RoBERTa, T5...)
- Native support for common IR & QA Datasets (MS MARCO, NQ, KILT, BEIR, ...)
- Deep integration with Huggingface Transformers and Datasets
- Efficient training and inference via stream-style data loading

## Installation

To install OpenMatch V2, follow these steps:
```bash
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```

Note: `-e` means **editable**, i.e. you can change the code directly in your directory.

### Dependency Management

We do not include all the requirements in the package. You may need to manually install some dependencies based on your environment:

• `torch` and `tensorboard` for model training and visualization.
Install with:
```bash
pip install torch tensorboard
```

• `faiss` for dense retrieval. Choose between `faiss-cpu` or `faiss-gpu` depending on your system. Make sure that the correct version of `faiss-gpu` is installed for your CUDA environment.
Install faiss with:
```bash
conda install faiss-cpu -c pytorch
# or
conda install faiss-gpu -c pytorch
```

Note: If you encounter GPU search errors (especially with CUDA >= 11.0), you may need to install `faiss-gpu` manually via conda instead of `pip`.

## Quick Start Guide

This section demonstrates how to set up and run a simple retrieval task using OpenMatch v2.

### Step 1: Data Preparation

First, select a supported dataset for training and evaluation, such as MS MARCO:
```bash
wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm marco.tar.gz
```

### Step 2: Train the Model

```bash
python -m openmatch.driver.train_dr \
    --output_dir $CHECKPOINT_DIR/msmarco/t5 \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --save_steps 20000 \
    --eval_steps 20000 \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.jsonl \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.jsonl \
    --fp16 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --logging_dir $LOG_DIR/msmarco/t5 \
    --evaluation_strategy steps
```

### Step 3: Inference and Retrieval

```bash
python -m openmatch.driver.build_index \
    --output_dir $EMBEDDING_DIR/msmarco/t5 \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5 \
    --per_device_eval_batch_size 256 \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv \
    --q_max_len 32 \
    --p_max_len 128 \
    --fp16
```
```bash
python -m openmatch.driver.retrieve \
    --output_dir $RESULT_DIR/msmarco/t5 \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5 \
    --query_path $COLLECTION_DIR/marco/dev.query.txt \
    --trec_save_path $RESULT_DIR/msmarco/t5/dev.trec \
    --fp16
```

### Step 4: Evaluation

```bash
python scripts/evaluate.py \
    -m mrr.10 \  # Specify your evaluation metric (e.g., MRR@10)
    $COLLECTION_DIR/marco/qrels.dev.tsv \
    $RESULT_DIR/msmarco/t5/dev.trec
```
Note: This Quick Start Guide provides a streamlined process for setting up and training a dense retrieval model with OpenMatch v2. For more detailed instructions or advanced configurations, refer to the documentation.

## Documentation

[![Documentation Status](https://readthedocs.org/projects/openmatch/badge/?version=latest)](https://openmatch.readthedocs.io/en/latest/?badge=latest)

We are actively working on the docs.

## Project Organizers

- Zhiyuan Liu
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)
- Zhenghao Liu
  * Northeastern University
  * [Homepage](https://edwardzh.github.io/)
- Chenyan Xiong
  * Microsoft Research AI
  * [Homepage](https://www.microsoft.com/en-us/research/people/cxiong/)
- Maosong Sun
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/staff/sms/)

## Acknowledgments

Our implementation uses [Tevatron](https://github.com/texttron/tevatron) as the starting point. We thank its authors for their contributions.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Open a pull request, ensuring that your code passes all tests and follows the project’s style guidelines.

## Contact

For any inquiries, please contact yushi17@foxmail.com.
