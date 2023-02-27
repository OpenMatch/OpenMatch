# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    feature: str = field(
        default="last_hidden_state",
        metadata={"help": "What feature to be extracted from the HF PLM"}
    )
    pooling: str = field(
        default="first",
        metadata={"help": "How to pool the features from the HF PLM"}
    )

    # out projection
    add_linear_head: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    encoder_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the encoder of T5"}
    )
    pos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a relevant document"}
    )
    neg_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a irrelevant document"}
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the embeddings"}
    )
    param_efficient_method: Optional[str] = field(
        default=None,
        metadata={"help": "Param efficient method used in model training"}
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to BEIR data directory"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    query_template: str = field(
        default=None,
        metadata={"help": "template for query"}
    )
    query_column_names: str = field(
        default=None,
        metadata={"help": "column names for the tsv data format"}
    )
    doc_template: str = field(
        default=None,
        metadata={"help": "template for doc"}
    )
    doc_column_names: str = field(
        default=None,
        metadata={"help": "column names for the tsv data format"}
    )
    all_markers: str = field(
        default=None,
        metadata={"help": "all markers in the template"}
    )

    encode_as_text_pair: bool = field(
        default=False,
        metadata={"help": "Whether to encode the query and passage as a text pair"}
    )


@dataclass
class DRTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    negatives_x_device: bool = field(default=False, metadata={
                                     "help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={
                            "help": "run the encoding loop"})
    use_mapping_dataset: bool = field(default=False, metadata={
                                      "help": "Use mapping dataset instead of iterable dataset in training"})
    grad_cache: bool = field(default=False, metadata={
                             "help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    distillation: bool = field(default=False, metadata={"help": "Use distillation"})
    distil_mode: str = field(default="pairwise", metadata={"help": "Distillation mode"})


@dataclass
class DRPretrainingDataArguments(DataArguments):
    train_n_passages: int = field(default=1)
    pretrain_strategies: str = field(
        default="crop",
        metadata={"help": "pretraining strategies"}
    )
    pretrain_target_field: str = field(
        default="text",
        metadata={"help": "pretraining target field"}
    )

    cropping_ratio_min: float = field(
        default=0.1,
        metadata={
            "help": "Minimum ratio of the cropped span to the original document span"}
    )
    cropping_ratio_max: float = field(
        default=0.5,
        metadata={
            "help": "Maximum ratio of the cropped span to the original document span"}
    )


@dataclass
class RRTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    use_mapping_dataset: bool = field(default=False, metadata={
                                      "help": "Use mapping dataset instead of iterable dataset in training"})
    margin: float = field(default=1.0)
    loss_fn: str = field(
        default="bce",
        metadata={"help": "loss function to use"}
    )


@dataclass
class QGTrainingArguments(Seq2SeqTrainingArguments):
    warmup_ratio: float = field(default=0.1)
    use_mapping_dataset: bool = field(default=False, metadata={
                                      "help": "Use mapping dataset instead of iterable dataset in training"})


@dataclass
class InferenceArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={
                          "help": "Use GPU for Faiss retrieval"})
    encoded_save_path: str = field(default=None, metadata={
                                   "help": "where to save the encode"})
    trec_save_path: str = field(default=None, metadata={
                                "help": "where to save the trec file"})
    encode_query_as_passage: bool = field(default=False, metadata={
                                          "help": "Treat input queries as passages instead of queries for encoding. Use corpus_path, doc_template and doc_column_names instead if you used this option."})
    trec_run_path: str = field(default=None, metadata={
                               "help": "previous stage TrecRun file"})
    id_key_name: str = field(default="id", metadata={
                             "help": "key name for id"})

    remove_identical: bool = field(default=False, metadata={
                                   "help": "remove identical passages"})

    reranking_depth: int = field(default=None, metadata={
                                 "help": "re-ranking depth"})
    retrieve_depth: int = field(default=100, metadata={
                                "help": "number of documents to retrieve in retriever"})

    max_inmem_docs: int = field(default=10000000, metadata={
                                "help": "max number of docs to keep in memory"})
    
    use_split_search: bool = field(
        default=False, 
        metadata={"help": "whether to split the entire corpus search into multiple sub-search jobs."}
    )


@dataclass
class QGInferenceArguments(Seq2SeqTrainingArguments):
    queries_save_path: str = field(default=None, metadata={
                                      "help": "where to save the queries"}) 
    qrels_save_path: str = field(default=None, metadata={
                                      "help": "where to save the qrels"})
    num_return_sequences: int = field(default=1, metadata={
                                        "help": "number of generated queries to return"})
    do_sample: bool = field(default=False, metadata={
                                        "help": "whether or not to use sampling; use greedy decoding otherwise"})
    top_k: int = field(default=25, metadata={
                                        "help": "top_k for beam search"})
    top_p: float = field(default=0.95, metadata={
                                        "help": "top_p for beam search"})