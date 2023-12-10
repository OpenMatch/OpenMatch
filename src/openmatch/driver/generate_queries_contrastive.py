import copy
import logging
import os
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    T5ForConditionalGeneration,
)

from openmatch.arguments import DataArguments, ModelArguments, QGInferenceArguments
from openmatch.dataset import InferenceDataset
from openmatch.retriever import ContrastiveQueryGenerator
from openmatch.utils import load_from_trec

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, QGInferenceArguments))
    parser.add_argument("--pos_doc_template", type=str, default="Positive: <text>")
    parser.add_argument("--neg_doc_template", type=str, default="Negative: <text>")
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, inference_args, other_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, inference_args, other_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        inference_args: QGInferenceArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if inference_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        inference_args.local_rank,
        inference_args.device,
        inference_args.n_gpu,
        bool(inference_args.local_rank != -1),
        inference_args.fp16,
    )
    logger.info("Encoding parameters %s", inference_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    try:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    except OSError:
        config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus_dataset_pos = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        template=other_args.pos_doc_template,
        full_tokenization=False,
        is_query=False,
        stream=False,
        cache_dir=model_args.cache_dir,
    )
    corpus_dataset_neg = copy.copy(corpus_dataset_pos)
    corpus_dataset_neg.template = other_args.neg_doc_template

    run = load_from_trec(inference_args.trec_run_path)

    generator = ContrastiveQueryGenerator(
        model, tokenizer, corpus_dataset_pos, corpus_dataset_neg, inference_args, data_args
    )
    ori_qids, pos_docids, neg_docids, generated_queries = generator.generate(run)
    with open(inference_args.queries_save_path, "w") as f_queries, open(
        inference_args.qrels_save_path, "w"
    ) as f_qrels:
        for ori_qid, pos_docid, neg_docid, queries in zip(
            ori_qids, pos_docids, neg_docids, generated_queries
        ):
            for qid, q in enumerate(queries):
                if len(q) < 10:
                    continue
                f_queries.write(
                    f"original_q_{ori_qid}_generated_q_{qid}_for_pos_{pos_docid}_neg_{neg_docid}\t{q}\n"
                )
                f_qrels.write(
                    f"original_q_{ori_qid}_generated_q_{qid}_for_pos_{pos_docid}_neg_{neg_docid}\t0\t{pos_docid}\t1\n"
                )
                f_qrels.write(
                    f"original_q_{ori_qid}_generated_q_{qid}_for_pos_{pos_docid}_neg_{neg_docid}\t0\t{neg_docid}\t0\n"
                )


if __name__ == "__main__":
    main()
