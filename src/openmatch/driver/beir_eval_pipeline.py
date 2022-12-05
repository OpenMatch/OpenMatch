import logging
import os
import sys

import pytrec_eval
from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import BEIRDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever
from openmatch.utils import save_as_trec
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    if (
        os.path.exists(encoding_args.output_dir)
        and os.listdir(encoding_args.output_dir)
    ):
        if not encoding_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = DRModelForInference.build(
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    beir_dataset = BEIRDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
        cache_dir=model_args.cache_dir
    )

    retriever = Retriever.build_all(model, beir_dataset.corpus_dataset, encoding_args)
    
    if encoding_args.use_split_search:
        retriever = Retriever.build_embeddings(model, beir_dataset.corpus_dataset, encoding_args)
        run = retriever.split_retrieve(beir_dataset.query_datasets["test"], topk=encoding_args.retrieve_depth)
    else:
        retriever = Retriever.build_all(model, beir_dataset.corpus_dataset, encoding_args)
        run = retriever.retrieve(beir_dataset.query_datasets["test"], topk=encoding_args.retrieve_depth)

    if encoding_args.local_process_index == 0:
            
        # Save trec file
        if encoding_args.trec_save_path is None:
            encoding_args.trec_save_path = os.path.join(encoding_args.output_dir, "test.trec")
        save_as_trec(run, encoding_args.trec_save_path)
        
        # compute metric
        evaluator = pytrec_eval.RelevanceEvaluator(
        beir_dataset.qrels["test"], {'ndcg_cut.10'})
        eval_results = evaluator.evaluate(run)

        def print_line(measure, scope, value):
            print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))
            with open(os.path.join(encoding_args.output_dir, "test_result.log"), "w", encoding="utf-8") as fw:
                fw.write('{:25s}{:8s}{:.4f}\n'.format(measure, scope, value))

        for query_id, query_measures in sorted(eval_results.items()):
            for measure, value in sorted(query_measures.items()):
                pass

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        for measure in sorted(query_measures.keys()):
            print_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                    for query_measures in eval_results.values()]))



if __name__ == '__main__':
    main()
