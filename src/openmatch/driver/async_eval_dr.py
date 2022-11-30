import glob
import logging
import os
import sys
import time

import pytrec_eval
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import BEIRDataset, InferenceDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever
from openmatch.utils import get_delta_model_class

logger = logging.getLogger(__name__)


def get_current_ckpt_step(model_args, evaluated_steps):
    all_ckpts = glob.glob(os.path.join(
        model_args.model_name_or_path, "checkpoint-*"))
    all_ckpt_steps = sorted(
        [int(os.path.basename(ckpt).split('-')[-1]) for ckpt in all_ckpts])
    for step in all_ckpt_steps:
        if step not in evaluated_steps:
            return step
    return None


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EncodingArguments))
    parser.add_argument("--beir", action="store_true")
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("-m", "--measure", type=str, default=None)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args, other_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args, other_args = parser.parse_args_into_dataclasses()
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
        level=logging.INFO if encoding_args.local_rank in [
            -1, 0] else logging.WARN,
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

    tb_writer = None
    if encoding_args.local_process_index == 0:
        tb_writer = SummaryWriter(encoding_args.logging_dir)

    if other_args.beir:
        beir_dataset = BEIRDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            stream=True,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            cache_dir=model_args.cache_dir
        )
        corpus_dataset = beir_dataset.corpus_dataset
        query_dataset = beir_dataset.query_datasets["test"]
        qrels = beir_dataset.qrels["test"]
        measure = other_args.measure or "ndcg_cut.10"
    else:
        corpus_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            is_query=False,
            stream=True,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            cache_dir=model_args.cache_dir
        )
        query_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            is_query=(not encoding_args.encode_query_as_passage),
            stream=True,
            batch_size=encoding_args.per_device_eval_batch_size,
            num_processes=encoding_args.world_size,
            process_index=encoding_args.process_index,
            cache_dir=model_args.cache_dir
        )
        with open(other_args.qrels) as f:
            qrels = pytrec_eval.parse_qrel(f)
        measure = other_args.measure or "map"

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    evaluated_steps = []
    while True:
        if encoding_args.world_size > 1:
            torch.distributed.barrier()
        cur_step = get_current_ckpt_step(model_args, evaluated_steps)
        if cur_step is not None:
            time.sleep(2)
            cur_ckpt = os.path.join(
                model_args.model_name_or_path, "checkpoint-{}".format(cur_step))
            logger.info("Loading checkpoint at step %s", cur_step)
            model = DRModelForInference.build(
                model_name_or_path=cur_ckpt,
                model_args=model_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
            if model_args.param_efficient_method:
                model_class = get_delta_model_class(
                    model_args.param_efficient_method)
                delta_model = model_class.from_finetuned(
                    cur_ckpt + '/delta_model', model, local_files_only=True)
                logger.info("Using param efficient method: %s",
                            model_args.param_efficient_method)

            retriever = Retriever.build_all(
                model, corpus_dataset, encoding_args)
            run = retriever.retrieve(query_dataset)

            if encoding_args.local_process_index == 0:
                evaluator = pytrec_eval.RelevanceEvaluator(
                    qrels, {measure})
                eval_results = evaluator.evaluate(run)

                for query_id, query_measures in sorted(eval_results.items()):
                    for measure, value in sorted(query_measures.items()):
                        pass

                # Scope hack: use query_measures of last item in previous loop to
                # figure out all unique measure names.
                #
                # TODO(cvangysel): add member to RelevanceEvaluator
                #                  with a list of measure names.
                for measure in sorted(query_measures.keys()):
                    value = pytrec_eval.compute_aggregated_measure(
                        measure, [query_measures[measure] for query_measures in eval_results.values()])
                    print_line(measure, 'all', value)
                    if tb_writer is not None:
                        tb_writer.add_scalar("eval/" + measure, value, cur_step)

            del retriever
            if encoding_args.world_size > 1:
                torch.distributed.barrier()
            evaluated_steps.append(cur_step)

        else:
            time.sleep(2)



if __name__ == '__main__':
    main()
