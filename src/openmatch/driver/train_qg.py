# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys
from functools import partial

import evaluate
from transformers import (AutoConfig, AutoTokenizer, DefaultDataCollator,
                          HfArgumentParser, Seq2SeqTrainer, set_seed)
from transformers.trainer_utils import EvalPrediction
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration

from openmatch.arguments import DataArguments
from openmatch.arguments import QGTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import MappingQGTrainDataset, StreamQGTrainDataset

logger = logging.getLogger(__name__)


def compute_metrics(evaluator, tokenizer: PreTrainedTokenizer, eval_output: EvalPrediction):
    predictions = tokenizer.batch_decode(eval_output.predictions, remove_special_tokens=True)
    label_ids = tokenizer.batch_decode(eval_output.label_ids, remove_special_tokens=True)
    results = evaluator.compute(predictions=predictions, references=label_ids)
    return results


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset_cls = MappingQGTrainDataset if training_args.use_mapping_dataset else StreamQGTrainDataset
    train_dataset = train_dataset_cls(
        tokenizer, 
        data_args, 
        shuffle_seed=training_args.seed, 
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    )
    eval_dataset = train_dataset_cls(
        tokenizer, 
        data_args, 
        is_eval=True, 
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    ) if data_args.eval_path is not None else None

    rouge = evaluate.load("rouge")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=partial(compute_metrics, rouge, tokenizer),
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
