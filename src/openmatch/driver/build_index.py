import logging
import os
import sys

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever
from openmatch.utils import get_delta_model_class
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoProcessor

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
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    except ValueError:
        processor = None

    model = DRModelForInference.build(
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if model_args.param_efficient_method:
        model_class = get_delta_model_class(model_args.param_efficient_method)
        delta_model = model_class.from_finetuned(model_args.model_name_or_path + '/delta_model', model, local_files_only=True)
        logger.info("Using param efficient method: %s", model_args.param_efficient_method)

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
        is_query=False,
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
        cache_dir=model_args.cache_dir
    )

    Retriever.build_embeddings(model, corpus_dataset, encoding_args)


if __name__ == '__main__':
    main()
